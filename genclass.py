###
'''
Replication of M2 from http://arxiv.org/abs/1406.5298
Title: Semi-Supervised Learning with Deep Generative Models
Authors: Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
Original Implementation (Theano): https://github.com/dpkingma/nips14-ssl
---
Code By: S. Saemundsson
'''
###
import os

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

import utils

class GenerativeClassifier(object):

    def __init__(   self,
                    dim_x, dim_z, dim_y,
                    num_examples, num_lab, num_batches,
                    p_x = 'gaussian',
                    q_z = 'gaussian_marg',
                    p_z = 'gaussian_marg',
                    hidden_layers_px = [500],
                    hidden_layers_qz = [500],
                    hidden_layers_qy = [500],
                    nonlin_px = tf.nn.softplus,
                    nonlin_qz = tf.nn.softplus,
                    nonlin_qy = tf.nn.softplus,
                    alpha = 0.1,
                    l2_loss = 0.0    ):


        self.dim_x, self.dim_z, self.dim_y = int(dim_x), int(dim_z), int(dim_y)

        self.distributions = {         'p_x':     p_x,            
                                    'q_z':     q_z,            
                                    'p_z':     p_z,            
                                    'p_y':    'uniform'    }

        self.num_examples = num_examples
        self.num_batches = num_batches
        self.num_lab = num_lab
        self.num_ulab = self.num_examples - num_lab

        assert self.num_lab % self.num_batches == 0, '#Labelled % #Batches != 0'
        assert self.num_ulab % self.num_batches == 0, '#Unlabelled % #Batches != 0'
        assert self.num_examples % self.num_batches == 0, '#Examples % #Batches != 0'

        self.batch_size = self.num_examples // self.num_batches
        self.num_lab_batch = self.num_lab // self.num_batches
        self.num_ulab_batch = self.num_ulab // self.num_batches

        self.beta = alpha * ( float(self.batch_size) / self.num_lab_batch )

        ''' Create Graph '''

        self.G = tf.Graph()

        with self.G.as_default():

            self.x_labelled_mu             = tf.placeholder( tf.float32, [None, self.dim_x] )
            self.x_labelled_lsgms         = tf.placeholder( tf.float32, [None, self.dim_x] )
            self.x_unlabelled_mu         = tf.placeholder( tf.float32, [None, self.dim_x] )
            self.x_unlabelled_lsgms     = tf.placeholder( tf.float32, [None, self.dim_x] )
            self.y_lab                  = tf.placeholder( tf.float32, [None, self.dim_y] )
            self.is_train_mode = tf.placeholder(tf.bool)

            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session()

            os.makedirs('summary', exist_ok=True)
            sub_d = len(os.listdir('summary'))
            self.train_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d))
            self.merged = tf.summary.merge_all()



    def _draw_sample( self, mu, log_sigma_sq ):

        epsilon = tf.random_normal( ( tf.shape( mu ) ), 0, 1 )
        # sample = tf.add( mu, 
        #          tf.mul(  
        #          tf.exp( 0.5 * log_sigma_sq ), epsilon ) )
        sample = mu + tf.exp(0.5*log_sigma_sq)*epsilon

        return sample

    def _generate_yx( self, x_mu, x_log_sigma_sq, reuse=False):
        x_sample = tf.cond(self.is_train_mode,
            lambda: self._draw_sample(x_mu, x_log_sigma_sq),
            lambda: x_mu)
        
        with tf.variable_scope('classifier', reuse = reuse):
            y_logits = self.classifier(inputs=x_sample, hidden_layers=[500],
                dim_output=self.dim_y, nonlinearity=tf.nn.softplus, reuse=reuse)

        return y_logits, x_sample

    def _generate_zxy( self, x, y, reuse = False ):

        with tf.variable_scope('encoder', reuse = reuse):
            encoder_out = self.encoder(inputs=tf.concat([x, y], axis=1), hidden_layers=[500],
                dim_output=2*self.dim_z, nonlinearity=tf.nn.softplus, reuse=reuse)
        z_mu, z_lsgms   = tf.split(encoder_out, 2, axis=1)
        z_sample        = self._draw_sample( z_mu, z_lsgms )

        return z_sample, z_mu, z_lsgms 

    def _generate_xzy( self, z, y, reuse = False ):

        with tf.variable_scope('decoder', reuse = reuse):
            decoder_out = self.decoder(inputs=tf.concat([z, y], axis=1), hidden_layers=[500],
                dim_output=2*self.dim_x, nonlinearity=tf.nn.softplus, reuse=reuse)
        x_recon_mu, x_recon_lsgms   = tf.split(decoder_out, 2, axis=1)

        return x_recon_mu, x_recon_lsgms

    def _objective( self ):

        ###############
        ''' L(x,y) ''' 
        ###############

        def L(x_recon, x, y, z):

            if self.distributions['p_z'] == 'gaussian_marg':

                log_prior_z = tf.reduce_sum( utils.tf_gaussian_marg( z[1], z[2] ), 1 )

            elif self.distributions['p_z'] == 'gaussian':

                log_prior_z = tf.reduce_sum( utils.tf_stdnormal_logpdf( z[0] ), 1 )

            if self.distributions['p_y'] == 'uniform':

                y_prior = (1. / self.dim_y) * tf.ones_like( y )
                log_prior_y = - tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y )

            if self.distributions['p_x'] == 'gaussian':

                log_lik = tf.reduce_sum( utils.tf_normal_logpdf( x, x_recon[0], x_recon[1] ), 1 )

            if self.distributions['q_z'] == 'gaussian_marg':

                log_post_z = tf.reduce_sum( utils.tf_gaussian_ent( z[2] ), 1 )

            elif self.distributions['q_z'] == 'gaussian':

                log_post_z = tf.reduce_sum( utils.tf_normal_logpdf( z[0], z[1], z[2] ), 1 )

            _L = log_prior_y + log_lik + log_prior_z - log_post_z

            return  _L

        ###########################
        ''' Labelled Datapoints '''
        ###########################

        self.y_lab_logits, self.x_lab = self._generate_yx( self.x_labelled_mu, self.x_labelled_lsgms )
        self.z_lab, self.z_lab_mu, self.z_lab_lsgms = self._generate_zxy( self.x_lab, self.y_lab )
        self.x_recon_lab_mu, self.x_recon_lab_lsgms = self._generate_xzy( self.z_lab, self.y_lab )

        L_lab = L(  [self.x_recon_lab_mu, self.x_recon_lab_lsgms], self.x_lab, self.y_lab,
                    [self.z_lab, self.z_lab_mu, self.z_lab_lsgms] )

        L_lab += - self.beta * tf.nn.softmax_cross_entropy_with_logits(logits=self.y_lab_logits, labels=self.y_lab )

        ############################
        ''' Unabelled Datapoints '''
        ############################

        def one_label_tensor( label ):

            indices = []
            values = []
            for i in range(self.num_ulab_batch):
                indices += [[ i, label ]]
                values += [ 1. ]

            _y_ulab = tf.sparse_tensor_to_dense( 
                      tf.SparseTensor( indices=indices, values=values, dense_shape=[ self.num_ulab_batch, self.dim_y ] ), 0.0 )

            return _y_ulab

        self.y_ulab_logits, self.x_ulab = self._generate_yx( self.x_unlabelled_mu, self.x_unlabelled_lsgms, reuse = True )

        for label in range(self.dim_y):

            _y_ulab = one_label_tensor( label )
            self.z_ulab, self.z_ulab_mu, self.z_ulab_lsgms = self._generate_zxy( self.x_ulab, _y_ulab, reuse = True )
            self.x_recon_ulab_mu, self.x_recon_ulab_lsgms = self._generate_xzy( self.z_ulab, _y_ulab, reuse = True )
            _L_ulab =   tf.expand_dims(
                        L(  [self.x_recon_ulab_mu, self.x_recon_ulab_lsgms], self.x_ulab, _y_ulab, 
                            [self.z_ulab, self.z_ulab_mu, self.z_ulab_lsgms]), 1)

            if label == 0: L_ulab = tf.identity( _L_ulab )
            else: L_ulab = tf.concat([L_ulab, _L_ulab], axis=1)

        self.y_ulab = tf.nn.softmax(self.y_ulab_logits)

        U = tf.reduce_sum(self.y_ulab*(L_ulab-tf.log(self.y_ulab)), 1)

        ########################
        ''' Prior on Weights '''
        ########################

        L_weights = 0.
        _weights = tf.trainable_variables()
        for w in _weights: 
            L_weights += tf.reduce_sum( utils.tf_stdnormal_logpdf( w ) )

        ##################
        ''' Total Cost '''
        ##################

        L_lab_tot = tf.reduce_sum( L_lab )
        U_tot = tf.reduce_sum( U )

        self.cost = ( ( L_lab_tot + U_tot ) * self.num_batches + L_weights ) / ( 
                - self.num_batches * self.batch_size )

        ##################
        ''' Evaluation '''
        ##################

        self.y_test_logits, _ = self._generate_yx(self.x_labelled_mu,
            self.x_labelled_lsgms, reuse=True)
        self.y_test_pred = tf.cast(tf.greater(tf.nn.softmax(self.y_test_logits), 0.5),
            tf.float32)

        y = self.y_lab
        y_ = self.y_test_pred
        self.eval_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_lab, logits=self.y_test_logits))
        tp = tf.reduce_sum(y*y_)
        tn = tf.reduce_sum((1-y)*(1-y_))
        fp = tf.reduce_sum((1-y)*y_)
        fn = tf.reduce_sum(y*(1-y_))
        self.eval_accuracy = (tp+tn)/(tp+tn+fp+fn)
        self.eval_precision = tp/(tp+fp+1e-5)
        self.eval_recall = tp/(tp+fn+1e-5)

        tf.summary.scalar('eval_accuracy', self.eval_accuracy)
        tf.summary.scalar('eval_cross_entropy', self.eval_cross_entropy)
        tf.summary.scalar('eval_precision', self.eval_precision)
        tf.summary.scalar('eval_recall', self.eval_recall)



    def train(      self, x_labelled, y, x_unlabelled,
                    epochs,
                    x_valid, y_valid,
                    print_every = 1,
                    learning_rate = 3e-4,
                    beta1 = 0.9,
                    beta2 = 0.999,
                    seed = 31415,
                    stop_iter = 100,
                    save_path = None,
                    load_path = None    ):


        ''' Session and Summary '''
        if save_path is None: 
            self.save_path = 'checkpoints/model_GC_{}-{}-{}_{}.cpkt'.format(
                self.num_lab,learning_rate,self.batch_size,time.time())
        else:
            self.save_path = save_path

        np.random.seed(seed)
        tf.set_random_seed(seed)

        with self.G.as_default():

            self.optimiser = tf.train.AdamOptimizer( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 )
            self.train_op = self.optimiser.minimize( self.cost )
            init = tf.global_variables_initializer()
            
        
        _data_labelled = np.hstack( [x_labelled, y] )
        _data_unlabelled = x_unlabelled
        x_valid_mu, x_valid_lsgms = x_valid[ :, :self.dim_x ], x_valid[ :, self.dim_x:2*self.dim_x ]

        with self.session as sess:

            sess.run(init)
            if load_path == 'default': self.saver.restore( sess, self.save_path )
            elif load_path is not None: self.saver.restore( sess, load_path )    

            best_eval_accuracy = 0.
            stop_counter = 0

            for epoch in tqdm(range(epochs)):

                ''' Shuffle Data '''
                np.random.shuffle( _data_labelled )
                np.random.shuffle( _data_unlabelled )

                ''' Training '''
                
                for x_l_mu, x_l_lsgms, y, x_u_mu, x_u_lsgms in utils.feed_numpy_semisupervised(    
                    self.num_lab_batch, self.num_ulab_batch, 
                    _data_labelled[:,:2*self.dim_x], _data_labelled[:,2*self.dim_x:],_data_unlabelled ):

                    training_result = sess.run( [self.train_op, self.cost],
                                            feed_dict = {    self.x_labelled_mu:            x_l_mu,     
                                                            self.x_labelled_lsgms:         x_l_lsgms,
                                                            self.y_lab:                 y,
                                                            self.x_unlabelled_mu:         x_u_mu,
                                                            self.x_unlabelled_lsgms:     x_u_lsgms,
                                                            self.is_train_mode: True} )

                    training_cost = training_result[1]

                ''' Evaluation '''
                stop_counter += 1
                res = sess.run([self.eval_accuracy, self.eval_cross_entropy,
                    self.eval_precision, self.eval_recall, self.merged],
                            feed_dict = {   self.x_labelled_mu:     x_valid_mu,
                                            self.x_labelled_lsgms:    x_valid_lsgms,
                                            self.y_lab:                y_valid,
                                            self.is_train_mode: False} )
                self.train_writer.add_summary(res[-1], epoch)
                eval_accuracy = res[0]

                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    self.saver.save( sess, self.save_path )
                    stop_counter = 0

                if stop_counter >= stop_iter:
                    print('Stopping GC training')
                    print('No change in validation accuracy for {} iterations'.format(stop_iter))
                    print('Best validation accuracy: {}'.format(best_eval_accuracy))
                    print('Model saved in {}'.format(self.save_path))
                    break

    def predict_labels(self, x):
        x_test_mu = x[:,:self.dim_x]
        x_test_lsgms = x[:,self.dim_x:2*self.dim_x]
        y_ = self.session.run(self.y_test_pred, feed_dict={self.x_labelled_mu: x_test_mu,
            self.x_labelled_lsgms: x_test_lsgms})
        return y_


    def encoder(self, inputs, hidden_layers, dim_output, nonlinearity, reuse):
        """ Create encoder graph

        Args:
            hidden_layers: list of integers specifies sizes of hidden layers
        """
        for l in hidden_layers:    
            inputs = tf.layers.dense(
                inputs=inputs,
                units=l,
                activation=nonlinearity,
                reuse=reuse)
        out = tf.layers.dense(
                inputs=inputs,
                units=dim_output,
                activation=None,
                reuse=reuse)
        return out
    

    def decoder(self, inputs, hidden_layers, dim_output, nonlinearity, reuse):
        """ Create decoder graph

        Args:
            hidden_layers: list of integers specifies sizes of hidden layers
        """
        for l in hidden_layers:    
            inputs = tf.layers.dense(
                inputs=inputs,
                units=l,
                activation=nonlinearity,
                reuse=reuse)
        out = tf.layers.dense(
                inputs=inputs,
                units=dim_output,
                activation=None,
                reuse=reuse)
        return out


    def classifier(self, inputs, hidden_layers, dim_output, nonlinearity, reuse):
        """ Create classifier graph

        Args:
            hidden_layers: list of integers specifies sizes of hidden layers
        """
        for l in hidden_layers:    
            inputs = tf.layers.dense(
                inputs=inputs,
                units=l,
                activation=nonlinearity,
                reuse=reuse)
        out = tf.layers.dense(
                inputs=inputs,
                units=dim_output,
                activation=None,
                reuse=reuse)
        return out