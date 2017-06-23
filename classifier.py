import os

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

import classifier_tools as c_tools

class GenerativeClassifier(object):

    def __init__(   self,
                    dim_x, dim_z, dim_y,
                    num_examples, num_lab, num_batches,
                    required_diseases,
                    labels_distribution,
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
                    l2_loss = 0.0):

        self.dim_x, self.dim_z, self.dim_y = int(dim_x), int(dim_z), int(dim_y)

        self.distributions = {      'p_x':     p_x,            
                                    'q_z':     q_z,            
                                    'p_z':     p_z,            
                                    'p_y':    'custom'    }

        self.num_examples = num_examples
        self.num_batches = num_batches
        self.num_lab = num_lab
        self.num_ulab = self.num_examples - num_lab
        self.required_diseases = required_diseases
        self.labels_distribution = labels_distribution

        assert self.num_lab % self.num_batches == 0, '#Labelled % #Batches != 0'
        assert self.num_ulab % self.num_batches == 0, '#Unlabelled % #Batches != 0'
        assert self.num_examples % self.num_batches == 0, '#Examples % #Batches != 0'

        self.batch_size = self.num_examples // self.num_batches
        self.num_lab_batch = self.num_lab // self.num_batches
        self.num_ulab_batch = self.num_ulab // self.num_batches

        self.beta = alpha * ( float(self.batch_size) / self.num_lab_batch )
        self.list_train_summary, self.list_test_summary = [], []

        self.create_graph()

        self.create_optimizer_graph(self.cost)

        os.makedirs('summary', exist_ok=True)
        sub_d = len(os.listdir('summary'))
        self.train_writer = tf.summary.FileWriter(logdir='summary/'+str(sub_d)+'/train')
        self.test_writer = tf.summary.FileWriter(logdir='summary/'+str(sub_d)+'/test')
        self.train_summary = tf.summary.merge(self.list_train_summary)
        self.test_summary = tf.summary.merge(self.list_test_summary)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(),
                            max_to_keep = 1000)


    # --------------------------------------------------------------------------
    def __enter__(self):
        return self


    # --------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()
        if self.sess is not None:
            self.sess.close()

    
    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')

        self.x_labelled_mu,\
        self.x_labelled_lsgms,\
        self.x_unlabelled_mu,\
        self.x_unlabelled_lsgms,\
        self.y_lab,\
        self.is_train_mode,\
        self.learning_rate = self.input_graph()

        L_lab = self.labelled_cost()

        U = self.unlabelled_cost()

        unbalance_loss = self.unbalance_loss()
        unbalance_loss = 0

        self.cost = self.create_cost_graph(L_lab, U, unbalance_loss)

        self.evaluation_graph()

        print('Done!')
      

    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        x_labelled_mu = tf.placeholder( tf.float32, [None, self.dim_x] )
        x_labelled_lsgms = tf.placeholder( tf.float32, [None, self.dim_x] )
        x_unlabelled_mu = tf.placeholder( tf.float32, [None, self.dim_x] )
        x_unlabelled_lsgms = tf.placeholder( tf.float32, [None, self.dim_x] )
        y_lab = tf.placeholder(tf.float32, [None, self.dim_y])
        is_train_mode = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32, [])
        return x_labelled_mu, x_labelled_lsgms, x_unlabelled_mu, x_unlabelled_lsgms,\
            y_lab, is_train_mode, learning_rate


    # --------------------------------------------------------------------------
    def _draw_sample( self, mu, log_sigma_sq ):
        epsilon = tf.random_normal( ( tf.shape( mu ) ), 0, 1 )
        sample = mu + tf.exp(0.5*log_sigma_sq)*epsilon
        return sample

    # --------------------------------------------------------------------------
    def _generate_yx( self, x_mu, x_log_sigma_sq, reuse=False):
        x_sample = tf.cond(self.is_train_mode,
            lambda: self._draw_sample(x_mu, x_log_sigma_sq), lambda: x_mu)
        with tf.variable_scope('classifier', reuse = reuse):
            y_logits = self.classifier(inputs=x_sample, hidden_layers=[500],
                dim_output=self.dim_y, nonlinearity=tf.nn.softplus, reuse=reuse)
        return y_logits, x_sample

    # --------------------------------------------------------------------------
    def _generate_zxy( self, x, y, reuse = False ):
        with tf.variable_scope('encoder', reuse = reuse):
            encoder_out = self.encoder(inputs=tf.concat([x, y], axis=1), hidden_layers=[500],
                dim_output=2*self.dim_z, nonlinearity=tf.nn.softplus, reuse=reuse)
        z_mu, z_lsgms   = tf.split(encoder_out, 2, axis=1)
        z_sample        = self._draw_sample( z_mu, z_lsgms )
        return z_sample, z_mu, z_lsgms 

    # --------------------------------------------------------------------------
    def _generate_xzy( self, z, y, reuse = False ):
        with tf.variable_scope('decoder', reuse = reuse):
            decoder_out = self.decoder(inputs=tf.concat([z, y], axis=1), hidden_layers=[500],
                dim_output=2*self.dim_x, nonlinearity=tf.nn.softplus, reuse=reuse)
        x_recon_mu, x_recon_lsgms   = tf.split(decoder_out, 2, axis=1)
        return x_recon_mu, x_recon_lsgms

    # --------------------------------------------------------------------------
    def L(self, x_recon, x, y, z):
        x_recon_mu, x_recon_lsgms = x_recon
        z_, z_mu, z_lsgms = z

        ''' L(x,y) ''' 
        if self.distributions['p_z'] == 'gaussian_marg':
            log_prior_z = tf.reduce_sum( c_tools.tf_gaussian_marg(z_mu, z_lsgms), 1 )
        # elif self.distributions['p_z'] == 'gaussian':
        #     log_prior_z = tf.reduce_sum( c_tools.tf_stdnormal_logpdf( z[0] ), 1 )

        if self.distributions['p_y'] == 'uniform':
            y_prior = (1. / self.dim_y) * tf.ones_like(y)
        else:
            y_prior = self.labels_distribution * tf.ones_like(y)
        log_prior_y = -tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y)



        log_lik = tf.reduce_sum(c_tools.tf_normal_logpdf(x, x_recon_mu, x_recon_lsgms), 1)

        if self.distributions['q_z'] == 'gaussian_marg':
            log_post_z = tf.reduce_sum(c_tools.tf_gaussian_ent(z_lsgms), 1)
        # elif self.distributions['q_z'] == 'gaussian':
        #     log_post_z = tf.reduce_sum( c_tools.tf_normal_logpdf( z[0], z[1], z[2] ), 1 )
        print(log_prior_y, log_lik, log_prior_z, log_post_z)

        _L = 2*log_prior_y + log_lik + log_prior_z - log_post_z

        return  _L


    # --------------------------------------------------------------------------
    def labelled_cost(self):
        print('\tlabelled_cost')
        self.y_lab_logits, self.x_lab = self._generate_yx(self.x_labelled_mu, self.x_labelled_lsgms)
        self.z_lab, self.z_lab_mu, self.z_lab_lsgms = self._generate_zxy( self.x_lab, self.y_lab )
        self.x_recon_lab_mu, self.x_recon_lab_lsgms = self._generate_xzy( self.z_lab, self.y_lab )

        L_lab = self.L(  [self.x_recon_lab_mu, self.x_recon_lab_lsgms], self.x_lab, self.y_lab,
                    [self.z_lab, self.z_lab_mu, self.z_lab_lsgms] )

        L_lab += - self.beta * tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_lab_logits, labels=self.y_lab)
        return L_lab


    # --------------------------------------------------------------------------
    def unlabelled_cost(self):
        print('\tunlabelled_cost')

        def one_label_tensor(label):
            ''' Create zero matrix shape [num_ulab_batch, dim_y] and ones in column label'''
            indices = []
            values = []
            for i in range(self.num_ulab_batch):
                indices += [[ i, label ]]
                values += [ 1. ]

            _y_ulab = tf.sparse_tensor_to_dense( 
                      tf.SparseTensor( indices=indices, values=values,
                        dense_shape=[ self.num_ulab_batch, self.dim_y ] ), 0.0 )
            return _y_ulab

        self.y_ulab_logits, self.x_ulab = self._generate_yx(self.x_unlabelled_mu,
            self.x_unlabelled_lsgms, reuse=True)

        for label in range(self.dim_y):
            _y_ulab = one_label_tensor( label )
            self.z_ulab, self.z_ulab_mu, self.z_ulab_lsgms = self._generate_zxy(
                self.x_ulab, _y_ulab, reuse=True)

            self.x_recon_ulab_mu, self.x_recon_ulab_lsgms = self._generate_xzy(
                self.z_ulab, _y_ulab, reuse=True)

            _L_ulab = tf.expand_dims(
                        self.L([self.x_recon_ulab_mu, self.x_recon_ulab_lsgms], self.x_ulab, _y_ulab, 
                            [self.z_ulab, self.z_ulab_mu, self.z_ulab_lsgms]), 1)

            if label == 0: L_ulab = tf.identity( _L_ulab )
            else: L_ulab = tf.concat([L_ulab, _L_ulab], axis=1)

        self.y_ulab = tf.nn.softmax(self.y_ulab_logits)

        U = tf.reduce_sum(self.y_ulab*(L_ulab-tf.log(self.y_ulab)), 1)
        return U


    # --------------------------------------------------------------------------
    def evaluation_graph( self ):
        print('\tevaluation_graph')

        self.pred = tf.cast(tf.greater(tf.nn.softmax(self.y_lab_logits), 0.5), tf.float32)

        
        preds = tf.reduce_max(self.y_lab_logits, axis=1)
        preds = tf.cast(tf.equal(logits, tf.expand_dims(pred, 1)), tf.float32)
        for i, disease in enumerate(self.required_diseases):
            y = self.y_lab[:,i]
            y_ = preds[:,i]
            tp = tf.reduce_sum(y*y_)
            tn = tf.reduce_sum((1-y)*(1-y_))
            fp = tf.reduce_sum((1-y)*y_)
            fn = tf.reduce_sum(y*(1-y_))
            pr = tp/(tp+fp+1e-5)
            re = tp/(tp+fn+1e-5)
            f1 = 2*pr*re/(pr+re+1e-5)

            self.list_test_summary.append(tf.summary.scalar(disease+' precision', pr))
            self.list_test_summary.append(tf.summary.scalar(disease+' recall', re))
            self.list_test_summary.append(tf.summary.scalar(disease+' f1 score', f1))


    # --------------------------------------------------------------------------
    def create_cost_graph(self, L_lab, U, unbalance_loss):
        print('\tcreate_cost_graph')
        #Prior on Weights
        L_weights = 0.
        _weights = tf.trainable_variables()
        for w in _weights: 
            L_weights += tf.reduce_sum( c_tools.tf_stdnormal_logpdf( w ) )

        #Total Cost
        L_lab_tot = -tf.reduce_sum(L_lab)/self.batch_size
        U_tot = -tf.reduce_sum(U)/self.batch_size
        L_weights = -L_weights/self.num_batches/self.batch_size
        # cost = ( ( L_lab_tot + U_tot ) * self.num_batches + L_weights ) / ( 
        #         - self.num_batches * self.batch_size )
        cost = L_lab_tot + U_tot + L_weights + 10*unbalance_loss

        self.list_train_summary.append(tf.summary.scalar('labelled loss', L_lab_tot))
        self.list_train_summary.append(tf.summary.scalar('unlabelled loss', U_tot))
        self.list_train_summary.append(tf.summary.scalar('L2 loss', L_weights))
        self.list_train_summary.append(tf.summary.scalar('unbalance loss', unbalance_loss))
        
        return cost

    # --------------------------------------------------------------------------
    def unbalance_loss(self):
        print('\tunbalance_loss')
        q = tf.nn.softmax(tf.reduce_sum(self.y_ulab, 0))
        print(q)
        cross_entropy = -tf.reduce_sum(tf.log(q+1e-5)*self.labels_distribution)
        
        self.list_train_summary.append(tf.summary.histogram(
            'predicted label distribution', tf.argmax(self.y_ulab, 1)))
        self.list_train_summary.append(tf.summary.histogram(
            'true label distribution', tf.argmax(self.y_lab, 1)))
        return cross_entropy


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                beta1=0.9, beta2=0.999)
            self.train_op = self.optimiser.minimize(cost)


    #---------------------------------------------------------------------------  
    def save_model(self, path = 'beat_detector_model', step = None):
        p = self.saver.save(self.sess, path, global_step = step)
        print("\tModel saved in file: %s" % p)


    #---------------------------------------------------------------------------
    def load_model(self, path):
        #path is path to file or path to directory
        #if path it is path to directory will be load latest model
        load_path = os.path.splitext(path)[0]\
        if os.path.isfile(path) else tf.train.latest_checkpoint(path)
        print('try to load {}'.format(load_path))
        self.saver.restore(self.sess, load_path)
        print("Model restored from file %s" % load_path)


    #---------------------------------------------------------------------------  
    def train(self, x_labelled, y, x_unlabelled, epochs, x_valid, y_valid, learning_rate):            
        
        _data_labelled = np.hstack( [x_labelled, y] )
        _data_unlabelled = x_unlabelled
        x_valid_mu, x_valid_lsgms = x_valid[:,:self.dim_x], x_valid[:,self.dim_x:2*self.dim_x]

        train_it = 0
        for epoch in tqdm(range(epochs)):
            ''' Shuffle Data '''
            np.random.shuffle( _data_labelled )
            np.random.shuffle( _data_unlabelled )

            ''' Training '''
            for x_l_mu, x_l_lsgms, y, x_u_mu, x_u_lsgms in c_tools.feed_numpy_semisupervised(    
                self.num_lab_batch, self.num_ulab_batch, 
                _data_labelled[:,:2*self.dim_x], _data_labelled[:,2*self.dim_x:],_data_unlabelled ):

                _, s = self.sess.run([self.train_op, self.train_summary],
                        feed_dict={self.x_labelled_mu:x_l_mu,     
                                    self.x_labelled_lsgms:x_l_lsgms,
                                    self.y_lab:y,
                                    self.x_unlabelled_mu:x_u_mu,
                                    self.x_unlabelled_lsgms:x_u_lsgms,
                                    self.is_train_mode:True,
                                    self.learning_rate:learning_rate})
                self.train_writer.add_summary(s, train_it)
                train_it += 1
            
            ''' Evaluation '''
            s = self.sess.run(self.test_summary,
                        feed_dict = {   self.x_labelled_mu:     x_valid_mu,
                                        self.x_labelled_lsgms:    x_valid_lsgms,
                                        self.y_lab:                y_valid,
                                        self.is_train_mode: False} )
            self.test_writer.add_summary(s, epoch)

    def predict_labels(self, x):
        x_test_mu = x[:,:self.dim_x]
        x_test_lsgms = x[:,self.dim_x:2*self.dim_x]
        y_ = self.sess.run(self.pred, feed_dict={self.x_labelled_mu: x_test_mu,
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


if __name__ == '__main__':
    GC = GenerativeClassifier(dim_x=128, dim_z=256, dim_y=10,
                    num_examples=1000, num_lab=100, num_batches=10,
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
                    l2_loss = 0.0)