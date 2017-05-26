from genclass import GenerativeClassifier
import numpy as np

from ecg.utils import tools
from ecg.utils.diseases import all_holter_diseases

def unison_shuffled_copies(list_of_arr):
    p = np.random.permutation(len(list_of_arr[0]))
    res = [a[p] for a in list_of_arr]
    return res


def encode_dataset(path_to_encoded_data):
    paths = tools.find_files(path_to_encoded_data, '*.npy')
    paths = paths[:500]

    mu = np.vstack([np.load(path).item()['mu'] for path in paths])
    sigma = np.vstack([np.load(path).item()['sigma'] for path in paths])
    y = np.vstack([np.load(path).item()['events'] for path in paths])
    
    #create targets
    list_of_disease = ['Ventricular_PVC'] #Atrial_PAC Ventricular_PVC
    y = np.stack([y[:,all_holter_diseases.index(d)] for d in list_of_disease], 1)
    other = (np.sum(y, 1) < 0.5).astype(float)
    y = np.concatenate((y, other[:,None]),1)

    assert sum(np.sum(y,1)>1.5) == 0, 'There are some multilabel events'
    for i in range(len(list_of_disease)+1):
        if i < len(list_of_disease):
            print('Find {0} {1}'.format(y.sum(0)[i], list_of_disease[i]))
        else:
            print('Find {0} other diseases'.format(y.sum(0)[i]))
    return mu, sigma, y


def balance_labels(n_labels, x_list, y_list):
    y_dim = y_list[0].shape[1]
    intervals = [n_labels//y_dim for i in range(y_dim)]
    intervals[-1] = n_labels - sum(intervals[:-1])
    x_lab = np.empty([0,x_list[0].shape[1]])
    y_lab = np.empty([0,y_list[0].shape[1]])
    for i, inter in enumerate(intervals):
        x_lab = np.vstack([x_lab, x_list[i][:inter,:]])
        x_list[i] = x_list[i][inter:,:]
        y_lab = np.vstack([y_lab, y_list[i][:inter,:]])
        y_list[i] = y_list[i][inter:,:]
    print('\nrest ')
    [print(i.shape, j.shape) for i, j in [y_list, x_list]]
    return x_lab, y_lab, x_list, y_list

def split_data(mu, sigma, y):
    mu, sigma, y = unison_shuffled_copies([mu, sigma, y])
    x = np.hstack([mu,sigma])
    x_list = [x[y[:,i]==1,:] for i in range(y.shape[1])]
    y_list = [y[y[:,i]==1,:] for i in range(y.shape[1])]
    
    n = int(input('enter number of labeled data '))
    x_lab, y_lab, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)

    n = int(input('enter number of validation data '))
    x_valid, y_valid, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)

    n = int(input('enter number of test data '))
    x_test, y_test, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)
    
    n = int(input('enter number of unlabeled data '))
    x_ulab, y_ulab, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)

    #add unlubeled data
    # x_test = np.vstack([x_test, x_list[1][:1000,:]])
    # y_test = np.vstack([y_test, y_list[1][:1000,:]])
    # print()
    # print(x_ulab.shape)
    # print(y_ulab.sum(0))

    return x_lab, y_lab, x_ulab, y_ulab, x_valid, y_valid, x_test, y_test

if __name__ == '__main__':
    
    #############################
    ''' Experiment Parameters '''
    #############################

    num_batches = 100       #Number of minibatches in a single epoch
    dim_z = 256              #Dimensionality of latent variable (z)
    epochs = 1001           #Number of epochs through the full dataset
    learning_rate = 3e-4    #Learning rate of ADAM
    alpha = 0.1             #Discriminatory factor (see equation (9) of http://arxiv.org/pdf/1406.5298v2.pdf)
    seed = 31415            #Seed for RNG

    #Neural Networks parameterising p(x|z,y), q(z|x,y) and q(y|x)
    hidden_layers_px = [ 500, 500 ]
    hidden_layers_qz = [ 500, 500 ]
    hidden_layers_qy = [ 500, 500 ]

    ####################
    ''' Load Dataset '''
    ####################
    mu, sigma, y = encode_dataset(path_to_encoded_data='../ECG_encoder/predictions/latent_states_PVC/')
    x_lab, y_lab, x_ulab, y_ulab, x_valid, y_valid, x_test, y_test = split_data(mu, sigma, y)
    num_lab = y_lab.shape[0]           #Number of labelled examples (total)

    dim_x = x_lab.shape[1] / 2
    dim_y = y_lab.shape[1]
    num_examples = y_lab.shape[0] + y_ulab.shape[0]
    
    ###################################
    ''' Train Generative Classifier '''
    ###################################

    GC = GenerativeClassifier(  dim_x, dim_z, dim_y,
                                num_examples, num_lab, num_batches,
                                hidden_layers_px    = hidden_layers_px, 
                                hidden_layers_qz    = hidden_layers_qz, 
                                hidden_layers_qy    = hidden_layers_qy,
                                alpha               = alpha )

    GC.train(   x_labelled      = x_lab, y = y_lab, x_unlabelled = x_ulab,
                x_valid         = x_valid, y_valid = y_valid,
                epochs          = epochs, 
                learning_rate   = learning_rate,
                seed            = seed,
                print_every     = 10,
                load_path       = None )


    ############################
    ''' Evaluate on Test Set '''
    ############################

    GC_eval = GenerativeClassifier(  dim_x, dim_z, dim_y, num_examples, num_lab, num_batches )

    with GC_eval.session:
        GC_eval.saver.restore( GC_eval.session, GC.save_path )
        GC_eval.predict_labels( x_test, y_test )
    
    