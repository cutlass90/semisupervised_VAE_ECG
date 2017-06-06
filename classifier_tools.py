import prettytensor as pt
import tensorflow as tf
import numpy as np

from ecg.utils import tools
from ecg.utils.diseases import all_holter_diseases

logc = np.log(2.*np.pi)
c = - 0.5 * np.log(2*np.pi)

def tf_normal_logpdf(x, mu, log_sigma_sq):

	return ( - 0.5 * logc - log_sigma_sq / 2. - tf.div( tf.square(x-mu), 2 * tf.exp( log_sigma_sq ) ) )

def tf_stdnormal_logpdf(x):

	return ( - 0.5 * ( logc + tf.square( x ) ) )

def tf_gaussian_ent(log_sigma_sq):

	return ( - 0.5 * ( logc + 1.0 + log_sigma_sq ) )

def tf_gaussian_marg(mu, log_sigma_sq):

	return ( - 0.5 * ( logc + ( tf.square( mu ) + tf.exp( log_sigma_sq ) ) ) )

def tf_binary_xentropy(x, y, const = 1e-10):

    return - ( x * tf.log ( tf.clip_by_value( y, const, 1.0 ) ) + \
             (1.0 - x) * tf.log( tf.clip_by_value( 1.0 - y, const, 1.0 ) ) )

def feed_numpy_semisupervised(num_lab_batch, num_ulab_batch, x_lab, y, x_ulab):
	""" Return batch for training

	Args:
		num_lab_batch: int, number of labeled samples in each batch
		num_ulab_batch: int, number of unlabeled samples in each batch

	"""

	size = x_lab.shape[0] + x_ulab.shape[0]
	batch_size = num_lab_batch + num_ulab_batch
	count = int(size / batch_size)

	dim = x_lab.shape[1]

	for i in range(count):
		start_lab = i * num_lab_batch
		end_lab = start_lab + num_lab_batch
		start_ulab = i * num_ulab_batch
		end_ulab = start_ulab + num_ulab_batch

		yield [	x_lab[start_lab:end_lab,:dim//2], x_lab[start_lab:end_lab,dim//2:dim], y[start_lab:end_lab],
				x_ulab[start_ulab:end_ulab,:dim//2], x_ulab[start_ulab:end_ulab,dim//2:dim] ]

def feed_numpy(batch_size, x):

	size = x.shape[0]
	count = int(size / batch_size)

	dim = x.shape[1]

	for i in range(count):
		start = i * batch_size
		end = start + batch_size

		yield x[start:end]

def print_metrics(epoch, *metrics):

	print(25*'-')
	for metric in metrics: 
		print('[{}] {} {}: {}'.format(epoch, metric[0],metric[1],metric[2]))
	print(25*'-')


def unison_shuffled_copies(list_of_arr):
    p = np.random.permutation(len(list_of_arr[0]))
    res = [a[p] for a in list_of_arr]
    return res


def encode_dataset(path_to_encoded_data, required_diseases):
    # required_diseases: list of string with name of reuiered diseases
    paths = tools.find_files(path_to_encoded_data, '*.npy')
    paths = paths[:500]

    mu = np.vstack([np.load(path).item()['mu'] for path in paths])
    sigma = np.vstack([np.load(path).item()['sigma'] for path in paths])
    y = np.vstack([np.load(path).item()['events'] for path in paths])
    
    #create targets
    y = np.stack([y[:,all_holter_diseases.index(d)] for d in required_diseases], 1)
    other = (np.sum(y, 1) < 0.5).astype(float)
    y = np.concatenate((y, other[:,None]),1)

    assert sum(np.sum(y,1)>1.5) == 0, 'There are some multilabel events'
    for i in range(len(required_diseases)+1):
        if i < len(required_diseases):
            print('Find {0} {1}'.format(y.sum(0)[i], required_diseases[i]))
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

def split_data(mu, sigma, y, n_lab=None, n_unlab=None, n_val=None):
    mu, sigma, y = unison_shuffled_copies([mu, sigma, y])
    x = np.hstack([mu,sigma])
    x_list = [x[y[:,i]==1,:] for i in range(y.shape[1])]
    y_list = [y[y[:,i]==1,:] for i in range(y.shape[1])]
    
    n = int(input('enter number of labeled data ')) if n_lab is None else n_lab
    x_lab, y_lab, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)

    n = int(input('enter number of validation data ')) if n_val is None else n_val
    x_valid, y_valid, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)
    
    n = int(input('enter number of unlabeled data ')) if n_unlab is None else n_unlab
    x_ulab, y_ulab, x_list, y_list = balance_labels(n_labels=n, x_list=x_list,
        y_list=y_list)

    return x_lab, y_lab, x_ulab, y_ulab, x_valid, y_valid