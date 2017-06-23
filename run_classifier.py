from classifier import GenerativeClassifier
import numpy as np

import classifier_tools as c_tools
from classifier_parameters import parameters as PARAM

####################
''' Load Dataset '''
####################
mu, sigma, y = c_tools.encode_dataset(PARAM['path_to_encoded_data'], PARAM['required_diseases'])
x_lab, y_lab, x_ulab, y_ulab, x_valid, y_valid = c_tools.split_data(mu, sigma, y, PARAM['required_diseases'])
num_lab = y_lab.shape[0]           #Number of labelled examples (total)

dim_x = x_lab.shape[1] / 2
dim_y = y_lab.shape[1]
num_examples = y_lab.shape[0] + y_ulab.shape[0]

###################################
''' Train Generative Classifier '''
###################################
GC = GenerativeClassifier(
    dim_x=dim_x,
    dim_z=PARAM['dim_z'],
    dim_y=dim_y,
    num_examples=num_examples,
    num_lab=num_lab,
    num_batches=PARAM['num_batches'],
    required_diseases=PARAM['required_diseases'],
    labels_distribution=PARAM['labels_distribution'],
    hidden_layers_px=PARAM['hidden_layers_px'], 
    hidden_layers_qz=PARAM['hidden_layers_qz'], 
    hidden_layers_qy=PARAM['hidden_layers_qy'],
    alpha=PARAM['alpha'])

GC.train(x_labelled=x_lab,
    y=y_lab,
    x_unlabelled=x_ulab,
    x_valid=x_valid,
    y_valid=y_valid,
    epochs=PARAM['epochs'],
    learning_rate=PARAM['learning_rate'])
    