parameters = {
"num_batches":100,       #Number of minibatches in a single epoch
"dim_z":256,              #Dimensionality of latent variable (z)
"epochs":1001,           #Number of epochs through the full dataset
"learning_rate":3e-4,    #Learning rate of ADAM
"alpha":0.1,             #Discriminatory factor (see equation (9) of http://arxiv.org/pdf/1406.5298v2.pdf)
"seed":31415,            #Seed for RNG

#Neural Networks parameterising p(x|z,y), q(z|x,y) and q(y|x)
"hidden_layers_px":[ 500, 500 ],
"hidden_layers_qz":[ 500, 500 ],
"hidden_layers_qy":[ 500, 500 ]
}

