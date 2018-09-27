# Colorization
* Used a Multi-layer perceptron that learns a function by training on a dataset, where is the number of dimensions for input and is the number of dimensions for output. 
* Given a set of features and a target, it can learn a non-linear function approximator for our regression. 

* We used MLPRegressor that trains using backpropagation with no activation function in the output layer. 

* It uses the squared error as the loss function and the output is a set of continuous values.

* We used hyperparameters alpha = 1e-5 (learning rate) and 2 hidden layers with 10, 15 neurons in the first and the second hidden layer respectively and 3 output neurons to output red, green and blue color values. 'relu' activation function was used and 'adam' (stochastic gradient-based optimizer ) optimizer was used for weight optimization.

* Colors for each of 'data.csv' has been provided in 'predict_soln.csv' (refer the file in the attachment)

