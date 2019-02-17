import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

# Input data
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# Real output
y = np.array([[0,0,1,1]]).T

np.random.seed(1)

# Initialize start weights in  interval [2;1)
weights1 = 2*np.random.random((3,4)) - 1
weights2 = 2*np.random.random((4,1)) - 1

layer1_error = np.array([[1,1,1,1]]).T
layer2_error = np.array([[1,1,1,1]]).T
epoch = 0

while (np.any(np.abs(layer2_error) > 0.01) and epoch < 10000):
    
    layer0 = X
    layer1 = sigmoid(np.dot(layer0,weights1))
    layer2 = sigmoid(np.dot(layer1,weights2))
    
    layer2_error = y - layer2
    layer2_delta = layer2_error*sigmoid_deriv(layer2)
    
    # Backpopagation itself: how much layer1 influence on the errors in the layer2?
    layer1_error = layer2_delta.dot(weights2.T)    
    layer1_delta = layer1_error * sigmoid_deriv(layer1)
    
    weights1 += np.dot(layer0.T,layer1_delta)
    weights2 += np.dot(layer1.T, layer2_delta)
    
    epoch += 1

print ("Epochs: ", epoch, "\n")
print ("Errors:\n", layer2_error, "\n")
print ("Tranig output:\n", layer2)