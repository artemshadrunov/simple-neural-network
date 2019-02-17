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

syn0 = 2*np.random.random((3,1)) - 1

l1_error = np.array([[1,1,1,1]]).T
iter = 0

while (np.any(np.abs(l1_error) > 0.01) and iter < 10000):
    
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    
    l1_error = y - l1
    
    l1_delta = l1_error * sigmoid_deriv(l1)
    
    syn0 += np.dot(l0.T,l1_delta)   
    
    iter += 1

print ("Iterations: ", iter, "\n")
print ("Errors:\n", l1_error, "\n")
print ("Tranig output:\n", l1)