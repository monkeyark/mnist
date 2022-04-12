Train MNIST dataset with CNN structures, then export the trained parameters.

W: input size
F: filter size / kernel size
S: stride
P: padding size on the border
O: output size
O = (W + 2P − F) / S + 1
output size = (input size + 2 * padding - kernel size) / stride + 1

(28 - 7) / 3 + 1 = 8

CNN structures
CL: convolutional layer, FL: fully connected layer

As there are only 10 categories for the MNIST dataset, the last FL output neuron is 10

CNN 1-2
1 CL, 2 FL
For CL, there are 4 filters, with filter size 7, stride 3
Activation function
For FL1, output neuron is 64, input neuron is 256, can be computed with the parameters of CL
Activation function
For FL2, output neuron is 10



CNN 2-1
2 CL, 1 FL
For CL1, 16 filters, each with filter size 7 and stride 2
Activation function
For CL2, 4 filters, each with filter size 5 and stride 2
Activation function
For FL, output neuron is 10

CNN 3-2
3 CL, 2 FL
For CL1, 16 filters, filter size 3 and stride 2
Activation function
For CL2, 4 filters, filter size 3 and stride 2
Activation function
For CL3, 16 filters, filter size 3 and stride 1
Activation function
For FL1, output neurons 64
For FL2, output neuron 10

CNN 4-2
4 CL, 2 FL
CL1, 16 filters, filter size 5, stride 2
Activation function
CL2, 4 filters, filter size 3, stride 1
Activation function
CL3, 16 filters, filter size 3, stride 2
Activation function
CL4, 4 filters, filter size 3, stride 1
FL1, output neuron 64
FL2, output neuron 10
