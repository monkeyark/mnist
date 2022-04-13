Train MNIST dataset with different CNN structures, then export the trained parameters.

## Terminology
W: input size
F: filter size / kernel size  
S: stride  
P: padding size on the border  
O: output size  
O = (W + 2P − F) / S + 1  
output size = (input size + 2 * padding - kernel size) / stride + 1  

CONV: convolutional layer  
FC: fully connected layer / linear layer  
input depth = input layer / dimension, MNIST default 1  
in_channle = input size / input width, MNIST default 28 (28x28 image size)  
out_channel = filter num / output depth / layer / dimension  
kernel_size = filter size / filter width  
stride = stride  
padding = padding  

As there are only 10 categories for the MNIST dataset, the last FC output neuron is 10

## CNN structure
### CNN 1-2
1 CONV, 2 FC  
CONV1, there are 4 filters, with filter size 7, stride 3  
Activation function  
FC1, output neuron 64  
Activation function  
FC2, output neuron 10  

| CNN1_2  |             |             |            |              |        |         |                         |
|---------|-------------|-------------|------------|--------------|--------|---------|-------------------------|
|         | input width | input depth | filter num | filter width | stride | padding | output width            |
| CONV    | 28          | 1           | 4          | 7            | 3      | 0       | floor[(28+2*0-7)/3]+1=8 |
|         | in_feature  | out_feature |            |              |        |         |                         |
| FC1     | 8^2*4=256   | 64          |            |              |        |         |                         |
| FC2     | 64          | 10          |            |              |        |         |                         |

### CNN 2-1
2 CONV, 1 FC  
CONV1, 16 filters, each with filter size 7 and stride 2  
Activation function  
CONV2, 4 filters, each with filter size 5 and stride 2  
Activation function  
FC, output neuron is 10  

| CNN2_1 |             |             |            |              |        |         |                          |
|--------|-------------|-------------|------------|--------------|--------|---------|--------------------------|
|        | input width | input depth | filter num | filter width | stride | padding | output width             |
| CONV1  | 28          | 1           | 16         | 7            | 2      | 0       | floor[(28+2*0-7)/2]+1=11 |
| CONV2  | 11          | 16          | 4          | 5            | 2      | 0       | floor[(11+2*0-5)/2]+1=4  |
|        | in_feature  | out_feature |            |              |        |         |                          |
| FC1    | 4^2*4=64    | 10          |            |              |        |         |                          |

### CNN 3-2
3 CONV, 2 FC  
CONV1, 16 filters, filter size 3 and stride 2  
Activation function  
CONV2, 4 filters, filter size 3 and stride 2  
Activation function  
CONV3, 16 filters, filter size 3 and stride 1  
Activation function  
FC1, output neurons 64  
FC2, output neuron 10  

| CNN3_2  |             |             |            |              |        |         |                          |
|---------|-------------|-------------|------------|--------------|--------|---------|--------------------------|
|         | input width | input depth | filter num | filter width | stride | padding | output width             |
| CONV1   | 28          | 1           | 16         | 3            | 2      | 0       | floor[(28+2*0-3)/2]+1=13 |
| CONV2   | 13          | 16          | 4          | 3            | 2      | 0       | floor[(13+2*0-3)/2]+1=6  |
| CONV3   | 6           | 4           | 16         | 3            | 1      | 0       | floor[(6+2*0-3)/1]+1=4   |
|         | in_feature  | out_feature |            |              |        |         |                          |
| FC1     | 4^2*16=256  | 64          |            |              |        |         |                          |
| FC2     | 64          | 10          |            |              |        |         |                          |


### CNN 4-2
4 CONV, 2 FC  
CONV1, 16 filters, filter size 5, stride 2  
Activation function  
CONV2, 4 filters, filter size 3, stride 1  
Activation function  
CONV3, 16 filters, filter size 3, stride 2  
Activation function  
CONV4, 4 filters, filter size 3, stride 1  
FC1, output neuron 64  
FC2, output neuron 10  

| CNN4_2  |             |             |            |              |        |         |                          |
|---------|-------------|-------------|------------|--------------|--------|---------|--------------------------|
|         | input width | input depth | filter num | filter width | stride | padding | output width             |
| CONV1   | 28          | 1           | 16         | 5            | 2      | 0       | floor[(28+2*0-5)/2]+1=12 |
| CONV2   | 12          | 16          | 4          | 3            | 1      | 0       | floor[(12+2*0-3)/1]+1=10 |
| CONV3   | 10          | 4           | 16         | 3            | 2      | 0       | floor[(10+2*0-3)/2]+1=4  |
| CONV4   | 4           | 16          | 4          | 3            | 1      | 0       | floor[(4+2*0-3)/1]+1=2   |
|         | in_feature  | out_feature |            |              |        |         |                          |
| FC1     | 2^2*4=16    | 64          |            |              |        |         |                          |
| FC2     | 64          | 10          |            |              |        |         |                          |
