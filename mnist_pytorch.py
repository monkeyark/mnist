# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""**Save the files into dataframes**"""

# Importing dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
	root = 'data',
	train = True,
	transform = ToTensor(), 
	download = True,
)
test_data = datasets.MNIST(
	root = 'data', 
	train = False, 
	transform = ToTensor()
)

# # Print train_data and test_data size
# print(train_data)
# print(train_data.data.size())
# print(train_data.targets.size())
# print(test_data)

# # Plot one train_data
# import matplotlib.pyplot as plt
# plt.imshow(train_data.data[0], cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

# # Plot multiple train_data
# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
# 	sample_idx = torch.randint(len(train_data), size = (1,)).item()
# 	img, label = train_data[sample_idx]
# 	figure.add_subplot(rows, cols, i)
# 	plt.title(label)
# 	plt.axis("off")
# 	plt.imshow(img.squeeze(), cmap = "gray")
# plt.show()

"""Preparing data for training with DataLoaders

The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.
"""

from torch.utils.data import DataLoader
loaders = {
	'train' : DataLoader(
		train_data,
		batch_size = 100,
		shuffle = True,
		num_workers = 1),

	'test'  : DataLoader(
		test_data,
		batch_size = 100,
		shuffle = True,
		num_workers = 1),
}
loaders

import torch.nn as nn
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels = 1,
				out_channels = 16,
				kernel_size = 5,
				stride = 1,
				padding = 2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		# fully connected layer, output 10 classes
		self.out = nn.Linear(32 * 7 * 7, 10)
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		# flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = x.view(x.size(0), -1)
		output = self.out(x)
		return output, x	# return x for visualization

cnn = CNN()
print(cnn)

# Define loss function
loss_func = nn.CrossEntropyLoss()
loss_func

# Define a Optimization Function
from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
optimizer

# Train the model
from torch.autograd import Variable
num_epochs = 10
def train(num_epochs, cnn, loaders):
	
	cnn.train()
	# Train the model
	total_step = len(loaders['train'])
		
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(loaders['train']):
			
			# gives batch data, normalize x when iterate train_loader
			b_x = Variable(images)   # batch x
			b_y = Variable(labels)   # batch y
			output = cnn(b_x)[0]
			loss = loss_func(output, b_y)
			
			# clear gradients for this training step   
			optimizer.zero_grad()           
			
			# backpropagation, compute gradients 
			loss.backward()    
			# apply gradients             
			optimizer.step()                
			
			if (i+1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
			pass
		pass
	pass
train(num_epochs, cnn, loaders)

# Evaluate the model on test data
def test():
	# Test the model
	cnn.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in loaders['test']:
			test_output, last_layer = cnn(images)
			pred_y = torch.max(test_output, 1)[1].data.squeeze()
			accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
			pass
	print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
pass

test()

# Print 10 predictions from test data
sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:10].numpy()
actual_number

test_output, last_layer = cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')