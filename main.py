import torch
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Importing dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
	root = 'data',
	train = True,
	transform = ToTensor(),
	download = True, # set True if need download dataset
)
test_data = datasets.MNIST(
	root = 'data',
	train = False,
	transform = ToTensor()
)

# Preparing data for training with DataLoaders
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

# --------------------------------------------------------
# Define CNN models
import torch.nn as nn
class CNN1_2(nn.Module):
	def __init__(self):
		super(CNN1_2, self).__init__()
		self.conv1 = nn.Sequential(
			# in_channels = 1, out_channels = 4, kernel_size = 7, stride = 3, padding = 0
			nn.Conv2d(1, 4, 7, 3, 0),
			nn.ReLU(),
		)
		self.fc1 = nn.Sequential(
			# in_features = 256, out_features = 64, bias = False
			nn.Linear(256, 64, False),
			nn.ReLU(),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(64, 10, False)
		)
	def forward(self, x):
		x = self.conv1(x)
		x = x.view(x.size(0), -1)	# flatten the output of conv
		x = self.fc1(x)
		output = self.fc2(x)
		return output, x	# return x for visualization

class CNN2_1(nn.Module):
	def __init__(self):
		super(CNN2_1, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 7, 2, 0),
			nn.ReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 4, 5, 2, 0),
			nn.ReLU(),
		)
		self.fc1 = nn.Sequential(
			nn.Linear(64, 10, False)
		)
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		output = self.fc1(x)
		return output, x

class CNN3_2(nn.Module):
	def __init__(self):
		super(CNN3_2, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 3, 2, 0),
			nn.ReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 4, 3, 2, 0),
			nn.ReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(4, 16, 3, 1, 0),
			nn.ReLU(),
		)
		self.fc1 = nn.Sequential(
			nn.Linear(256, 64, False),
			nn.ReLU(),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(64, 10, False)
		)
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		output = self.fc2(x)
		return output, x

class CNN4_2(nn.Module):
	def __init__(self):
		super(CNN4_2, self).__init__()
		layer = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 5, 2, 0),
			nn.ReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 4, 3, 1, 0),
			nn.ReLU(),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(4, 16, 3, 2, 0),
			nn.ReLU(),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(16, 4, 3, 1, 0),
			nn.ReLU(),
		)
		self.fc1 = nn.Sequential(
			nn.Linear(16, 64, False),
			nn.ReLU(),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(64, 10, False)
		)
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		output = self.fc2(x)
		return output, x

cnn_models = [CNN1_2(), CNN2_1(), CNN3_2(), CNN4_2()]

# --------------------------------------------------------

# Define loss function
loss_func = nn.CrossEntropyLoss()

# Define a Optimization Function
from torch import optim
optimizers = []
for model in cnn_models:
	optimizers.append(optim.Adam(model.parameters(), lr = 0.01))


# Train the model
from torch.autograd import Variable

def train(num_epochs, loaders, cnn, optimizer):
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

# for model, opt in zip(cnn_models, optimizers):
# 	print(model.__ne__)
# 	num_epochs = 1	# Define epoch
# 	train(num_epochs, loaders, model, opt)

# # Evaluate the model on test data
# def test(cnn):
# 	# Test the model
# 	cnn.eval()
# 	with torch.no_grad():
# 		correct = 0
# 		total = 0
# 		for images, labels in loaders['test']:
# 			test_output, last_layer = cnn(images)
# 			pred_y = torch.max(test_output, 1)[1].data.squeeze()
# 			accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
# 			pass
# 	print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
# pass

# for model in cnn_models:
# 	test(model)
# 	# Print 10 predictions from test data
# 	sample = next(iter(loaders['test']))
# 	imgs, lbls = sample

# 	actual_number = lbls[:10].numpy()

# 	test_output, last_layer = model(imgs[:10])
# 	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# 	print(f'Prediction number: {pred_y}')
# 	print(f'Actual number: {actual_number}')


# TODO output weight of all layers of every CNN model
# for model in cnn_models:
# 	print(model.conv1[0].weight)

# print('--------------------------------------------')
# # CNN1_2
# model = cnn_models[0]
# print(model.__ne__, 'conv1')
# print(model.conv1[0].weight)
# print(model.__ne__, 'fc1')
# print(model.fc1[0].weight)	# TODO weight NOT fully output by default
# print(model.__ne__, 'fc2')
# print(model.fc2[0].weight)
# print('--------------------------------------------')
# # CNN2_1
# model = cnn_models[1]
# print(model.__ne__, 'conv1')
# print(model.conv1[0].weight)
# print(model.__ne__, 'conv2')
# print(model.conv2[0].weight)
# print(model.__ne__, 'fc1')
# print(model.fc1[0].weight)
# print('--------------------------------------------')
# # CNN3_2
# model = cnn_models[2]
# print(model.__ne__, 'conv1')
# print(model.conv1[0].weight)
# print(model.__ne__, 'conv2')
# print(model.conv2[0].weight)
# print(model.__ne__, 'conv3')
# print(model.conv3[0].weight)
# print(model.__ne__, 'fc1')
# print(model.fc1[0].weight)
# print(model.__ne__, 'fc2')
# print(model.fc2[0].weight)
# print('--------------------------------------------')
# # CNN4_2
# model = cnn_models[3]
# print(model.__ne__, 'conv1')
# print(model.conv1[0].weight)
# print(model.__ne__, 'conv2')
# print(model.conv2[0].weight)
# print(model.__ne__, 'conv3')
# print(model.conv3[0].weight)
# print(model.__ne__, 'conv4')
# print(model.conv4[0].weight)
# print(model.__ne__, 'fc1')
# print(model.fc1[0].weight)
# print(model.__ne__, 'fc2')
# print(model.fc2[0].weight)

train(1, loaders, cnn_models[0], optimizers[0])
print('--------------------------------------------')
# CNN1_2
model = cnn_models[0]
# print(model.__ne__, 'conv1')
# print(model.conv1[0].weight)
torch.set_printoptions(edgeitems=256)
tensor = model.fc1[0].weight.data
# print(tensor)

import numpy as np
np.set_printoptions(edgeitems=256, linewidth=1000000)
numpytensor = tensor.detach().numpy()
# numpytensor = tensor.detach().numpy()
arr = np.array2string(numpytensor, max_line_width=100,separator=',')
print(arr)
print('--------------------------------------------')

f = open("demofile3.txt", "w")
f.write(arr)
f.close()