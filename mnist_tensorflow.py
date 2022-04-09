# https://medium.com/analytics-vidhya/a-brief-study-of-convolutional-neural-network-cnn-using-mnist-digit-recognizer-e054cf8863bf

import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# importing files
train = pd.read_csv('data/mnist-train.csv')
test = pd.read_csv('data/mnist-test.csv')

train

"""**Separating the data and label**"""

train_data = train.loc[:,"pixel0":]
train_label= train.loc[:, "label"]

"""**Convert train data to numpy array** """

train_data = np.array(train_data)
train_label = tf.keras.utils.to_categorical(train_label, num_classes=10, dtype='float32')

"""**Convert test data to numpy array**"""

test_data = test.loc[:, "pixel0":]
test_data = np.array(test_data)

"""**Reshaping the data from (28,28) to (28,28,1).We write '1' here for keras to know its a greyscale image, it would not actually change the number of values.**"""

train_data = train_data.reshape(train_data.shape[0],28,28,1)
test_data  = test_data.reshape(test_data.shape[0],28,28,1)

#Normalize the values betwenn 0 to 1
train_data = train_data/255.0
test_data  = test_data/255.0

"""**DEFINE THE CNN LAYERS**"""

model = tf.keras.models.Sequential
([
	tf.keras.layers.Conv2D(32, (5,5), activation='relu',input_shape=(28,28,1), padding= 'same'),
	tf.keras.layers.Conv2D(32, (5,5), activation = 'relu', padding='same'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same'),
	tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024,activation = 'relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10,activation = 'softmax')
])

model.compile(optimizer = "adam",
	loss='categorical_crossentropy',
	metrics=['accuracy'])

history = model.fit(train_data,train_label,epochs = 25)

model.summary()

"""**Evaluation and Predictions**"""

evaluation = model.evaluate(test_data)
print(evaluation)

predictions = model.predict(test_data)
prediction = []

for i in predictions:
	prediction.append(np.argmax(i))

"""**Storing the predictions**"""

#making a dataframe to save predictions and data values
submission =  pd.DataFrame({
		"ImageId": test.index+1,
		"Label": prediction
	})

submission.to_csv('submission.csv', index=False)

submission

import matplotlib.pyplot as plt
image = train_data[0].reshape(28,28)
plt.imshow(image)
image2 = train_data[0].reshape(30,30)
plt.imshow(image2)

"""**Use Model to Predict Result for Single Example**"""

result = model.predict(np.array([train_data[0]]))
predicted_value = np.argmax(result)
print(predicted_value)