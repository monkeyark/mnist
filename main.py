import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# importing files
train = pd.read_csv('data/mnist-train.csv')
test = pd.read_csv('data/mnist-test.csv')

# separating the data and label
train_data = train.loc[:,"pixel0":]
train_label= train.loc[:, "label"]

# convert train and test data to numpy array
train_data = np.array(train_data)
train_label = tf.keras.utils.to_categorical(train_label, num_classes=10, dtype='float32')
test_data = test.loc[:, "pixel0":]
test_data = np.array(test_data)

# reshaping and normalizing the data.
train_data = train_data.reshape(train_data.shape[0],28,28,1)
test_data  = test_data.reshape(test_data.shape[0],28,28,1)
# normalize the values between 0 to 1
train_data = train_data/255.0
test_data  = test_data/255.0

# define CNN layer
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
model.compile(optimizer = "adam", loss='categorical_crossentropy',
metrics=['accuracy'])

# fit the model
history = model.fit(train_data,train_label,epochs = 25)

# check model summary use
model.summary()

# make predictions and save them
predictions = model.predict(test_data)
prediction = []
for i in predictions:
	prediction.append(np.argmax(i))
#making a dataframe to save predictions and data values
submission =  pd.DataFrame({
"ImageId": test.index+1,
"Label": prediction
})
submission.to_csv('submission.csv', index=False)


# use model to predict result
image = train_data[0].reshape(28,28)
plt.imshow(image)
result = model.predict(np.array([train_data[0]]))
predicted_value = np.argmax(result)
print(predicted_value)