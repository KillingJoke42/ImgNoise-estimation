from tensorflow.keras import models, layers, optimizers
import numpy as np
from matplotlib import image
import os
import cv2

def load_dataset():
	iter_paths = ["../../../../datasets/Berkeley/saltpepper"]#, "../../../../datasets/Pascal/saltpepper"]

	train_images, train_labels, test_images, test_labels = list(), list(), list(), list()
		
	for iter_path in iter_paths:
		im_list = os.listdir(iter_path)	
		for i in range(2000):
			img = image.imread(iter_path + "/" + im_list[i])
			img = cv2.resize(img, (256, 256))
			params = im_list[i].split("_")
			train_labels.append(float(params[1][1:]))
			train_images.append(img)

	train_images = np.array(train_images).reshape((2000, 256, 256, 1))
	train_labels = np.array(train_labels)
	train_images = train_images / 255.0
	np.save('np_data/train_images_saltpepper.npy', train_images)
	np.save('np_data/train_labels_saltpepper.npy', train_labels)
	del(train_images)
	del(train_labels)

	iter_path = "../../../../datasets/USC/saltpepper"

	im_list = os.listdir(iter_path)
	for i in range(2000):
		img = image.imread(iter_path + "/" + im_list[i])
		img = cv2.resize(img, (256, 256))
		params = im_list[i].split("_")
		test_labels.append(float(params[1][1:]))
		test_images.append(img)

	test_images = np.array(test_images).reshape((2000, 256, 256, 1))
	test_labels = np.array(test_labels)
	test_images = test_images / 255.0
	np.save('np_data/test_images_saltpepper.npy', test_images)
	np.save('np_data/test_labels_saltpepper.npy', test_labels)
	del(test_images)
	del(test_labels)

def train():
	train_images = np.load('np_data/train_images_saltpepper.npy')
	train_labels = np.load('np_data/train_labels_saltpepper.npy')

	model = models.Sequential()
	model.add(layers.Conv2D(32, (5,5), (2,2), activation='relu', input_shape=(256,256,1)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64, (5,5), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64, (5,5), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(128, (5,5), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(128, (3,3), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse'])
	model.fit(train_images, train_labels, epochs=50, validation_split=0.2)

	model.save('models/saltpepper.h5')
	del(train_images)
	del(train_labels)

# def train_v():
# 	train_images = np.load('np_data/train_images_saltpepper.npy')
# 	train_labels_v = np.load('np_data/train_labels_v_saltpepper.npy')

# 	model = models.Sequential()
# 	model.add(layers.Conv2D(32, (5,5), (2,2), activation='relu', input_shape=(256,256,1)))
# 	model.add(layers.Conv2D(32, (5,5), (2,2), activation='relu'))
# 	model.add(layers.MaxPooling2D((2,2)))
# 	model.add(layers.Conv2D(64, (5,5), activation='relu'))
# 	model.add(layers.Conv2D(64, (5,5), activation='relu'))
# 	model.add(layers.MaxPooling2D((2,2)))
# 	model.add(layers.Conv2D(128, (5,5), activation='relu'))
# 	model.add(layers.Conv2D(128, (5,5), activation='relu'))
# 	model.add(layers.MaxPooling2D((2,2)))
# 	model.add(layers.Flatten())
# 	model.add(layers.Dense(64, activation='tanh'))
# 	model.add(layers.Dense(32, activation='tanh'))
# 	model.add(layers.Dense(1, activation='sigmoid'))

# 	optimizer = optimizers.SGD(lr=0.01)
# 	model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
# 	model.fit(train_images, train_labels_v, epochs=50, validation_split=0.2)

# 	model.save('models/saltpepper_v.h5')
# 	del(train_images)
# 	del(train_labels_v)

def test():
	test_images = np.load('np_data/test_images_saltpepper.npy')
	test_labels = np.load('np_data/test_labels_saltpepper.npy')

	model = models.load_model('models/saltpepper.h5')
	loss = model.evaluate(test_images, test_labels)
	print(loss)

	del(test_images)
	del(test_labels)

def image_test():
	model = models.load_model('models/saltpepper.h5')
	img_path = "../../../../datasets/USC/saltpepper"
	img = os.listdir(img_path)[2]
	im = image.imread(img_path + "/" + img)
	params = img.split("_")
	print(params)
	im = im.reshape((1,256,256,1))
	im = im / 255.0
	print(model.predict(im))

# load_dataset()
# train()
# test()
image_test()