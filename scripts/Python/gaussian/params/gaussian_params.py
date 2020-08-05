from tensorflow.keras import models, layers, optimizers
import numpy as np
from matplotlib import image
import os

def load_dataset():
	iter_path = "../../../../datasets/Berkeley/gaussian"

	train_images, train_labels_m, train_labels_v, test_images, test_labels_m, test_labels_v = list(), list(), list(), list(), list(), list()
		
	for im in os.listdir(iter_path):
		img = image.imread(iter_path + "/" + im)
		params = im.split("_")
		train_labels_m.append(float(params[1][1:]))
		train_labels_v.append(float(params[2][1:]))
		train_images.append(img)

	iter_path = "../../../../datasets/Pascal/gaussian"

	for im in os.listdir(iter_path):
		img = image.imread(iter_path + "/" + im)
		params = im.split("_")
		test_labels_m.append(float(params[1][1:]))
		test_labels_v.append(float(params[2][1:]))
		test_images.append(img)

	train_images = np.array(train_images).reshape((2000, 256, 256, 1))
	train_labels_m, train_labels_v = np.array(train_labels_m), np.array(train_labels_v)
	test_images = np.array(test_images).reshape((2000, 256, 256, 1))
	test_labels_m, test_labels_v = np.array(test_labels_m), np.array(test_labels_v)

	test_images, train_images = test_images / 255.0, train_images / 255.0

	print(train_labels_m.shape, train_labels_v.shape)
	print(train_images.shape)
	print(test_labels_v.shape, test_labels_m.shape)
	print(test_images.shape)

	np.save('np_data/train_images_gaussian.npy', train_images)
	np.save('np_data/test_images_gaussian.npy', test_images)
	np.save('np_data/train_labels_m_gaussian.npy', train_labels_m)
	np.save('np_data/train_labels_v_gaussian.npy', train_labels_v)
	np.save('np_data/test_labels_m_gaussian.npy', test_labels_m)
	np.save('np_data/test_labels_v_gaussian.npy', test_labels_v)

	del(train_images)
	del(test_images)
	del(train_labels_m)
	del(train_labels_v)
	del(test_labels_m)
	del(test_labels_v)

def train_m():
	train_images = np.load('np_data/train_images_gaussian.npy')
	train_labels_m = np.load('np_data/train_labels_m_gaussian.npy')

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
	model.add(layers.Dense(1, activation='tanh'))

	optimizer = optimizers.SGD(lr=0.01)
	model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
	model.fit(train_images, train_labels_m, epochs=50, validation_split=0.2)

	model.save('models/gaussian_m.h5')
	del(train_images)
	del(train_labels_m)

def train_v():
	train_images = np.load('np_data/train_images_gaussian.npy')
	train_labels_v = np.load('np_data/train_labels_v_gaussian.npy')

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

	optimizer = optimizers.SGD(lr=0.01)
	model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
	model.fit(train_images, train_labels_v, epochs=50, validation_split=0.2)

	model.save('models/gaussian_v.h5')
	del(train_images)
	del(train_labels_v)

def test_v():
	test_images = np.load('np_data/test_images_gaussian.npy')
	test_labels_v = np.load('np_data/test_labels_v_gaussian.npy')

	model = models.load_model('models/gaussian_v.h5')
	loss = model.evaluate(test_images, test_labels_v)
	print(loss)

	del(test_images)
	del(test_labels_v)

def test_m():
	test_images = np.load('np_data/test_images_gaussian.npy')
	test_labels_m = np.load('np_data/test_labels_m_gaussian.npy')

	model = models.load_model('models/gaussian_m.h5')
	loss = model.evaluate(test_images, test_labels_m)
	print(loss)

	del(test_images)
	del(test_labels_m)

def image_test():
	model = models.load_model('models/gaussian_v.h5')
	model2 = models.load_model('models/gaussian_m.h5')
	img_path = "../../../../datasets/USC/gaussian"
	img = os.listdir(img_path)[2]
	im = image.imread(img_path + "/" + img)
	params = img.split("_")
	print(params)
	im = im.reshape((1,256,256,1))
	im = im / 255.0
	print(model.predict(im))
	print(model2.predict(im))

# load_dataset()
# train_m()
# train_v()
# test_m()
# test_v()
image_test()