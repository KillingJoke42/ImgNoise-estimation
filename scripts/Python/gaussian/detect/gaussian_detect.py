from tensorflow.keras import models, layers
import numpy as np
import os
from matplotlib import image
import cv2
import random
#from sklearn.utlis import shuffle

train_img_dirs = {'gaussian':['../../../../datasets/Berkeley/gaussian', '../../../../datasets/Pascal/gaussian'],
					'original':['../../../../datasets/Berkeley/orig', '../../../../datasets/Pascal/orig']}

test_img_dirs = {'gaussian':'../../../../datasets/USC/gaussian', 'original':'../../../../datasets/USC/orig'}

def prep_dataset():
	train_images, train_labels, test_images, test_labels = list(), list(), list(), list()

	for key in train_img_dirs.keys():
		for dir_ in train_img_dirs[key]:
			img_list = os.listdir(dir_)
			for i in range(200):
				im = image.imread(dir_ + "/" + img_list[i])
				if key=="original":
					im = cv2.resize(im, (256,256))
				train_images.append(im)
				train_labels.append(0 if key=="original" else 1)

	for key in test_img_dirs.keys():
		dir_ = test_img_dirs[key]
		img_list = os.listdir(dir_)
		for i in range(100):
			im = image.imread(dir_ + "/" + img_list[i])
			if key=="original":
					im = cv2.resize(im, (256,256))
			test_images.append(im)
			test_labels.append(0 if key=="original" else 1)

	print(len(train_images), len(test_images))

	train_images, test_images = np.array(train_images), np.array(test_images)
	train_labels, test_labels = np.array(train_labels), np.array(test_labels)
	train_images, test_images = train_images / 255.0, test_images / 255.0
	train_images, test_images = train_images.reshape((800, 256, 256, 1)), test_images.reshape((200, 256, 256, 1))

	mapIndexPosition = list(zip(train_images, train_labels))
	random.shuffle(mapIndexPosition)
	train_images, train_labels = zip(*mapIndexPosition)

	mapIndexPosition = list(zip(test_images, test_labels))
	random.shuffle(mapIndexPosition)
	test_images, test_labels = zip(*mapIndexPosition)

	np.save('np_data/train_images_gaussian_detect.npy', train_images)
	np.save('np_data/train_labels_gaussian_detect.npy', train_labels)
	np.save('np_data/test_images_gaussian_detect.npy', test_images)
	np.save('np_data/test_labels_gaussian_detect.npy', test_labels)

	del(train_images)
	del(train_labels)
	del(test_images)
	del(test_labels)

def train():
	train_images = np.load('np_data/train_images_gaussian_detect.npy')
	train_labels = np.load('np_data/train_labels_gaussian_detect.npy')

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
	model.add(layers.Dense(2, activation='softmax'))

	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	model.fit(train_images, train_labels, epochs=20, validation_split=0.1)

	model.save('models/gaussian_detect.h5')

	del(train_images)
	del(train_labels)

def test():
	test_images = np.load('np_data/test_images_gaussian_detect.npy')
	test_labels = np.load('np_data/test_labels_gaussian_detect.npy')

	model = models.load_model('models/gaussian_detect.h5')

	loss, acc = model.evaluate(test_images, test_labels)
	print(loss, acc)