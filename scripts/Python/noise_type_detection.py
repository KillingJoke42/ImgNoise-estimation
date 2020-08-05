import numpy as np
from tensorflow.keras import models, layers
from matplotlib import image
import os

class model:
	def __init__(self, train_images=list(), train_labels=list(), test_images=list(), test_labels=list(), model_type="gaussian"):
		self.train_images = train_images
		self.train_labels = train_labels
		
		self.test_images = test_images
		self.test_labels = test_labels

		if model_type in set(["gaussian", 'poisson', 'saltpepper', 'speckle']):
			self.model_type = model_type
		self.model = 0

	def load_data(self, set_name="Berkeley", image_path="../../datasets"):
		if set_name not in set(["Berkeley", "Pascal", "USC"]):
			return
		iter_path = "{}/{}/{}".format(image_path, set_name, self.model_type)
		image_fileset = os.listdir(iter_path)
		for i in range(2000):
			im = image_fileset[i]
			img = image.imread(iter_path + "/" + im)
			params = im.split("_")
			if i < 1000:
				self.train_labels.append([float(params[1][1:]), float(params[2][1:])])
				self.train_images.append(img)
			else:
				self.test_labels.append([float(params[1][1:]), float(params[2][1:])])
				self.test_images.append(img)

		self.train_images = np.array(self.train_images)
		self.train_labels = np.array(self.train_labels)
		self.test_images = np.array(self.test_images)
		self.test_labels = np.array(self.test_labels)
		print(self.train_labels.shape)
		print(self.train_images.shape)
		print(self.test_labels.shape)
		print(self.test_images.shape)

	def save_data(self):
		np.save('train_images_gaussian_lite.npy', self.train_images)
		np.save('train_labels_gaussian_lite.npy', self.train_labels)
		np.save('test_images_gaussian_lite.npy', self.test_images)
		np.save('test_labels_gaussian_lite.npy', self.test_labels)

	def free_mem(self):
		del(self.train_images)
		del(self.train_labels)
		del(self.test_images)
		del(self.test_labels)

	def load_from_npy(self):
		self.train_images = np.load('train_images_gaussian_lite.npy')
		self.train_labels = np.load('train_labels_gaussian_lite.npy')
		self.test_images = np.load('test_images_gaussian_lite.npy')
		self.test_labels = np.load('test_labels_gaussian_lite.npy')

	def get_shape(self):
		print(self.train_labels.shape)
		print(self.train_images.shape)
		print(self.test_labels.shape)
		print(self.test_images.shape)

	def normalize(self):
		self.train_images = self.train_images.reshape((1000, 512, 512, 1))
		self.test_images = self.test_images.reshape((1000, 512, 512, 1))
		self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

	def build_model(self):
		model = models.Sequential()
		model.add(layers.Conv2D(32, (7,7), (2,2), activation='relu', input_shape=(512,512,1)))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(64, (7,7), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(64, (7,7), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(96, (5,5), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(96, (3,3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(128, (3,3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Flatten())
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dense(32, activation='relu'))
		model.add(layers.Dense(2))
		self.model = model

	def model_summary(self):
		self.model.summary()

	def train_model(self):
		self.model.compile(optimizer="adam", loss="mae", metrics=["accuracy"])
		self.model.fit(self.train_images, self.train_labels, epochs=20, validation_split=0.1)

	def save_model(self):
		models.save_model(self.model, 'gaussian_lite.h5')

def main():
	mod = model()
	#mod.train_images = np.load('train_images_gaussian_lite.npy')
	#mod.train_labels = np.load('train_labels_gaussian_lite.npy')
	#mod.load_data()
	mod.load_from_npy()
	mod.normalize()
	mod.get_shape()
	#mod.free_mem()
	mod.build_model()
	#mod.model_summary()
	mod.train_model()

if __name__ == "__main__":
	main()