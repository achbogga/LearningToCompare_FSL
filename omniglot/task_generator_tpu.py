#Modified task_generator with a single loader which can deliver two inputs for the TPU training.

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
	npimg = img.numpy()
	plt.axis("off")
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()

class Rotate(object):
	def __init__(self, angle):
		self.angle = angle
	def __call__(self, x, mode="reflect"):
		x = x.rotate(self.angle)
		return x

def china_drinks_sku_folders(data_folder = '/home/caffe/achu/Data/china_drinks/image_data/cropped_images', no_of_training_samples = 10, no_of_validation_samples = 20, validation_split_percentage = 0.1):

	sku_folders = [os.path.join(data_folder, family, sku) \
				for family in os.listdir(data_folder) \
				if os.path.isdir(os.path.join(data_folder, family)) \
				for sku in os.listdir(os.path.join(data_folder, family))]
	random.seed(1)
	random.shuffle(sku_folders)

	print ('Total SKUS in the dataset: ', len(sku_folders))
	req_samples = no_of_training_samples+no_of_validation_samples
	sku_folders = [folder for folder in sku_folders if len(os.listdir(folder))>=req_samples]
	print ('Total SKUS in the dataset with '+str(req_samples)+' samples: ', len(sku_folders))

	val_split = int(validation_split_percentage*len(sku_folders))
	train_split = len(sku_folders) - val_split

	metatrain_sku_folders = sku_folders[:train_split]
	print ('Train split SKUS: ', len(metatrain_sku_folders))
	#print ('Debug: ', metatrain_sku_folders[0])
	metaval_sku_folders = sku_folders[train_split:]
	print ('Val split SKUS: ', len(metaval_sku_folders))

	return metatrain_sku_folders,metaval_sku_folders

class ChinaDrinksTask(object):
	# This class is for task generation for both meta training and meta testing.
	# For meta training, we use all 20 samples without valid set (empty here).
	# For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
	# If set num_samples = 20 and chracter_folders = metatrain_sku_folders, we generate tasks for meta training
	# If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
	def __init__(self, sku_folders, num_classes, train_num,test_num):

		self.sku_folders = sku_folders
		self.num_classes = num_classes
		self.train_num = train_num
		self.test_num = test_num

		#print ('Sampling :', num_classes, ' from ', len(sku_folders))

		class_folders = random.sample(self.sku_folders,self.num_classes)
		selected_skus = [folder.split('/')[-1] for folder in class_folders]
		labels_ar = np.array(range(len(selected_skus)))
		labels = dict(zip(selected_skus, labels_ar))
		samples = dict()

		self.train_roots = []
		self.test_roots = []
		for c in class_folders:

			temp = [os.path.join(c, x) for x in os.listdir(c)]
			samples[c] = temp
			#print ('Debug0: ', len(temp))

			self.train_roots += samples[c][:train_num]
			self.test_roots += samples[c][train_num:(train_num+test_num)]
		#print ('Debug: ', self.train_roots[0])

		self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
		self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

	def get_class(self, sample):
		#print ('Debug: ', sample)
		return str(sample.split('/')[-2])


class FewShotDataset(Dataset):

	def __init__(self, task, image_size = 160, split='train', transform=None, target_transform=None):
		self.transform = transform # Torch operations on the input image
		self.target_transform = target_transform
		self.task = task
		self.split = split
		self.image_size = image_size
		self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
		self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

	def __len__(self):
		return len(self.image_roots)

	def __getitem__(self, idx):
		raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class ChinaDrinks(FewShotDataset):

	def __init__(self, *args, **kwargs):
		super(ChinaDrinks, self).__init__(*args, **kwargs)

	def __getitem__(self, idx):
		try:
			image_root = self.image_roots[idx]
			image = Image.open(image_root)
			image = image.convert('L')
			image = image.resize((self.image_size,self.image_size), resample=Image.LANCZOS) # per Chelsea's implementation
			#image = np.array(image, dtype=np.float32)
			if self.transform is not None:
				image = self.transform(image)
			label = self.labels[idx]
			if self.target_transform is not None:
				label = self.target_transform(label)
			return image, label
		except IndexError:
			print ('Index Error: ', idx, len(self.image_roots))
			return None, None

class ClassBalancedSampler(Sampler):
	''' Samples 'num_inst' examples each from 'num_cl' pools
		of examples of size 'num_per_class' '''

	def __init__(self, sample_num_per_class, query_num_per_class, num_cl, train_num_inst, test_num_inst, train_shuffle=False, test_shuffle=True):
		self.sample_num_per_class = sample_num_per_class
		self.query_num_per_class = query_num_per_class
		self.num_cl = num_cl
		self.train_num_inst = train_num_inst
		self.test_num_inst = test_num_inst
		self.train_shuffle = train_shuffle
		self.test_shuffle = test_shuffle

	def __iter__(self):
		# return a single list of indices, assuming that items will be grouped by class
		if self.shuffle:
			batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
		else:
			batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
		batch = [item for sublist in batch for item in sublist]

		if self.shuffle:
			random.shuffle(batch)
		return iter(batch)

	def __len__(self):
		return 1

def get_data_loader(task, image_size = 160, sample_num_per_class=1, query_num_per_class=1, train_shuffle=False, query_shuffle=True, rotation=0, num_workers = 3):
	# NOTE: batch size here is # instances PER CLASS
	normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

	dataset = ChinaDrinks(task, split=split, image_size = image_size, transform=transforms.Compose([Rotate(rotation),transforms.ToTensor()]))#,normalize]))

	
	sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, task.test_num, train_shuffle=shuffle, test_shuffle=test_shuffle)
	
	loader = DataLoader(dataset, batch_size=(sample_num_per_class+query_num_per_class)*task.num_classes, sampler=sampler, num_workers=num_workers)

	return loader

