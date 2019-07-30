#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("--dataset_folder",type = str, default = 64)
parser.add_argument("--feature_dim",type = int, default = 64)
parser.add_argument("--class_num",type = int, default = 5)
parser.add_argument("--training_samples_per_class",type = int, default = 5)
parser.add_argument("--support_set_samples_per_class",type = int, default = 15)
parser.add_argument("--episode",type = int, default= 5000)
parser.add_argument("--test_episode", type = int, default = 1000)
parser.add_argument("--learning_rate", type = float, default = 0.001)
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument("--hidden_unit",type=int,default=128)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.training_samples_per_class
QUERY_NUM_PER_CLASS = args.support_set_samples_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

class CNNEncoder(nn.Module):
	"""docstring for ClassName"""
	def __init__(self):
		super(CNNEncoder, self).__init__()
		self.layer1 = nn.Sequential(
						nn.Conv2d(1,64,kernel_size=3,padding=0),
						nn.BatchNorm2d(64, momentum=1, affine=True),
						nn.ReLU(),
						nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
						nn.Conv2d(64,64,kernel_size=3,padding=0),
						nn.BatchNorm2d(64, momentum=1, affine=True),
						nn.ReLU(),
						nn.MaxPool2d(2))
		self.layer3 = nn.Sequential(
						nn.Conv2d(64,64,kernel_size=3,padding=1),
						nn.BatchNorm2d(64, momentum=1, affine=True),
						nn.ReLU())
		self.layer4 = nn.Sequential(
						nn.Conv2d(64,64,kernel_size=3,padding=1),
						nn.BatchNorm2d(64, momentum=1, affine=True),
						nn.ReLU())

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		#out = out.view(out.size(0),-1)
		return out # 64

class RelationNetwork(nn.Module):
	"""docstring for RelationNetwork"""
	def __init__(self,feature_size,hidden_size):
		super(RelationNetwork, self).__init__()
		self.layer1 = nn.Sequential(
						nn.Conv2d(128,64,kernel_size=3,padding=1),
						nn.BatchNorm2d(64, momentum=1, affine=True),
						nn.ReLU(),
						nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
						nn.Conv2d(64,64,kernel_size=3,padding=1),
						nn.BatchNorm2d(64, momentum=1, affine=True),
						nn.ReLU(),
						nn.MaxPool2d(2))
		self.fc1 = nn.Linear(feature_size,hidden_size)
		self.fc2 = nn.Linear(hidden_size,1)

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0),-1)
		out = F.relu(self.fc1(out))
		out = F.sigmoid(self.fc2(out))
		return out

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('BatchNorm') != -1:
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif classname.find('Linear') != -1:
		n = m.weight.size(1)
		m.weight.data.normal_(0, 0.01)
		m.bias.data = torch.ones(m.bias.data.size())

def main():
	writer = SummaryWriter('/home/caffe/achu/logs/pytorch_omniglot_FSL.log')
	# Step 1: init data folders
	print("init data folders")
	# init character folders for dataset construction
	metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders(data_folder = '../datas/omniglot_resized/', no_of_training_samples = args.training_samples_per_class, no_of_validation_samples = args.support_set_samples_per_class)

	# Step 2: init neural networks
	print("init neural networks")

	feature_encoder = CNNEncoder()
	relation_network = RelationNetwork(FEATURE_DIM,HIDDEN_UNIT)

	feature_encoder.apply(weights_init)
	relation_network.apply(weights_init)

	if torch.cuda.device_count() >= 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		feature_encoder = nn.DataParallel(feature_encoder)
		relation_network = nn.DataParallel(relation_network)
	else:
		feature_encoder.cuda(GPU)
		relation_network.cuda(GPU)

	feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
	feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
	relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
	relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

	if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
		feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location='cuda:0'))
		print("load feature encoder success")
	if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
		relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location='cuda:0'))
		print("load relation network success")

	# Step 3: build graph
	print("Training...")

	last_accuracy = 0.0

	for episode in range(EPISODE):

		feature_encoder_scheduler.step(episode)
		relation_network_scheduler.step(episode)

		# init dataset
		# sample_dataloader is to obtain previous samples for compare
		# batch_dataloader is to batch samples for training
		degrees = random.choice([0,90,180,270])
		task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
		sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
		batch_dataloader = tg.get_data_loader(task,num_per_class=QUERY_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)


		# sample datas
		samples,sample_labels = sample_dataloader.__iter__().next()
		batches,batch_labels = batch_dataloader.__iter__().next()

		# calculate features
		sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5
		sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,5,5)
		sample_features = torch.sum(sample_features,1).squeeze(1)
		batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5

		# calculate relations
		# each batch sample link to every samples to calculate relations
		# to form a 100x128 matrix for relation network
		sample_features_ext = sample_features.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
		batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
		batch_features_ext = torch.transpose(batch_features_ext,0,1)

		relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
		relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

		mse = nn.MSELoss().cuda(GPU)
		one_hot_labels = Variable(torch.zeros(QUERY_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
		loss = mse(relations,one_hot_labels)


		# training

		feature_encoder.zero_grad()
		relation_network.zero_grad()

		loss.backward()

		torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
		torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

		feature_encoder_optim.step()
		relation_network_optim.step()

		writer.add_scalar('Training loss', loss.data, episode+1)

		if (episode+1)%100 == 0:
				print("episode:",episode+1,"loss",loss.data)

		if (episode+1)%5000 == 0:

			# test
			print("Testing...")
			total_rewards = 0

			for i in range(TEST_EPISODE):
				degrees = random.choice([0,90,180,270])
				task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS,)
				sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
				test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

				sample_images,sample_labels = sample_dataloader.__iter__().next()
				test_images,test_labels = test_dataloader.__iter__().next()

				test_labels = test_labels.cuda()

				# calculate features
				sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
				sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,5,5)
				sample_features = torch.sum(sample_features,1).squeeze(1)
				test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

				# calculate relations
				# each batch sample link to every samples to calculate relations
				# to form a 100x128 matrix for relation network
				sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
				test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
				test_features_ext = torch.transpose(test_features_ext,0,1)

				relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
				relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

				_,predict_labels = torch.max(relations.data,1)

				rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]

				total_rewards += np.sum(rewards)

			test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE

			print("validation accuracy:",test_accuracy)
			writer.add_scalar('Validation accuracy', test_accuracy, episode+1)

			if test_accuracy > last_accuracy:

				# save networks
				torch.save(feature_encoder.state_dict(),str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
				torch.save(relation_network.state_dict(),str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

				print("save networks for episode:",episode)

				last_accuracy = test_accuracy





if __name__ == '__main__':
	main()
