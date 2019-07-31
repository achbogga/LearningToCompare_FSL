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
import task_generator_china_drinks as tgcd
import os
import math
import argparse
import random
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition for China Drinks Dataset")
parser.add_argument("--dataset_folder",type = str, default = '/home/caffe/achu/Data/china_drinks/image_data/cropped_images')
parser.add_argument("--event_logs",type = str, default = '/home/caffe/achu/logs/pytorch_china_drinks_FSL_event_logs')
parser.add_argument("--channel_dim",type = int, default = 64)
parser.add_argument("--class_num",type = int, default = 150)
parser.add_argument("--training_samples_per_class",type = int, default = 5)
parser.add_argument("--support_set_samples_per_class",type = int, default = 20)
parser.add_argument("--episode",type = int, default= 10000)
parser.add_argument("--test_episode", type = int, default = 100)
parser.add_argument("--learning_rate", type = float, default = 0.001)
parser.add_argument("--validation_split_percentage", type = float, default = 0.1)
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument("--hidden_unit",type=int,default=512)
parser.add_argument("--image_size",type=int,default=160)
parser.add_argument("--record_training_loss_every_n_episodes", type=int, default = 10)
parser.add_argument("--record_validation_loss_every_n_episodes", type=int, default = 100)
args = parser.parse_args()


def get_cnn_output_dims(W, K, P, S):
    return (int(W-K+2*P)/S)+1
def cnn_final_output_dims(image_size):
    dim1_calc = get_cnn_output_dims(image_size, 3, 0, 1)/2
    dim2_calc = get_cnn_output_dims(dim1_calc, 3, 0, 1)/2
    dim3_calc = get_cnn_output_dims(dim2_calc, 3, 1, 1)
    final_output = get_cnn_output_dims(dim3_calc, 3, 1, 1)
    return final_output
def rn_dims_before_FCN(input_dims):
    dim1_calc = get_cnn_output_dims(input_dims, 3, 1, 1)/2
    final_output = get_cnn_output_dims(dim1_calc, 3, 1, 1)/2
    return final_output

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
	def __init__(self,fcn_size,hidden_unit):
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
		self.fc1 = nn.Linear(fcn_size,hidden_unit)
		self.fc2 = nn.Linear(hidden_unit,1)

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0),-1)
		out = F.relu(self.fc1(out))
		out = torch.sigmoid(self.fc2(out))
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
	writer = SummaryWriter(args.event_logs)
	# Step 1: init data folders
	print("init data folders")
	# init sku folders for dataset construction
	metatrain_sku_folders,metatest_sku_folders = tgcd.china_drinks_sku_folders(data_folder = args.dataset_folder, no_of_training_samples = args.training_samples_per_class, no_of_validation_samples = args.support_set_samples_per_class, validation_split_percentage = args.validation_split_percentage)

	# Step 2: init neural networks
	print("init neural networks")

	cnn_output_dims = cnn_final_output_dims(args.image_size)
	rn_dims = rn_dims_before_FCN(cnn_output_dims)
	fcn_size = args.channel_dim*(rn_dims**2)


	feature_encoder = CNNEncoder()
	relation_network = RelationNetwork(fcn_size,args.hidden_unit)

	feature_encoder.apply(weights_init)
	relation_network.apply(weights_init)

	if torch.cuda.device_count() >= 1:
		print("CUDA devices found", torch.cuda.device_count(), args.gpu)
		feature_encoder = nn.DataParallel(feature_encoder)
		relation_network = nn.DataParallel(relation_network)
	else:
		feature_encoder.cuda(args.gpu)
		relation_network.cuda(args.gpu)

	feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=args.learning_rate)
	feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
	relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=args.learning_rate)
	relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

	if os.path.exists(str("./models/omniglot_feature_encoder_" + str(args.class_num) +"way_" + str(args.training_samples_per_class) +"shot.pkl")):
		feature_encoder.load_state_dict(torch.load(str("/home/caffe/achu/models/china_drinks_feature_encoder_" + str(args.class_num) +"way_" + str(args.training_samples_per_class) +"shot.pkl"), map_location='cuda:0'))
		print("load feature encoder success")
	if os.path.exists(str("./models/omniglot_relation_network_"+ str(args.class_num) +"way_" + str(args.training_samples_per_class) +"shot.pkl")):
		relation_network.load_state_dict(torch.load(str("/home/caffe/achu/models/china_drinks_relation_network_"+ str(args.class_num) +"way_" + str(args.training_samples_per_class) +"shot.pkl"), map_location='cuda:0'))
		print("load relation network success")

	# Step 3: build graph
	print("Training...")

	last_accuracy = 0.0

	for episode in range(args.episode):

		feature_encoder_scheduler.step(episode)
		relation_network_scheduler.step(episode)

		# init dataset
		# sample_dataloader is to obtain previous samples for compare
		# batch_dataloader is to batch samples for training
		degrees = random.choice([0,90,180,270])
		task = tgcd.ChinaDrinksTask(metatrain_sku_folders,args.class_num,args.training_samples_per_class,args.support_set_samples_per_class)
		sample_dataloader = tgcd.get_data_loader(task, image_size = args.image_size, num_per_class=args.training_samples_per_class,split="train",shuffle=False,rotation=degrees)
		batch_dataloader = tgcd.get_data_loader(task, image_size = args.image_size, num_per_class=args.support_set_samples_per_class,split="test",shuffle=True,rotation=degrees)


		# sample datas
		samples,sample_labels = sample_dataloader.__iter__().next()
		batches,batch_labels = batch_dataloader.__iter__().next()

		# calculate features
		sample_features = feature_encoder(Variable(samples).cuda(args.gpu)) # 5x64*5*5
		sample_features = sample_features.view(args.class_num,args.training_samples_per_class,args.channel_dim,cnn_output_dims,cnn_output_dims)
		sample_features = torch.sum(sample_features,1).squeeze(1)
		batch_features = feature_encoder(Variable(batches).cuda(args.gpu)) # 20x64*5*5

		# calculate relations
		# each batch sample link to every samples to calculate relations
		# to form a 100x128 matrix for relation network
		sample_features_ext = sample_features.unsqueeze(0).repeat(args.support_set_samples_per_class*args.class_num,1,1,1,1)
		batch_features_ext = batch_features.unsqueeze(0).repeat(args.class_num,1,1,1,1)
		batch_features_ext = torch.transpose(batch_features_ext,0,1)

		relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,args.channel_dim*2,cnn_output_dims,cnn_output_dims)
		relations = relation_network(relation_pairs).view(-1,args.class_num)

		mse = nn.MSELoss().cuda(args.gpu)
		one_hot_labels = Variable(torch.zeros(args.support_set_samples_per_class*args.class_num, args.class_num).scatter_(1, batch_labels.view(-1,1), 1)).cuda(args.gpu)
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

		if (episode+1)%args.record_training_loss_every_n_episodes == 0:
				print("episode:",episode+1,"loss",loss.data)

		if (episode+1)%args.record_validation_loss_every_n_episodes == 0:

			# test
			print("Testing...")
			total_rewards = 0

			for i in range(args.test_episode):
				degrees = random.choice([0,90,180,270])
				task = tgcd.ChinaDrinksTask(metatest_sku_folders,args.class_num,args.training_samples_per_class,args.training_samples_per_class)
				sample_dataloader = tgcd.get_data_loader(task, image_size = args.image_size, num_per_class=args.training_samples_per_class,split="train",shuffle=False,rotation=degrees)
				test_dataloader = tgcd.get_data_loader(task, image_size = args.image_size, num_per_class=args.training_samples_per_class,split="test",shuffle=True,rotation=degrees)

				sample_images,sample_labels = sample_dataloader.__iter__().next()
				test_images,test_labels = test_dataloader.__iter__().next()

				test_labels = test_labels.cuda()

				# calculate features
				sample_features = feature_encoder(Variable(sample_images).cuda(args.gpu)) # 5x64
				sample_features = sample_features.view(args.class_num,args.training_samples_per_class,args.channel_dim,cnn_output_dims,cnn_output_dims)
				sample_features = torch.sum(sample_features,1).squeeze(1)
				test_features = feature_encoder(Variable(test_images).cuda(args.gpu)) # 20x64

				# calculate relations
				# each batch sample link to every samples to calculate relations
				# to form a 100x128 matrix for relation network
				sample_features_ext = sample_features.unsqueeze(0).repeat(args.training_samples_per_class*args.class_num,1,1,1,1)
				test_features_ext = test_features.unsqueeze(0).repeat(args.class_num,1,1,1,1)
				test_features_ext = torch.transpose(test_features_ext,0,1)

				relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,args.channel_dim*2,cnn_output_dims,cnn_output_dims)
				relations = relation_network(relation_pairs).view(-1,args.class_num)

				test_mse = nn.MSELoss().cuda(args.gpu)
				test_labels = test_labels.cpu()
				test_one_hot_labels = Variable(torch.zeros(args.training_samples_per_class*args.class_num, args.class_num).scatter_(1, test_labels.view(-1,1), 1))
				test_one_hot_labels = test_one_hot_labels.cuda(args.gpu)
				test_loss = test_mse(relations,test_one_hot_labels)
				if (i+1)%args.record_validation_loss_every_n_episodes == 0:
					print("test_episode:",i+1,"loss",test_loss.data)
					writer.add_scalar('Validation loss', test_loss.data, episode+1)
					print ('Validation loss for episode '+str(episode+1)+': '+ str(test_loss.data))

				test_labels = test_labels.cuda()

				_,predict_labels = torch.max(relations.data,1)

				rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(args.class_num*args.training_samples_per_class)]

				total_rewards += np.sum(rewards)

			test_accuracy = total_rewards/1.0/args.class_num/args.training_samples_per_class/args.test_episode

			print("validation accuracy:",test_accuracy)
			writer.add_scalar('Validation accuracy', test_accuracy, episode+1)

			if test_accuracy > last_accuracy:

				# save networks
				torch.save(feature_encoder.state_dict(),str("/home/caffe/achu/models/china_drinks_feature_encoder_" + str(args.class_num) +"way_" + str(args.training_samples_per_class) +"shot.pkl"))
				torch.save(relation_network.state_dict(),str("/home/caffe/achu/models/china_drinks_relation_network_"+ str(args.class_num) +"way_" + str(args.training_samples_per_class) +"shot.pkl"))

				print("save networks for episode:",episode+1)

				last_accuracy = test_accuracy


if __name__ == '__main__':
	main()
