import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_tpu as tgtpu
import os
import math
import argparse
import random
# Hyper Parameters

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

CLASS_NUM = 100
SAMPLE_NUM_PER_CLASS = 20
QUERY_NUM_PER_CLASS = 20
EPISODE = 1001
TEST_EPISODE = 100
LEARNING_RATE = 0.005
NO_OF_TPU_CORES = 8
GPU = 0
HIDDEN_UNIT = 256
IMAGE_SIZE = 28
CHANNEL_DIM = 64
DIMS = int(cnn_final_output_dims(IMAGE_SIZE))
RN_DIMS = int(rn_dims_before_FCN(DIMS))
FCN_SIZE = int(CHANNEL_DIM*(RN_DIMS**2))
DATASET_FOLDER = '/home/caffe/achu/Data/china_drinks/image_data_train/cropped_images/'
VALIDATION_SPLIT_PERCENTAGE = 0.1

class CNN_Plus_RNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNN_Plus_RNEncoder, self).__init__()
        self.cnn_layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.cnn_layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.cnn_layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.cnn_layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        
        #RelationNetwork
        self.rn_layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.rn_layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.rn_fc1 = nn.Linear(FCN_SIZE,HIDDEN_UNIT)
        self.rn_fc2 = nn.Linear(HIDDEN_UNIT,1)

    def forward(self, samples, batches):
        #print (samples.size(), batches.size())
        sample_features = self.cnn_layer1(samples)
        sample_features = self.cnn_layer2(sample_features)
        sample_features = self.cnn_layer3(sample_features)
        sample_features = self.cnn_layer4(sample_features)
        cnn_features = sample_features
        
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,CHANNEL_DIM, DIMS, DIMS)
        sample_features = torch.sum(sample_features,1).squeeze(1)
        
        #batch features calculated from test for RN
        batch_features = batches
        batch_features = self.cnn_layer1(batch_features)
        batch_features = self.cnn_layer2(batch_features)
        batch_features = self.cnn_layer3(batch_features)
        batch_features = self.cnn_layer4(batch_features)
        cnn_batch_features = batch_features

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat((QUERY_NUM_PER_CLASS)*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        
        #print ('batch_features_ext: ', batch_features_ext.size())
        #print ('sample_features_ext: ', sample_features_ext.size())
        
        #[(no_of_query_images*(no_of_classes**2)), [CHANNEL_dim*2], dims, dims]
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2)
        #print ('relation_pairs after concat: ', relation_pairs.size())
        relation_pairs = relation_pairs.view(-1,CHANNEL_DIM*2,DIMS,DIMS)
        #print ('relation_pairs before RN view: ', relation_pairs.size())
        relation_pairs = self.rn_layer1(relation_pairs)
        relation_pairs = self.rn_layer2(relation_pairs)
        #print ('relation_pairs after rn layer 2: ', relation_pairs.size())
        relation_pairs = relation_pairs.view(relation_pairs.size(0),-1)
        #print ('relation_pairs before FCN: ', relation_pairs.size())
        relation_pairs = F.relu(self.rn_fc1(relation_pairs))
        relation_pairs = torch.sigmoid(self.rn_fc2(relation_pairs))
        return relation_pairs
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

def main_single_model():
    # Step 1: init data folders
    print("init data folders", flush=True)
    # init character folders for dataset construction
    metatrain_character_folders,metatest_character_folders = tgtpu.china_drinks_sku_folders(DATASET_FOLDER, SAMPLE_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, VALIDATION_SPLIT_PERCENTAGE)

    # Step 2: init neural networks
    print("init neural networks")

    relation_network = CNN_Plus_RNEncoder()
    relation_network.apply(weights_init)
    relation_network.cuda(GPU)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
    
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    relation_network = nn.DataParallel(relation_network)
    #else:
    #    relation_network.cuda(GPU)


    if os.path.exists(str("./models/omniglot_full_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_full_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location='cuda:0'))
        print("load full relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0,90,180,270])
        task = tgtpu.ChinaDrinksTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
        sample_batch_dataloader = tgtpu.get_data_loader(task, image_size = IMAGE_SIZE, sample_num_per_class=SAMPLE_NUM_PER_CLASS, query_num_per_class = QUERY_NUM_PER_CLASS,train_shuffle=False, query_shuffle=True, rotation=degrees, num_workers = NO_OF_TPU_CORES)

        # sample datas
        samples,sample_labels,batches,batch_labels = sample_batch_dataloader.__iter__().next()        
        relation_scores = relation_network(Variable(samples).cuda(GPU), Variable(batches).cuda(GPU))
        relations = relation_scores.view(-1,CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(QUERY_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)


        # training

        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)
        relation_network_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data, flush = True)

        if (episode+1)%5000 == 0:

            # test
            print("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                degrees = random.choice([0,90,180,270])
                task = tgtpu.ChinaDrinksTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS,)
                sample_test_dataloader = tgtpu.get_data_loader(task,IMAGE_SIZE,sample_num_per_class=SAMPLE_NUM_PER_CLASS,query_num_per_class=QUERY_NUM_PER_CLASS,train_shuffle=False,test_shuffle= True,rotation=degrees, num_workers = NO_OF_TPU_CORES)        

                sample_images, sample_labels, test_images, test_labels = sample_test_dataloader.__iter__().next()
                
                print (sample_images.size, test_images.size)

                test_labels = test_labels.cuda()
                
                relations = relation_network(Variable(samples).cuda(GPU), Variable(batches).cuda(GPU)).view(-1,CLASS_NUM)

                _,predict_labels = torch.max(relations.data,1)

                rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE

            print("validation accuracy:",test_accuracy)

            if test_accuracy > last_accuracy:

                # save networks                
                #torch.save(relation_network.state_dict(),str("./models/omniglot_full_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy
main_single_model()