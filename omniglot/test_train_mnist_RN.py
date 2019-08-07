import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=256,
    momentum=0.5,
    lr=0.001,
    target_accuracy=50,
    log_steps=10,
    num_epochs=1000)

from common_utils import TestCase, run_tests
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import unittest
import task_generator_tpu as tgtpu
import random
import math
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
QUERY_NUM_PER_CLASS = SAMPLE_NUM_PER_CLASS
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
DATASET_FOLDER = '/home/caffe/achu/Data/china_drinks/prod_train'
VALIDATION_SPLIT_PERCENTAGE = 0.2

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

        self.apply(weights_init)


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



def train_mnist():
  torch.manual_seed(1)
  # Step 1: init data folders
  print("init data folders", flush=True)
  # init character folders for dataset construction
  metatrain_character_folders,metatest_character_folders = tgtpu.china_drinks_sku_folders(DATASET_FOLDER, SAMPLE_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, VALIDATION_SPLIT_PERCENTAGE)

  
  devices = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
  # Scale learning rate to num cores
  lr = FLAGS.lr * len(devices)
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model_parallel = dp.DataParallel(CNN_Plus_RNEncoder, device_ids=devices)


  degrees = random.choice([0,90,180,270])
  train_task = tgtpu.ChinaDrinksTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
  train_sample_batch_dataloader = tgtpu.get_data_loader(train_task, image_size = IMAGE_SIZE, sample_num_per_class=SAMPLE_NUM_PER_CLASS, query_num_per_class = QUERY_NUM_PER_CLASS,train_shuffle=False, query_shuffle=True, rotation=degrees, num_workers = NO_OF_TPU_CORES)

  test_task = tgtpu.ChinaDrinksTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS)
  test_sample_test_dataloader = tgtpu.get_data_loader(test_task,IMAGE_SIZE,sample_num_per_class=SAMPLE_NUM_PER_CLASS,query_num_per_class=QUERY_NUM_PER_CLASS,train_shuffle=False,query_shuffle= True,rotation=degrees, num_workers = NO_OF_TPU_CORES)

  def train_loop_fn(model, loader, device, context):
    relation_network = model
    #relation_network.apply(weights_init)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
    mse = nn.MSELoss()
    tracker = xm.RateTracker()

    for x, (samples,sample_labels,batches,batch_labels) in loader:



      relation_network_scheduler.step(episode)

      relation_network.zero_grad()
      #relation_network_optim.zero_grad()
      relation_scores = relation_network(Variable(samples), Variable(batches))
      relations = relation_scores.view(-1,CLASS_NUM)
      one_hot_labels = Variable(torch.zeros(QUERY_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1))
      loss = mse(relations,one_hot_labels)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)
      xm.optimizer_step(relation_network_optim)
      tracker.add(FLAGS.batch_size)
      print ('Debug: ', x, loss.item())
      if x % FLAGS.log_steps == 0:
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(device, x, loss.item(),
                                                        tracker.rate()))

  def test_loop_fn(model, loader, device, context):
    relation_network = model
    total_rewards = 0
    for x, (samples,sample_labels,batches,batch_labels) in loader:
      relation_scores = relation_network(Variable(samples), Variable(batches))
      relations = relation_scores.view(-1,CLASS_NUM)
      _,predict_labels = torch.max(relations.data,1)
      rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]
      total_rewards += np.sum(rewards)

    test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE
    print('[{}] Accuracy={:.2f}%'.format(device,
                                         100*test_accuracy))
    return test_accuracy

  accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_sample_batch_dataloader)
    accuracies = model_parallel(test_loop_fn, test_sample_test_dataloader)
    accuracy = sum(accuracies) / len(devices)
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy * 100.0


class TrainMnist(TestCase):

  def tearDown(self):
    super(TrainMnist, self).tearDown()
    if FLAGS.tidy and os.path.isdir(FLAGS.datadir):
      shutil.rmtree(FLAGS.datadir)

  def test_accurracy(self):
    self.assertGreaterEqual(train_mnist(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
