### Section 1 - First, let's import everything we will be needing.
## this is similar to run model but generate and store embeddings in a temp location
from __future__ import print_function, division
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys
from PIL import ImageFile
from data_loader import *
import folder_datasets
import random
import video_transforms as vt
from model_classes import *
from losses import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_mode", "-rm", help="mode of run train/eval/finetune", type=str, default='train')
parser.add_argument("--save_dir", "-sd", help="directory where models to be saved", type=str, default='saves/')
parser.add_argument("--test_model", "-tm", help="model to be tested.",type=str, default='')
parser.add_argument("--test_subj","-ts",  help="name or id of the test subject. ex. subject01 or subject03",type=str, required=True)
parser.add_argument("--data_dir", "-dd", help="location of input embeddings. default to original embd.",type=str, default='cropped_handpatches/')
parser.add_argument("--handshape_dir", "-hdir", help="location of the saved handshape to pretrain embedder.",type=str, default='data_iters/iter2')
parser.add_argument("--num_epochs","-ne",  help="number of epochs.",type=int, default=50)
parser.add_argument("--batch_size","-bs",  help="size of each batch.",type=int, default=16)
parser.add_argument("--learning_rate","-lr",  help="learning rate.",type=float, default=0.001)
parser.add_argument("--save_model",  help="if specified model checkpoint will be saved during training", action='store_true')
parser.add_argument("--gpu_id","-gpu",  help="gpu id.",type=int, default=0)
parser.add_argument("--sample_rate","-sr",  help="sample rate.",type=int, default=20)
parser.add_argument("--all_subs",  help="all subjects in the dataset",type=list, default=['subject{:02d}'.format(i) for i in range(1, 13)])
parser.add_argument("--crop_type","-ct",  help="crop type of patches",type=str, default='both_hand')
parser.add_argument("--random_crop_size","-rcs",  help="random crop size data augmentation.",type=int, default=100)
parser.add_argument("--resize_size","-rs",  help="resize size of images.",type=int, default=100)
# unused options, kept for tracking codes
#parser.add_argument("--model_name","-mn",  help="model name to be saved",type=str, default='')
#parser.add_argument("--fusion_type", "-ft", help="type of fusion.",type=str, default='concat')
#parser.add_argument("--res_file","-resfile",  help="result file name",type=str, default='temp_res')
#parser.add_argument("--verbose",  help="print logs", action='store_true')
#parser.add_argument("--pt_model","-pt",  help="pre trained sign model to be evaluated",type=str, default='')
#parser.add_argument("--resnet_arch","-ra",  help="resnet architecture to be used",type=str, default='resnet18')
args = parser.parse_args()
train_subs = [s for s in args.all_subs if s!=args.test_subj]
test_subs = [args.test_subj]



os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_id)
dataset_mean = [0.52100614, 0.49904737, 0.46344589] # hand patch (140, 100)
dataset_std = [0.25671347, 0.26215637, 0.28081052]

# transforms for video cropped hand data
img_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(100),
        #transforms.Resize(args.resize_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(100),
        #transforms.Resize(args.resize_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(100),
        #transforms.Resize(args.resize_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])
}
## transforms for handshape annotated data
handshape_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(args.random_crop_size),
        #transforms.Resize(args.resize_size),
        transforms.RandomGrayscale(0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(args.random_crop_size),
        #transforms.Resize(args.resize_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(args.random_crop_size),
        #transforms.Resize(args.resize_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ]),
}
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
dset_types = ['train','test']
sub_dicts = dict(zip(dset_types, [[s for s in args.all_subs if s!=args.test_subj], [args.test_subj], [args.test_subj]]))

random_classes = None
num_class_sign = 51
num_class = 41

cross_split = args.test_subj!=''
image_datasets = {x: folder_datasets.ImageFolder(os.path.join(args.handshape_dir),
    handshape_transforms[x], cross_split=cross_split, subs=sub_dicts[x]) for x in dset_types}
image_dset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                           shuffle=True, num_workers=10)
            for x in dset_types}
image_dset_sizes = {x: len(image_datasets[x]) for x in dset_types}
class_names = image_datasets['train'].classes
for i, c in enumerate(class_names):
  print (i, c)
num_class = len(class_names)

print (image_dset_sizes)
dset_loaders = image_dset_loaders
dsets = image_datasets 
dset_sizes = image_dset_sizes

def get_loaded_model(saved_model_loc=''):
  '''
  Returns a loaded model and optimizer
  saved_model_loc : if given state dict is loaded from saved location
  ''' 
  model = HandShapeConvNet(num_class, model_type='')
  if saved_model_loc:
    print ('loading model for test purpose', saved_model_loc)
    model.load_state_dict(torch.load(args.test_model))
  model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
  return model, optimizer


def learn_handshape(model, optimizer, save_model_name=''):
  model.train()
  criterion = nn.CrossEntropyLoss()
  dset_types = ['train', 'test']
  best_acc = 0.0
  eval_only = args.run_mode in ['test','eval', 'val']
  for epoch in range(args.num_epochs):
    for phase in dset_types:
      model.train()
      if eval_only and phase =='train':
        continue
      if phase == 'test':
        model.eval()
      running_loss = 0.0
      running_corrects = 0
      for i, batch_data in enumerate(dset_loaders[phase]):
        input_data, labels = batch_data
        input_data = input_data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
          #print (input_data.size())
          outputs = model(input_data)
          #print (outputs.size())
          #sys.exit()
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          if phase == 'train':
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * input_data.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if i and i%500==0:
          print (i, 'iters completed')
      epoch_loss = running_loss / dset_sizes[phase]
      epoch_acc = running_corrects.double() / dset_sizes[phase]
      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))       
      if phase in ['val', 'test','eval'] and eval_only:
        break
      if phase=='test' and epoch_acc > best_acc:
        best_acc = epoch_acc
        if args.save_model:
          try:
            state_dict = model.module.state_dict()
          except AttributeError:
            state_dict = model.state_dict()
          print ('Saving ..', save_model_name)
          torch.save(state_dict, '{}'.format(save_model_name))
    if phase in ['val', 'test','eval'] and eval_only:
      break


assert args.run_mode=='train' or args.test_model, "Provide test model location"
saved_model_loc = args.test_model               # add arguments in case of teste/eval
model, optimizer = get_loaded_model(saved_model_loc)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.to(device)
save_model_name = os.path.join(args.save_dir, 'handshape-model-{}'.format(args.test_subj))
learn_handshape(model, optimizer, save_model_name=save_model_name)


def imshow(inp, title=None):
  """Imshow for Tensor."""
  print ('imshow inp', inp[:,0,0])
  inp = inp.numpy().transpose((1, 2, 0))
  #mean = np.array([0.485, 0.456, 0.406])
  #std = np.array([0.229, 0.224, 0.225])
  mean = np.array(dataset_mean)
  std = np.array(dataset_std)
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  plt.imshow(inp)
  if title is not None:
      plt.title(title)
  plt.pause(0.001)  # pause a bit so that plots are updated

