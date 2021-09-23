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
from sklearn.metrics import confusion_matrix
from model_classes import *
from losses import *
import argparse
from einops import rearrange

# arguments section
parser = argparse.ArgumentParser()
parser.add_argument("--run_mode", "-rm", help="mode of run train/eval/finetune", type=str, default='train')
parser.add_argument("--save_dir", "-sd", help="directory where models to be saved", type=str, default='saves/')
parser.add_argument("--test_model", "-tm", help="model to be tested.",type=str, default='')
parser.add_argument("--fusion_type", "-ft", help="type of fusion.",type=str, default='concat')
parser.add_argument("--test_subj","-ts",  help="name or id of the test subject.",type=str, default=None)
parser.add_argument("--data_dir", "-dd", help="location of input embeddings. default to original embd.",type=str, default='cropped_handpatches/')
parser.add_argument("--num_epochs","-ne",  help="number of epochs.",type=int, default=50)
parser.add_argument("--batch_size","-bs",  help="size of each batch.",type=int, default=16)
parser.add_argument("--learning_rate","-lr",  help="learning rate.",type=float, default=0.001)
parser.add_argument("--save_model",  help="if specified model checkpoint will be saved during training", action='store_true')
parser.add_argument("--verbose",  help="print logs", action='store_true')
parser.add_argument("--gpu_id","-gpu",  help="gpu id.",type=int, default=0)
parser.add_argument("--sample_rate","-sr",  help="sample rate.",type=int, default=10)
parser.add_argument("--all_subs",  help="all subjects in the dataset",type=list, default=['subject{:02d}'.format(i) for i in range(1, 13)])
parser.add_argument("--crop_type","-ct",  help="crop type of patches",type=str, default='both_hand')
parser.add_argument("--res_file","-resfile",  help="result file name",type=str, default='temp_res')
parser.add_argument("--random_crop_size","-rcs",  help="random crop size data augmentation.",type=int, default=100)
parser.add_argument("--model_name","-mn",  help="model name to be saved",type=str, default='')
parser.add_argument("--hand_cnn","-hcnn",  help="cnn hand feature extractor.",type=str, required=True)
parser.add_argument("--freeze_embedder",  help="freeze embedder cnn model in case of sign rec", action='store_true')

#parser.add_argument("--pt_model","-pt",  help="pre trained sign model to be evaluated",type=str, default='')

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

num_class_sign = 51
num_class = 41


avail_classes = None # none means all classes will be taken
dsets = {}
x = 'test'
dsets[x] = RGBBothDataset(root_dir=args.data_dir, subs=sub_dicts[x], img_transform=img_transforms[x], cons_classes=avail_classes, sample_rate=args.sample_rate, crop_type=args.crop_type)

x = 'train'
dsets[x] = RGBBothDataset(root_dir=args.data_dir, subs=sub_dicts[x], img_transform=img_transforms[x], cons_classes=avail_classes, sample_rate=args.sample_rate, crop_type=args.crop_type)
# batch_size always 1 since this is extracting features and writing to a temp location 
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=1, shuffle=False, num_workers=10, pin_memory=True) for x in dset_types}
dset_sizes = {x: len(dsets[x]) for x in dset_types}

print (dset_sizes)

def train_model_on_embedding(input_loc):
  print (torch.cuda.current_device())
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print ('device', device)
  input_path = os.path.join(input_loc)
  dset_types = ['train','test']
  dset_subs = {'train':train_subs, 'test':test_subs}
  #dsets = {x:HandEmbdDataset(input_path, dset_subs[x], sample_rate=sample_rate, train_type=args.model_type) for x in dset_types}

  dsets = {}
  dsets['test'] = HandEmbdDataset(input_path, dset_subs['test'], sample_rate=args.sample_rate, train_type=args.crop_type, test_data=True, test_subj=args.test_subj)

  label2temp, temp2label = dsets['test'].get_label_dicts()  # if test dataset in cs, get class err
  num_class = len(label2temp)
  print (input_path, 'temp input path for embedding')
  dsets['train'] = HandEmbdDataset(input_path, dset_subs['train'], sample_rate=args.sample_rate, train_type=args.crop_type, classes = label2temp.keys(), test_subj=args.test_subj)

  dataloaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
                for x in dset_types}
  dataset_sizes = {x: len(dsets[x]) for x in dset_types}
  print (dataset_sizes)
  ds = dsets['train'][0]
  print (ds[0].shape)
  print (ds[0].sum())
  print (ds[1])
  #print (ds[2])

  input_len = ds[0].shape[-1]//2
  label2temp, temp2label = dsets['train'].get_label_dicts()  # if test dataset in cs, get class err
  num_class = len(label2temp)
  if args.verbose:
    for c in label2temp.keys():
      print (c)
    print (num_class)
    for arg in vars(args):
      print (arg, getattr(args, arg))
  
  model = HandLSTM(input_len, 1024, 2, num_class,  dropout=0.5, train_type=args.crop_type)


  if args.run_mode in ['finetune', 'test']:
    model.load_state_dict(torch.load('{}/{}'.format(args.save_dir, save_model)))
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  assert(args.run_mode!=''), 'invalid run_mode!'
  eval_only = args.run_mode in ['test', 'eval']
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  criterion = nn.CrossEntropyLoss()
  model.to(device)

  since = time.time()
  num_epochs = args.num_epochs
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in dset_types:
        if phase == 'train':
            #scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        y_true, y_pred = [], []
        for inputs, labels in dataloaders[phase]:
            #print (inputs.size(), inputs.sum())
            #print (labels)
            inputs = inputs.float().to(device)      # float() is necessary for 32 vs 64 bit problem
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                _, outputs = model(inputs)
                #print (outputs.size())
                #sys.exit()
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                if phase!='train':
                  y_true += labels.tolist()
                  y_pred += preds.tolist()
                # backward + optimize only if in training phase
                if phase == 'train' and not eval_only:
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        
        # deep copy the model
        if not eval_only and phase == 'test' and epoch_acc > best_acc:
            print ('Best accuracy aquired!')
            best_acc = epoch_acc  
        print('{} Loss: {:.4f} Acc: {:.4f} ## Max Test Acc so far: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_acc))


'''
Given a trained model, this produces embeddings 
and store them in temp location so that those can be used
in lstm training in later
'''
def write_embeddings(model):
  if args.freeze_embedder:
    print ('freezing everything')
    for name, param in model.named_parameters():
        param.requires_grad=False

  tempdir = tempfile.TemporaryDirectory()    # chaingig args.embd_out to temp location
  args.embd_out = tempdir.name
  
  
  emb_out = os.path.join(args.embd_out, args.test_subj)
  label_out = os.path.join(args.embd_out, args.test_subj+'_labels')
  if not os.path.exists(emb_out):
    os.makedirs(emb_out)
    print ('creating temp location', emb_out)
  if not os.path.exists(label_out):
    os.makedirs(label_out)  
    print ('creating temp location', label_out)
  


  dset_types=['train', 'test']
  model.eval()
  for epoch in range(1):
    for phase in dset_types:
      running_loss = 0.0
      running_corrects = 0
      for i, batch_data in enumerate(dset_loaders[phase]):
        input_data, labels, fn = batch_data
        sign_name = os.path.basename(fn[0])
        #print (input_data.size())
        input_data = rearrange(input_data, 'b h t c hi wi -> (b h t) c hi wi')
        #print (input_data.size())
        input_data = input_data.to(device)    # input data is of (Bx2xT)x3x100x100
        labels = labels.to(device)
        all_embs, all_outs = model(input_data)          # (Bx2xTx2048) 
        #print (all_embs.size(), all_outs.size())
        all_embs = rearrange(all_embs, '(b h t) e -> b h t e', b=1, h=2)    # B always 1 because we need file name later
        all_outs = rearrange(all_outs, '(b h t) c -> b h t c', b=1, h=2) 
        #print (all_embs.size(), all_outs.size())
        embs_l, embs_r = all_embs[:,0], all_embs[:,1]
        outputs_l, outputs_r = all_outs[:,0], all_outs[:,1]
        #sys.exit()
        
        #embs, outputs_l, outputs_r = model(input_data)
        #embs_l, embs_r = torch.chunk(embs, 2, dim=-1)
        
        
        _, preds_l = torch.max(outputs_l, -1)
        
        narr_l =  np.array((preds_l.cpu().data.numpy()))[0] # modified
        narr_l =  np.array((preds_l.cpu().data.numpy()))
        narr_l[narr_l==35] = 1000 # rest pos
        narr_l[narr_l==7] = 1000 # garbage
        
        _, preds_r = torch.max(outputs_r, -1)
        #narr_r =  np.array((preds_r.cpu().data.numpy()))[0] # modified
        narr_r =  np.array((preds_r.cpu().data.numpy()))
        narr_r[narr_r==35] = 1000 # rest pos
        narr_r[narr_r==7] = 1000 # garbage
        
        sign_name = os.path.basename(fn[0]).split('.')[0]
        if i and i%500==0:
          print ('(temp loc)writing sign #', i, sign_name) 
        left_write = embs_l.detach().cpu().numpy()
        #print ('writing check', left_write.sum(), left_write.shape)
        np.save(os.path.join(emb_out, sign_name), left_write)
        #sys.exit()
        right_write = embs_r.detach().cpu().numpy()
        #print (left_write.shape, right_write.shape)
        #sys.exit()
        #np.save(os.path.join(emb_out, sign_name.replace('left_hand', 'right_hand',1)), embs_r.detach().cpu().numpy()[0])
        np.save(os.path.join(emb_out, sign_name.replace('left_hand', 'right_hand',1)), right_write)
        np.save(os.path.join(label_out, sign_name+'-masks'), narr_l)
        np.save(os.path.join(label_out, sign_name.replace('left_hand', 'right_hand',1)+'-masks'), narr_r)
    ###
  train_model_on_embedding(args.embd_out)
  print ('cleaning ', args.embd_out)
  tempdir.cleanup()



## load the hand-shape CNN for feature extraction
hand_model = HandShapeConvNet(num_class, model_type='generate')   # this is a feature generator
saved_model_loc = args.hand_cnn
pt_dict = torch.load('{}'.format(saved_model_loc))
hand_model.load_state_dict(pt_dict)
print ('loading hand shape CNN done!')
hand_model.to(device)

write_embeddings(hand_model)

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

