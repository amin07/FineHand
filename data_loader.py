from __future__ import print_function, division
import collections
import json
import glob
import cv2
import math
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict
from PIL import ImageFile, Image
import random


'''
dataset loads saved embeddings
'''
class HandEmbdDataset(Dataset):

  def __init__(self, root_dir, subs, transform=None, sample_rate=10, train_type='', skip_garbage=True, classes = None, test_data=False, test_subj=''):
    """
    Args:
        root_dir (string): Directory of the files.
        subs (list): subjects to be included in this dataset (test/train)
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    #if test_data:
    #  root_dir = os.path.join(os.path.dirname(root_dir), 'handshape_embedding_cross_sub','aiswarya')
    #mask_dir = os.path.join(root_dir, '{}_frame-labels'.format(test_subj))
    mask_dir = os.path.join(root_dir, '{}_labels'.format(test_subj))
    root_dir = os.path.join(root_dir, '{}'.format(test_subj))
  
    all_files = os.listdir(root_dir)
    #if args.verbose:
    #  print (len(all_files))
    all_files = [f for f in all_files if f.split('-')[-1].split('_')[1] in subs]  
    #if args.verbose:
    #  print (all_files[:5])
    #  print (len(all_files))
    #all_files = [f for f in all_files if f.split('-')[0].split('_')[0]==train_type]  
    all_files = list(set([f.split('-')[-1] for f in all_files]))  # taking the generic last part
    #if args.verbose:
    #  print (len(all_files))
    all_files = [f for f in all_files if classes==None or f.split('_')[0] in classes]
    class_list = sorted(list(set([f.split('-')[-1].split('_')[0] for f in all_files])))  
    self.label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
    self.temp2label = dict([(c[0], c[1]) for c in enumerate(class_list)])
    self.file_list = all_files
    self.root_dir = root_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.sample_rate = sample_rate
    self.sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    self.skip_garbage = skip_garbage
 
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    
    samples = [np.zeros((self.sample_rate, 2048),dtype=np.float32), np.zeros((self.sample_rate, 2048), dtype=np.float32)]
    hands = ['left_hand', 'right_hand']
    #samples = []
    for i, h in enumerate(hands):
      curr_file = h+'-'+self.file_list[idx]
      mask_file = os.path.join(self.mask_dir, curr_file.split('.')[0]+'-masks.npy')
      #print (curr_file, mask_file)
      #print (os.path.exists(mask_file), os.path.exists((os.path.join(self.root_dir, curr_file))))
      try:
        label_masks = np.load(mask_file)
        sample = np.load(os.path.join(self.root_dir, curr_file))
        #print ('here', sample.shape, label_masks.shape)
        #print (label_masks)
        #print (samples[0].shape, samples[1].shape)
        if self.skip_garbage:
          sample = sample[label_masks!=1000]
        #print (sample.shape)
        sample = sample[self.sampling_f(self.sample_rate, sample.shape[0])]
        #print ('here')
        #samples.append(sample)
        samples[i] = sample
      except Exception as e:
        #print ('continueing', curr_file)
        #print (e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #print(exc_type, fname, exc_tb.tb_lineno)
        continue
    #sys.exit()
    #print (len(samples))
    return np.concatenate(samples, axis=-1), self.label2temp[self.file_list[idx].split('-')[-1].split('_')[0]]

  def get_label_dicts(self):
    return dict(self.label2temp), dict(self.temp2label)

'''
Return both hand data in concat fasihon
'''
class RGBBothDataset(Dataset):

  def __init__(self,root_dir='', subs=[], transform=None, img_transform=None, cons_classes=None, crop_window=(100,100), crop_type='', tr_model=None, sample_rate=20):
    """
    Args:
        subs = considered subjects for this dataset
        transform (callable, optional): Optional transform to be applied
            on a sample.
        img_transform (callable, optional): transform that applies to individual frames of the video
    """
    print ('considering subs', subs)
    assert (root_dir!=''), 'Must provide root dir!'
    vid_file_names = os.listdir(root_dir)
    vid_file_names = [f for f in vid_file_names if f.split('-')[-1].split('_')[1] in subs]
    if cons_classes:
      vid_file_names = [f for f in vid_file_names if f.split('-')[-1].split('_')[0] in cons_classes]
    if crop_type:
      vid_file_names = [f for f in vid_file_names if f.split('-')[0] == 'left_hand']
      
    print (vid_file_names[:5])
    #vid_file_names = vid_file_names[:500]
    #random.shuffle(vid_file_names)
    self.transform = transform
    self.img_transform = img_transform
    class_list = sorted(list(set([f.split('-')[-1].split('_')[0] for f in vid_file_names])))
    self.label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
    self.temp2label = dict([(c[0], c[1]) for c in enumerate(class_list)])
    self.tr_model = tr_model
    self.sr = sample_rate
    self.sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    self.file_list = [os.path.join(root_dir, f) for f in vid_file_names]
  def __len__(self):
    return len(self.file_list)
  def __getitem__(self, idx):
    vid_file = self.file_list[idx]
    ignore_flag = 0    
    sample = self._get_img_array(vid_file)
    r_sample  = self._get_img_array(vid_file.replace('left_hand', 'right_hand'))
    
    len_sample = len(sample)
    r_len_sample = len(r_sample)
    sample = [sample[i] for i in self.sampling_f(self.sr, len_sample)]  # sampling
    r_sample = [r_sample[i] for i in self.sampling_f(self.sr, r_len_sample)] 

    if self.transform:
      sample = self.transform(sample)
      r_sample = self.transorm(r_sample)
    if self.img_transform:
      sample = [self.img_transform(s) for s in sample]
      r_sample = [self.img_transform(s) for s in r_sample]
    sample = torch.stack(sample, 0)
    r_sample = torch.stack(r_sample, 0)
    
    class_name = os.path.basename(vid_file).split('-')[-1].split('_')[0]
    minlen = min(len(sample), len(r_sample))
    sample, r_sample = sample[:minlen], r_sample[:minlen]
    try:
      return torch.stack((sample,r_sample), 0), self.label2temp[class_name], vid_file
      #return torch.stack((sample,r_sample), 0), self.label2temp[class_name]
    except:
      minlen = min(len(sample), len(r_sample))
      sample, r_sample = sample[:minlen], r_sample[:minlen]
      return torch.stack((sample,r_sample), 0), self.label2temp[class_name], vid_file
      #return torch.stack((sample,r_sample), 0), self.label2temp[class_name]

  def _get_img_array(self, file_name):
    vid = cv2.VideoCapture(file_name)
    img_ar = []
    while True:
      ret, frame = vid.read()
      if ret==False:
        break
      #img_ar.append(frame-self.mean_img)    # subtracting mean img
      frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #this is necessary cause most model trained on RGB order, if you want to show using opencv, just convert to BGR before that ********
      img_ar.append(frame) 
    vid.release()
    return np.array(img_ar)

  
  def get_label_dicts(self):
    return dict(self.label2temp)
'''
Create dataset for isogd: only rgb frames
**** THIS version is for ASL
Read from already cropped hand patch data
'''
class RGBDataset(Dataset):

  def __init__(self,root_dir='', subs=[], transform=None, img_transform=None, cons_classes=None, crop_window=(100,100), crop_type='', tr_model=None, sample_rate=20):
    """
    Args:
        subs = considered subjects for this dataset
        transform (callable, optional): Optional transform to be applied
            on a sample.
        img_transform (callable, optional): transform that applies to individual frames of the video
    """
    print ('considering subs', subs)
    assert (root_dir!=''), 'Must provide root dir!'
    vid_file_names = os.listdir(root_dir)
    vid_file_names = [f for f in vid_file_names if f.split('-')[-1].split('_')[1] in subs]
    if cons_classes:
      vid_file_names = [f for f in vid_file_names if f.split('-')[-1].split('_')[0] in cons_classes]
    if crop_type:
      vid_file_names = [f for f in vid_file_names if f.split('-')[0] == crop_type]
      
    print (vid_file_names[:5])
    #vid_file_names = vid_file_names[:100]
    self.transform = transform
    self.img_transform = img_transform
    class_list = sorted(list(set([f.split('-')[-1].split('_')[0] for f in vid_file_names])))
    self.label2temp = dict([(c[1], c[0]) for c in enumerate(class_list)])
    self.temp2label = dict([(c[0], c[1]) for c in enumerate(class_list)])
    self.tr_model = tr_model
    self.sr = sample_rate
    self.sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    self.file_list = [os.path.join(root_dir, f) for f in vid_file_names]
  def __len__(self):
    return len(self.file_list)
  def __getitem__(self, idx):
    vid_file = self.file_list[idx]
    ignore_flag = 0    
    sample = self._get_img_array(vid_file)
    len_sample = len(sample)
    sample = [sample[i] for i in self.sampling_f(self.sr, len_sample)]  # sampling

    if self.transform:
      sample = self.transform(sample)
    if self.img_transform:
      sample = [self.img_transform(s) for s in sample]
    sample = torch.stack(sample, 0)
    
    class_name = os.path.basename(vid_file).split('-')[-1].split('_')[0]
    return sample, self.label2temp[class_name]

  def _get_img_array(self, file_name):
    vid = cv2.VideoCapture(file_name)
    img_ar = []
    while True:
      ret, frame = vid.read()
      if ret==False:
        break
      #img_ar.append(frame-self.mean_img)    # subtracting mean img
      frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #this is necessary cause most model trained on RGB order, if you want to show using opencv, just convert to BGR before that ********
      img_ar.append(frame) 
    vid.release()
    return np.array(img_ar)

  '''
  Given the video file location, this method gets the openpose poses
  '''
  def _get_poses(self, vid_file):
    #dir_name = os.path.dirname(vid_file)
    #folder_name = os.path.basename(dir_name)
    #json_loc = os.path.join(os.path.dirname(os.path.dirname(dir_name)),'frame_data/{}_frames_jsons'.format(folder_name, folder_name))
    
    sub_name = os.path.basename(os.path.dirname(vid_file))
    vid_name = os.path.basename(vid_file).split('.')[0]
    json_files = sorted(glob.glob(os.path.join(self.pose_dir, '{}_json'.format(sub_name), vid_name+'*')))

    crop_win = self.crop_window
    crop_h, crop_w = crop_win
    center_x , center_y = crop_w//2, crop_h//2
    sample = self._get_img_array(vid_file)

    poses = []
    for i, jf in enumerate(json_files):
      frame_pose = []
      try:
        with open(jf, 'r') as f:
          json_dat = json.load(f)['people'][0]    # top dict has keys: version, people
          frame_pose = [json_dat['pose_keypoints_2d'],json_dat['hand_left_keypoints_2d'],json_dat['hand_right_keypoints_2d']]
          poses.append(frame_pose)

      except Exception as e:
        #print (e)
        continue
    return poses
  
  def get_label_dicts(self):
    return dict(self.label2temp)




'''
Create dataset for isogd: only rgb frames
'''
class IsoGDRGBDataset(Dataset):

  def __init__(self, root_dir, label_loc, transform=None, img_transform=None, crop_window=(100,100), crop_type='body', cons_classes=range(60)):
    """
    Args:
        root_dir (string): Directory of the files. Must be ended with train, test or valid,
        That's how it would know what type of dataset
        transform (callable, optional): Optional transform to be applied
            on a sample.
        img_transform (callable, optional): transform that applies to individual frames of the video
    """
    self.crop_type=crop_type
    self.crop_window = crop_window
    all_files = os.listdir(root_dir)
    files = []
    self.file_list = []
    '''
    # files have to populate in a different way to keep consistent with isogd_hand_crop
    for f in all_files:
      files += [os.path.join(root_dir, f, fn) for fn in os.listdir(os.path.join(root_dir, f)) if 'M' in fn.split('_')]        # only taking rgb names, depth and json will be created on fly
    self.file_list = files
    '''
    self.root_dir = root_dir
    self.transform = transform
    self.img_transform = img_transform
    self.label_dict = {}  
    file_locs = []
    class_dict = defaultdict(lambda : [])
    with open(label_loc, 'r') as f:
      for line in f:
        parts = line.strip().split(' ')
        if int(parts[-1]) not in cons_classes:
          continue
        self.label_dict[os.path.basename(parts[0]).split('.')[0].split('_')[-1]] = int(parts[-1])-1
        class_dict[int(parts[-1])].append(parts)
    print (class_dict.keys())
    
    print (root_dir)
    for k, v in class_dict.items():
      ss = sorted(v)[5:20]       # taking 5th to 20th sample
      for s in ss:
        self.file_list.append(os.path.join(root_dir, *(s[0].split('/')[1:])))
    #if cons_classes:
    #  self.file_list = [f for f in files if self.label_dict[os.path.basename(f).split('.')[0].split('_')[-1]] in cons_classes] 
    self.label2class = {cons_classes[i]: i for i in range(len(cons_classes))}
    #print (self.file_list[:5])
    #print (len(self.label_dict.keys()))
    #sys.exit()
  def __len__(self):
    return len(self.file_list)
  def __getitem__(self, idx):
    vid_file = self.file_list[idx]
    ignore_flag = 0    
    ####### cuttin start ################
    sample = self._get_img_array(vid_file)
    poses = self._get_poses(vid_file)
    frame_h, frame_w = sample.shape[1], sample.shape[2]
    crop_h, crop_w = self.crop_window
    if self.crop_type=='body':
      joint_id = 1
      if len(poses):
        origin_poses = [fp[0][joint_id*3:(joint_id+1)*3] for fp in poses]
        center_x = int(sum([t[0] for t in origin_poses])/len(origin_poses))
        center_y = int(sum([t[1] for t in origin_poses])/len(origin_poses))
        
      # cutting a 200X200 patch centered at center_x, center_y
      half_h, half_w = crop_h//2, crop_w//2
      sx, sy = max(0, center_y-half_h), max(0, center_x-half_w)
      if sx+crop_h>240:
        sx = sx - ((sx+crop_h)-240)
      if sy+crop_w>320:
        sy = sy - ((sy+crop_w)-320)

      sample = np.array([s[sx:sx+crop_h,sy:sy+crop_w,:] for s in sample])    
    else:
      # 1 for left, 2 for right
      joint_id = None
      joint_id = 1 if self.crop_type=='left_hand' else 2
      assert(joint_id!=None, "crop type not legal")      

      #print (len(poses), len(sample))
      hand_poses = np.array([t[joint_id] for t in poses])
      hand_poses = np.reshape(hand_poses, (hand_poses.shape[0], 21, 3))
      origin_poses = np.mean(hand_poses, axis=1)
      temp_sample, temp_origin_poses = sample, origin_poses
      filtered_data = [t for t in zip(origin_poses, sample) if not (t[0][0]==0. or t[0][1]==0.)]
      sample = [t[1] for t in filtered_data]
      origin_poses = np.array([t[0] for t in filtered_data])
      
      hand_motion = ([np.sqrt(sum([(t[0]-t[1])**2 for t in zip(origin_poses[0,:2], cf[:2])])) for cf in origin_poses])
      if len(hand_motion):
        hand_motion = sum(hand_motion)/len(hand_motion)
      else:
        hand_motion = 0.
      if hand_motion<10.0: #this case we ignore the sample
        ignore_flag = 1
        sample, origin_poses = temp_sample, temp_origin_poses

      #print (hand_motion)
      center_x = [int(t[0]) for t in origin_poses]
      center_y = [int(t[1]) for t in origin_poses]
      half_h, half_w = crop_h//2, crop_w//2
      sx = [max(0, cy-half_h) for cy in center_y]
      sy = [max(0, cx-half_w) for cx in center_x]
      for i in range(len(sx)):
        if sx[i]+crop_h>frame_h:
          sx[i] = sx[i] - ((sx[i]+crop_h)-frame_h)
        if sy[i]+crop_w>frame_w:
          sy[i] = sy[i] - ((sy[i]+crop_w)-frame_w)
    
      hand_sample = []
      for t in zip(sample, sx, sy):
        #print (t[0].shape, t[1], t[2])
        hand_sample.append(t[0][t[1]:t[1]+crop_h, t[2]:t[2]+crop_w,:])
      sample=np.array(hand_sample)
    ################ cutting end ################

    original_sample = sample.copy()
    #print (vid_file)
    if self.transform:
      sample = self.transform(sample)
    if self.img_transform:
      sample = [self.img_transform(s) for s in sample]
    
    
    sample = torch.stack(sample, 0)
    #return sample, self.label_dict[os.path.basename(vid_file).split('.')[0].split('_')[-1]]
    return sample, self.label_dict[os.path.basename(vid_file).split('.')[0].split('_')[-1]], ignore_flag, vid_file, original_sample

  def _get_img_array(self, file_name):
    vid = cv2.VideoCapture(file_name)
    img_ar = []
    while True:
      ret, frame = vid.read()
      if ret==False:
        break
      frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #this is necessary cause most model trained on RGB order, if you want to show using opencv, just convert to BGR before that ********
      img_ar.append(frame) 
    vid.release()
    return np.array(img_ar)

  '''
  Given the video file location, this method gets the openpose poses
  '''
  def _get_poses(self, vid_file):
    dir_name = os.path.dirname(vid_file)
    folder_name = os.path.basename(dir_name)
    json_loc = os.path.join(os.path.dirname(os.path.dirname(dir_name)),'frame_data/{}_frames_jsons'.format(folder_name, folder_name))
    json_files = sorted(glob.glob(os.path.join(json_loc, os.path.basename(vid_file).split('.')[0]+'*')))

    crop_win = self.crop_window
    crop_h, crop_w = crop_win
    center_x , center_y = crop_w//2, crop_h//2
    sample = self._get_img_array(vid_file)

    poses = []
    for i, jf in enumerate(json_files):
      frame_pose = []
      try:
        with open(jf, 'r') as f:
          json_dat = json.load(f)['people'][0]    # top dict has keys: version, people
          frame_pose = [json_dat['pose_keypoints_2d'],json_dat['hand_left_keypoints_2d'],json_dat['hand_right_keypoints_2d']]
          poses.append(frame_pose)

      except Exception as e:
        #print (e)
        continue
    return poses
  
  def get_label_dicts(self):
    return dict(self.name2class)
