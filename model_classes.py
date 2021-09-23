import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from torchvision import datasets, models, transforms



'''
works on embedding data
'''
class HandLSTM(nn.Module):
  def __init__(self, input_len, hidden_size, num_layers, num_class,  dropout=0.5, train_type=''):
    super(HandLSTM, self).__init__()
    hidden_size = hidden_size//2            # for bidir
    self.concat = nn.LSTM(2*input_len, hidden_size, num_layers, batch_first=True, dropout=dropout)
    self.lefthand = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
    self.righthand = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True) 
    self.leftsmax = nn.Linear(hidden_size*2, num_class)
    self.rightsmax = nn.Linear(hidden_size*2, num_class)
    self.bothsmax = nn.Linear(2*hidden_size, num_class)
    #train_type='both'
    #assert (train_type in ['left','right','both']), 'train type must be given!'
    assert (train_type in ['left_hand','right_hand','both_hand']), 'train type must be given!'
    self.tr_type = train_type
    self.num_layers = num_layers
    self.hidden_size = hidden_size
  
  def forward(self, x):
    if self.tr_type=='left_hand':
      x = x.chunk(2, -1)[0]
    elif self.tr_type=='right_hand':
      x = x.chunk(2, -1)[1]
    
    effective_batch = x.size(0)
    if self.tr_type=='both_hand':
      #lstm_out, states = self.concat(x)
      #print (x.size())
      #sys.exit()
      x_l = x.chunk(2, -1)[0]
      x_r = x.chunk(2, -1)[1]
      lstm_out, states = self.lefthand(x_l)
      _ , cstate = states
      cstate_dir = cstate.view((self.num_layers, 2, effective_batch, self.hidden_size))[-1]
      #print (cstate_dir.size())
      cstate_l = cstate_dir.transpose(0,1).flatten(-2,-1) # 2xbatchxstate to batchx(2*state)
      #print (cstate_l.size())
      #sys.exit()
      logits_l = self.leftsmax(cstate_l)
      lstm_out, states = self.righthand(x_r)
      _ , cstate = states
      cstate_dir = cstate.view((self.num_layers, 2, effective_batch, self.hidden_size))[-1]
      cstate_r = cstate_dir.transpose(0,1).flatten(-2,-1)
      logits_r = self.rightsmax(cstate_r)
      #cstate_both = torch.cat((cstate_l, cstate_r), dim=-1)
      #print (cstate_both.size())
      #sys.exit()
      #return None, self.bothsmax(cstate_both)
      return None, torch.mean(torch.stack((logits_l, logits_r)), dim=0)  # logit avg
    else:
      lstm_out, states = self.lefthand(x)       # this is okay, cause we are not learning both hands together
    _ , cstate = states
    cstate = cstate.view((self.num_layers, 1, effective_batch, self.hidden_size))[-1][0]
    logits = self.leftsmax(cstate)
    return None, logits
    x_ch = torch.chunk(x, 2, dim=-1)
    if self.tr_type in ['left', 'both']:
      lstm_out, states = self.lefthand(x_ch[0])
      _ , cstate = states
      cstate_left = cstate.view((self.num_layers, 1, effective_batch, self.hidden_size))[-1][0]
      logits_left = self.leftsmax(cstate_left)
      if self.tr_type=='left':
        return None, logits_left
    if self.tr_type in ['right', 'both']:
      lstm_out, states = self.righthand(x_ch[1])
      _ , cstate = states
      cstate_right = cstate.view((self.num_layers, 1, effective_batch, self.hidden_size))[-1][0]
      logits_right = self.rightsmax(cstate_right)
      if self.tr_type=='right':
        return None, logits_right
    cat_input = torch.cat((cstate_left, cstate_right), dim=-1)
    logits = self.bothsmax(cat_input)
    return None, logits
    return None, torch.mean(torch.stack((logits_left, logits_right)), dim=0)






class HandShapeConvNet(nn.Module):
  def __init__(self, num_class,  dropout=0.5, model_type=''):
    """
    This is the hand shape CNN learning model.
    input_len : vector size for backend resnet CNN output, here assumed 2048 for resnet 50 architecture
    hidden_size : linear layer size to produce hand shape image prediction
    model_type : if set as generate, forward is called for left hand data and right hand data sepearately
                  else means now we are training this model using hand shape images
    """
    super(HandShapeConvNet, self).__init__()
    res_model = list(models.resnet50(pretrained=True).children())[:-1]
    emb_len = 2048             # fixed for res-50 model
    self.base_net = nn.Sequential(*res_model)
    self.base_fc = nn.Linear(emb_len, num_class)
    self.model_type = model_type

  def forward(self, x):
    embs = self._get_batch_embeddings(x)
    if self.model_type=='generate':
      return embs, self.base_fc(embs)
    return self.base_fc(embs)

  def _get_batch_embeddings(self, x):
    '''
    Returen embedding sequences for a batch of data
    model : base model network
    x : data
    '''
    batch_size = x.size()    # taking only one, assuming batch size 1
    outputs = self.base_net(x).flatten(-3,-1)
    return outputs


