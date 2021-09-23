import sys
import os
import numpy as np
import torch


def pairwise_l2_distance(embs1, embs2):
  """Computes pairwise distances between all rows of embs1 and embs2."""
  #norm1 = tf.reduce_sum(tf.square(embs1), 1)
  #norm1 = tf.reshape(norm1, [-1, 1])
  #norm2 = tf.reduce_sum(tf.square(embs2), 1)
  #norm2 = tf.reshape(norm2, [1, -1])
  
  norm1 = torch.norm(embs1)
  norm1 = torch.reshape(norm1, (-1, 1))
  norm2 = torch.norm(embs2)
  norm2 = torch.reshape(norm2, (1, -1))
  # Max to ensure matmul doesn't produce anything negative due to floating
  # point approximations.
  #dist = tf.maximum(norm1 + norm2 - 2.0 * tf.matmul(embs1, embs2, False, True), 0.0)
  dist = torch.max(norm1 + norm2 - 2.0 * torch.matmul(embs1, embs2), 0.0)

  return dist


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
  """Returns similarity between each all rows of embs1 and all rows of embs2.
  The similarity is scaled by the number of channels/embedding size and
  temperature.
  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.
  Returns:
    similarity: Tensor, [M, N] tensor denoting similarity between embs1 and
      embs2.
  """
  channels = embs1.size()[1]
  # Go for embs1 to embs2.
  if similarity_type == 'cosine':
    similarity = torch.matmul(embs1, embs2.transpose(1,0))
  elif similarity_type == 'l2':
    similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
  else:
    raise ValueError('similarity_type can either be l2 or cosine.')

  # Scale the distance  by number of channels. This normalization helps with
  # optimization.
  similarity /= channels
  # Scale the distance by a temperature that helps with how soft/hard the
  # alignment should be.
  similarity /= temperature

  return similarity


def align_pair_of_sequences(embs1, embs2, similarity_type, temperature):
  max_num_steps = embs1.size()[0]  
  sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)   # 20X20 where 20 is seq len/num frames
  softmaxed_sim_12 = torch.nn.functional.softmax(sim_12, dim=1)
  nn_embs = torch.matmul(softmaxed_sim_12, embs2) # soft nn embeddings 20X100 where 20 seq len, 100 embd len
  #print ('nn_embs', nn_embs.size())
  sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)
  logits = sim_21     # logits values are similariry (as similar as strong logit)
  softmaxed_logits = torch.nn.functional.softmax(logits, dim=1)
  #print (softmaxed_logits)
  #labels = tf.one_hot(tf.range(max_num_steps), max_num_steps)
  labels = torch.from_numpy(np.eye(max_num_steps, dtype=np.float32)[np.array(range(max_num_steps))]).cuda()
  #print (labels.requires_grad)
  #sys.exit()
  return softmaxed_logits, labels

def compute_alignment_loss(embs):
  batch_size = embs.size()[0]
  similarity_type='cosine'
  temperature = 0.1
  logit_list, label_list = [], []
  for i in range(batch_size):
    for j in range(batch_size):
      # We do not align the sequence with itself.
      if i != j:
        logits, labels = \
          align_pair_of_sequences(embs[i], embs[j], similarity_type, temperature)
        logit_list.append(logits)
        label_list.append(labels)
  logits = torch.cat(logit_list, 0)
  labels = torch.cat(label_list, 0)
  #print (logits.size(), labels.size())
  #print (type(logits), type(labels))
  align_loss = -torch.mul(labels, torch.log(logits)).mean()
  return align_loss
    
'''
def main():
  embs1 = torch.from_numpy(np.random.random((20, 100)))
  embs2 = torch.from_numpy(np.random.random((20, 100)))
  logits, labels = align_pair_of_sequences(embs1, embs2, 'cosine',0.1)
  print (logits.size(), labels.size())
if __name__ == "__main__":
  main()
'''
