"""
Plots logits 
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch

def plot_logits(logits, model_name):
  """
  Arguments:
    logit_list: tensor of shape (batch_size, seq_len, vocab_size)
  Returns:
    Nothing; writes a plot to disk
  """
  logits = logits.reshape(-1, logits.size()[-1])
  partition_functions = torch.log(1+1/(torch.sum(torch.exp(logits), dim=1)+1e-30))
  #logits = torch.max(logits, dim=1).values
  partition_functions = list(sorted(partition_functions.tolist()))
  print(partition_functions)
  plt.hist(partition_functions, label=model_name)
  plt.title('Distribution of token-level KL divergence')
  plt.ylabel('Frequency of occurrence')
  plt.xlabel('KL divergence for one token')
  plt.legend()
  plt.savefig('kls.png')
  plt.clf()

  partition_functions = torch.sum(torch.exp(logits), dim=1) + 1e-30
  #logits = torch.max(logits, dim=1).values
  partition_functions = list(sorted(partition_functions.tolist()))[:-50]
  print(partition_functions)
  plt.hist(partition_functions, label=model_name)
  plt.title('Distribution of partition function Z')
  plt.ylabel('Frequency of occurrence')
  plt.xlabel('KL divergence for one token')
  plt.legend()
  plt.savefig('logits.png')


