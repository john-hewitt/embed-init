import matplotlib.pyplot as plt
import math
import re
plt.rcParams["figure.figsize"] = (4.2,2.2)

def get_scores(path):
  print()
  print(path)
  lines = open(path).read().split('\n')
  lines = list(filter(lambda x: "{'eval_loss'" in x, lines))
  print(lines)
  lines = [re.search(r'{.*}', x).group() for x in lines]
  print(lines)
  lines = list(map(lambda x: math.exp(float(x.split(' ')[1][:-1])), lines))
  print(lines)
  return(lines)

#x = [100,200,300,400,500,600,700,800,900,1000]

def make_plot(avg_emb, default, zeros, name, ylim):
  l = min(len(avg_emb), len(default), len(zeros))
  avg_emb = avg_emb[:l]
  default = default[:l]
  zeros = zeros[:l]
  x = [20*i for i in range(len(avg_emb))]
  plt.plot(x, avg_emb, label='AvgEmb')
  plt.plot(x, default, label='default')
  plt.plot(x, zeros, label='zeros')
  plt.title('Wikitext adaptation perplexity for {}'.format(name))
  plt.xlabel('Gradient steps')
  plt.xlabel('Perplexity')
  plt.legend()
  plt.ylim(ylim)
  plt.savefig(name+'.png', dpi=200)
  plt.clf()
  plt.cla()

#eleuther_avg_emb = get_scores('adapt-avg_emb-EleutherAI-gpt-neo-125M-seed0')
#eleuther_default = get_scores('adapt-default-EleutherAI-gpt-neo-125M-seed0')
#eleuther_zeros =   get_scores('adapt-zeros-EleutherAI-gpt-neo-125M-seed0')

gpt2_avg_emb = [(x1+x2/2) for x1, x2 in zip(get_scores('logs/ppl-avg_emb-gpt2-seed2'),
  get_scores('log3/ppl-avg_emb-gpt2-seed1'))]
gpt2_default = [(x1+x2/2) for x1, x2 in zip(get_scores('log3/ppl-default-gpt2-seed2'),
  get_scores('log3/ppl-default-gpt2-seed1'))]
gpt2_zeros =   [(x1+x2/2) for x1, x2 in zip(get_scores('log3/ppl-zeros-gpt2-seed2'),
  get_scores('log3/ppl-zeros-gpt2-seed1'))]

gpt2_medium_avg_emb = [(x1+x2/2) for x1, x2 in zip(get_scores('logs/ppl-avg_emb-gpt2-medium-seed1'),
  get_scores('logs/ppl-avg_emb-gpt2-medium-seed2'))]
gpt2_medium_default =[(x1+x2/2) for x1, x2 in zip(get_scores('log3/ppl-default-gpt2-medium-seed2'),
  get_scores('log3/ppl-default-gpt2-medium-seed1'))]
gpt2_medium_zeros =  [(x1+x2/2) for x1, x2 in zip(get_scores('log3/ppl-zeros-gpt2-medium-seed2'),
    get_scores('log3/ppl-zeros-gpt2-medium-seed1'))]

gpt2_large_avg_emb = get_scores('logs/ppl-avg_emb-gpt2-large-seed2')
gpt2_large_default = get_scores('log3/ppl-default-gpt2-large-seed2')
gpt2_large_zeros =   get_scores('log3/ppl-zeros-gpt2-large-seed2')

eleuther_avg_emb = [(x1+x2/2) for x1, x2 in zip(get_scores('logs/ppl-avg_emb-EleutherAI-gpt-neo-125M-seed1'),
  get_scores('logs/ppl-avg_emb-EleutherAI-gpt-neo-125M-seed2'))]
eleuther_default = [(x1+x2/2) for x1, x2 in zip(get_scores('log3/ppl-default-EleutherAI-gpt-neo-125M-seed1'),
  get_scores('log3/ppl-default-EleutherAI-gpt-neo-125M-seed2'))]
eleuther_zeros =   [(x1+x2/2) for x1, x2 in zip(get_scores('log3/ppl-zeros-EleutherAI-gpt-neo-125M-seed1'),
  get_scores('log3/ppl-zeros-EleutherAI-gpt-neo-125M-seed2'))]

make_plot(gpt2_avg_emb, gpt2_default, gpt2_zeros, name='gpt2', ylim=(30,100))
make_plot(gpt2_medium_avg_emb, gpt2_medium_default, gpt2_medium_zeros, name='gpt2-medium', ylim=(25,75))
make_plot(gpt2_large_avg_emb, gpt2_large_default, gpt2_large_zeros, name='gpt2-large', ylim=(15,35))
make_plot(eleuther_avg_emb, eleuther_default, eleuther_zeros, name='eleuther-125M', ylim=(-2000,100000))
