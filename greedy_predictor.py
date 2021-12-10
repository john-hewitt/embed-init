"""
Computes efficient-qa official metrics on NQ-open
"""

import os
import tempfile
import urllib.request
from tqdm import tqdm
import torch
import json
import subprocess
#with urllib.request.urlopen('http://www.example.com/') as f:
#      html = f.read().decode('utf-8')

dev_sets = {
    'nq': 'natural-questions/nq_open/NQ-open.efficientqa.dev.1.1.jsonl',
    'wq': 'wq-dev.jsonl'
    }

NEW_TOKEN = '[NEW]'

wikitext_newtokens_list = ['<unk>', '@-@', '@.@', '@,@', 'became', 'several', 'began', 'following', 'took', 'included', 'moved', 'considered', 'career', 'came', 'continued', 'songs', 'Division', 'become', 'however', 'further', 'gave', 'Dylan', 'tropical', 'returned', 'reached', 'noted', 'throughout', 'across', 'himself', 'performed', 'miles', 'television', 'areas', 'previous', 'died', 'U.S.', 'various', 'County', 'caused', 'characters', 'others', 'construction', '’', 'remained', 'located', 'scored', 'completed', 'Kingdom', 'novel', 'least', 'followed', 'appeared', 'island', 'Cup', 'originally', 'addition', 'hurricane', 'southern', 'critics', 'eventually', 'taken', 'eastern', 'decided', 'Battalion', 'believed', 'Carey', 'poem', 'generally', 'Federer', 'towards', 'elements', 'weeks', 'brought', 'Brigade', 'No.', 'Wales', 'highway', 'featured', 'aircraft', 'tour', 'appearance', 'earlier', 'victory', 'operations', 'troops', 'attempt', 'wanted', '19th', 'northern', 'possible', 'A.', 'saying', 'whom', 'Missouri', 'Island', 'praised', 'remains', 'winds', 'increased', 'previously']

#DEV_SET_PATH = 'https://raw.githubusercontent.com/efficientqa/nq-open/master/NQ-open.dev.jsonl'
PYTHON7_LOCATION = '/u/scr/johnhew/jag-code/adding-tokens/efficientqa-eval/bin/python3'
#with urllib.request.urlopen(DEV_SET_PATH) as f:
#      html = f.read().decode('utf-8')
#      with open(os.path.basename(

def compute_nq_metrics(model, dataset, tokenizer, path, test_dataset, num_beams):
  DEV_SET_PATH = dev_sets[test_dataset]

  # First, download the dev set if you don't have it because :')
  #if not os.path.exists(os.path.basename(DEV_SET_PATH)):
  #  with open(os.path.basename(DEV_SET_PATH), 'w') as fout:
  #    with urllib.request.urlopen(DEV_SET_PATH) as fin:
  #      html = fin.read().decode('utf-8')
  #      fout.write(html)
  # Next, generate a prediction for each element in the dev set and write to disk
  device = list(model.parameters())[0].device
  #sep_id = tokenizer(NEW_TOKEN)['input_ids'][0]
  sep_id = tokenizer(NEW_TOKEN)['input_ids'][0]
  with open(path, 'w') as fout:
    #with tempfile.TemporaryFile(mode='w', encoding='utf-8') as tmpfile:
    for step, raw_inputs in tqdm(enumerate(dataset)):
      question = tokenizer(raw_inputs['question'])
      inputs = {
          'input_ids': torch.tensor([ question['input_ids'] + [sep_id]], device=device)
          }
      #print(tokenizer.convert_ids_to_tokens(inputs['input_ids'])
      outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_length=50, num_beams=num_beams)
      just_predicted_answer = outputs.sequences[0].tolist()[1:]
      just_predicted_answer = just_predicted_answer[1:] if just_predicted_answer[0] == sep_id else just_predicted_answer
      just_predicted_answer = just_predicted_answer if just_predicted_answer[-1] == sep_id else just_predicted_answer + [sep_id]
      just_predicted_answer = just_predicted_answer[just_predicted_answer.index(sep_id)+1:]
      just_predicted_answer = just_predicted_answer[:just_predicted_answer.index(sep_id)]
      
      prediction = {
          'question': raw_inputs['question'],
          'prediction': tokenizer.decode(just_predicted_answer)
          }
      fout.write(json.dumps(prediction) +'\n')
  subprocess.Popen([PYTHON7_LOCATION,
    'language/language/orqa/evaluation/evaluate_predictions.py',
    '--references_path' , DEV_SET_PATH,
    '--predictions_path', path],
    stdout=open(path.strip('/') + '.acc', 'w'))
      
