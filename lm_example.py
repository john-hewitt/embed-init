#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Modifications by John Hewitt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
import shutil
from datasets import load_dataset
import greedy_predictor
from tqdm import tqdm

def nop(it, *a, **k):
  return it

tqdm.tqdm = nop

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from plot import plot_logits

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.10.0.dev0")

#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    do_wikitext_type_addition: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add a bunch of types from wikitext"})

    new_type_string: Optional[str] = field(
        default='[NEW]', metadata={"help": "The string form of the new type to be added"})

    do_block_texts: Optional[bool] = field(
        default=True, metadata={"help": "Whether to concatenate all examples and regroup"})

    do_nq_train: Optional[bool] = field(
        default=False, metadata={"help": "Whether to load and tokenize examples as QA"})

    do_nq_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to decode predictions and run the efficientQA scorer"})
    
    do_plot_logits: Optional[bool] = field(
        default=False, metadata={"help": "Plot the logits of the model"})

    new_token_method: Optional[str] = field(
        default='default', metadata={"help": "Method to initialize the new token "})

    avg_emb_epsilon: Optional[float] = field(
        default=0.0001, metadata={"help": "Epsilon in covariance matrix for avgemb"})

    num_beams: Optional[int] = field(
        default=1, metadata={"help": "Number of beams in beam search decoding"})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "jsonl", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "jsonl", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        #if 'test' in raw_datasets.keys(): #WQ -- using as a diagnostic
        #  raw_datasets['validation'] = raw_datasets['test']

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
        if extension == "jsonl":
            extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if data_args.new_token_method != 'no_new_token':
      if data_args.new_type_string != '[NEW]':
        greedy_predictor.NEW_TOKEN = data_args.new_type_string 
      if not data_args.do_wikitext_type_addition:
        tokenizer.add_tokens([greedy_predictor.NEW_TOKEN])
        logger.info('Added new token: {}'.format(greedy_predictor.NEW_TOKEN))
      else:
        tokenizer.add_tokens(greedy_predictor.wikitext_newtokens_list)
        logger.info('Added new tokens: {}'.format(greedy_predictor.wikitext_newtokens_list))
    else:
      logger.info('Not adding new token')

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Do the token replacement
    params = model.state_dict()
    #print(params.keys())
    param_name = 'transformer.wte.weight' if 'transformer.wte.weight' in params else 'bert.embeddings.word_embeddings.weight'
    if data_args.new_token_method in {'avg_emb_add'}:
      num_inits = 1 if not data_args.do_wikitext_type_addition else 100
      original_embs = params[param_name][:-num_inits,:]
      mean = torch.mean(original_embs, dim=0)
      samples = params[param_name][-num_inits:,:]
      sample_norms = torch.norm(samples, dim=1)

      #params[param_name][-num_inits:,:] = (samples * torch.norm(mean) / sample_mean / torch.sqrt(torch.tensor(float(mean.size()[0])))) + mean
      params[param_name][-num_inits:,:] = (samples.t() / sample_norms).t() * torch.norm(mean) * 1e-8 + mean

      #params[param_name][-num_inits:,:] = (params[param_name][-num_inits:,:] * torch.norm(mean) / torch.mean(params[param_name][-num_inits:,:], dim=1) / torch.sqrt(params.size()[1])) + mean
      #params[param_name][-num_inits:,:] = params[param_name][-num_inits:,:] * torch.norm(mean) + mean #/ torch.mean(params[param_name][-num_inits:,:], dim=1) / torch.sqrt(params.size()[1])) + mean
      model.load_state_dict(params)
    elif data_args.new_token_method in {'avg_emb', 'norm_avg_emb', 'renorm_avg_emb', 'both_avg_emb'}:
      num_inits = 1 if not data_args.do_wikitext_type_addition else 100
      original_embs = params[param_name][:-num_inits,:]
      mean = torch.mean(original_embs, dim=0)
      diff = (original_embs - mean)
      cov = (diff.T @ diff) / tokenizer.vocab_size
      dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
      for i in range(num_inits):
        logger.info("Performing AvgEmb")
        sample = dist.sample()
        if data_args.new_token_method == 'norm_avg_emb':
          sample = sample * int(params[param_name][-1-i,:].size()[0])/2000
        if data_args.new_token_method == 'renorm_avg_emb':
          sample = sample * torch.norm(params[param_name][-1,:]) / torch.norm(sample)
        if data_args.new_token_method == 'both_avg_emb':
          sample = sample + params[param_name][-1-i,:] #original_embs[-1-i,:]
        params[param_name][-1-i,:] = sample
      model.load_state_dict(params)
    elif data_args.new_token_method == 'endoftext':
      logger.info("Performing endoftext")
      params = model.state_dict()
      endoftext_index = tokenizer.encode('<|endoftext|>')[0]
      params[param_name][-1,:] = params[param_name][endoftext_index,:]
      model.load_state_dict(params)
    elif data_args.new_token_method == 'default':
      logger.info("Performing Default init")
    elif data_args.new_token_method == 'zeros':
      params = model.state_dict()
      num_inits = 1 if not data_args.do_wikitext_type_addition else 100
      if data_args.do_wikitext_type_addition:
        params[param_name][-1,:] = torch.zeros_like(params[param_name][-1,:])
      else:
        params[param_name][-100:,:] = torch.zeros_like(params[param_name][-100:,:])
      model.load_state_dict(params)
    elif data_args.new_token_method == 'no_new_token':
      greedy_predictor.NEW_TOKEN = '<|endoftext|>'
      pass
    else:
      raise ValueError("unknown init type: {}".format(data_args.new_token_method))




    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            if data_args.do_nq_train:

              answer_string = 'answer' if 'answer' in examples else 'answers'
              examples['question'] = [x.replace('\n', ' ') for x in examples['question']]
              examples[answer_string] = [[y.replace('\n', ' ') for y in x] for x in examples[answer_string]]
              #print(examples['question'])

              example_texts = [('' if examples['question'][i][0] == ' ' else ' ')
                  + examples['question'][i]
                  + '' # ('' if examples['question'][i][-1] == ' ' else ' ')
                  + greedy_predictor.NEW_TOKEN
                  + ('' if examples[answer_string][i][0][0] == ' ' else ' ')
                  + examples[answer_string][i][0]
                  + '' # ('' if examples[answer_string][i][0][-1] == ' ' else ' ')
                  + greedy_predictor.NEW_TOKEN
                   for i in range(len(examples['question']))]
              #print(example_texts)

              SEP_INDEX = tokenizer(greedy_predictor.NEW_TOKEN)['input_ids'][0]
              tokenized_examples = [tokenizer(x)['input_ids'] for x in example_texts]
              logger.warning(str(tokenizer.convert_ids_to_tokens(tokenized_examples[0])))
              labels = [([-100 for i in range(x.index(SEP_INDEX)+1)] + x[x.index(SEP_INDEX)+1:]) for x in tokenized_examples]
              output = {
                  'input_ids': tokenized_examples,
                  'labels': labels
                  }

              #question = tokenizer(examples['question'])
              #print([x[0] for x in examples[answer_string]])
              #answer = tokenizer([x[0] for x in examples[answer_string]])
              #sep_index = [len(question['input_ids'][i]) for i in range(len(question['input_ids']))]
              #output = {
              #    #'input_ids' : [question['input_ids'][i] + tokenizer(NEW_TOKEN)['input_ids'] + answer['input_ids'][i] + tokenizer(NEW_TOKEN)['input_ids'] for i in range(len(question['input_ids']))],
              #    #'labels': [[-100 for x in question['input_ids'][i]] + [-100] + answer['input_ids'][i] + tokenizer(NEW_TOKEN)['input_ids'] for i in range(len(question['input_ids']))],
              #    'input_ids' : [question['input_ids'][i] + tokenizer('<|endoftext|>')['input_ids'] + answer['input_ids'][i] + tokenizer('<|endoftext|>')['input_ids'] for i in range(len(question['input_ids']))],
              #    'labels': [[-100 for x in question['input_ids'][i]] + [-100] + answer['input_ids'][i] + tokenizer('<|endoftext|>')['input_ids'] for i in range(len(question['input_ids']))],
              #    }
              #print(tokenizer.convert_ids_to_tokens(output['input_ids'][0]))
              #print(output['labels'][0])
              #if not data_args.do_block_texts:
              #  max_len = max((len(x) for x in output['input_ids']))
              #  output = {
              #      'input_ids': [output['input_ids'][i] + [0]*(max_len - len(output['input_ids'][i])) for i in range(len(output))],
              #      'labels': [output['labels'][i] + [-100]*(max_len - len(output['labels'][i])) for i in range(len(output))],
              #      }
                #print([len(output['input_ids'][x]) for x in range(len(output['input_ids']))])
                #print(output)
            else:
              output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            #batch_size=training_args.per_device_train_batch_size if data_args.do_nq_train else 1000,
            batch_size=training_args.per_device_train_batch_size if data_args.do_nq_train else 2,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )



    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if not data_args.do_nq_train:
          result["labels"] = result["input_ids"].copy()

        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    if data_args.do_block_texts:
      with training_args.main_process_first(desc="grouping texts together"):
          lm_datasets = tokenized_datasets.map(
              group_texts,
              batched=True,
              batch_size=2,
              num_proc=data_args.preprocessing_num_workers,
              load_from_cache_file=not data_args.overwrite_cache,
              desc=f"Grouping texts in chunks of {block_size}",
          )
    else:
      lm_datasets = tokenized_datasets
    #else:
    #  lm_datasets = tokenized_datasets.map(
    #      lambda examples: {k: [torch.tensor(x) for x in examples[k]] for k in examples}
    #      )

    # Possibly sample and plot some logits
    if data_args.do_plot_logits:
      print('Plotting logits')
      logit_list = []
      for _, data in tqdm(zip(range(10), lm_datasets['train'])):
        data = {k:torch.tensor([data[k]]) for k in data}
        ##data = tokenizer(data['input_ids'], return_tensors='pt')
        logits = model(input_ids=data['input_ids']).logits
        logit_list.append(logits.clone().detach().cpu())
      plot_logits(torch.cat(logit_list, dim=1), model_args.model_name_or_path)
      if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
      shutil.copy('logits.png', os.path.join(training_args.output_dir, 'logits.png'))


    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    def pad_collator(examples):
      max_len = max((len(x['input_ids']) for x in examples))
      output = {}
      for k in examples[0]:
        if k == 'labels':
          output[k] = torch.tensor([examples[i][k] + [-100]*(max_len - len(examples[i][k])) for i in range(len(examples))])
        else:
          output[k] = torch.tensor([examples[i][k] + [0]*(max_len - len(examples[i][k])) for i in range(len(examples))])
      return output

    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=pad_collator if data_args.do_nq_train else default_data_collator,
        #data_collator=default_data_collator,
    )

    def write_some_samples():
      samples_path = os.path.join(training_args.output_dir, 'samples.txt')
      with open(samples_path, 'w') as fout:
        for i in range(5):
          sample_ids = model.generate(return_dict_in_generate=True, output_scores=True, max_new_tokens=50, do_sample=True).sequences[0]
          fout.write(tokenizer.decode(sample_ids)+'\n')
    write_some_samples()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    if data_args.do_nq_eval:
      logger.info("*** NQ Evaluate ***")
      dataset_id = 'wq' if (data_args.dataset_name == 'web_questions'
          or 'wq' in data_args.train_file) else 'nq'
      greedy_predictor.compute_nq_metrics(
          model,
          raw_datasets['validation'],
          tokenizer,
          os.path.join(training_args.output_dir, 'nq-preds-beams{}.txt'.format(data_args.num_beams)),
          dataset_id,
          num_beams = data_args.num_beams
          )



    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

