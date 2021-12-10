### Rough experiments on new word embedding initialization

This rough set of scripts largely modifies an (old?) Hugginface script `lm_example.py`; modifications were written to support the blog post [Initializing New Word Embeddings for Pretrained Language Models](https://nlp.stanford.edu//~johnhew//vocab-expansion.html), in which I briefly study different methods for initializing new word embeddings in a pretrained LM.

To run the experiments. run `wikitext_scripts.sh`; this will run LM finetuning on wikipedia text for a few LMs across a few initialization strategies. 100 words that are common in wikitext but are currently split by the `gpt2` tokenizer are added to the vocabulary of each model in each experiment.
You can also modify the bash for loops in this file to run the experiments on different models or a subset of models.

I had a bunch of OOM errors on my GPUs, so the script `make_adapt_plots.py` makes the validation-PPL-over-time plots in the blog post for the models that didn't fail.
