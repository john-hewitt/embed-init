#!/bin/bash
#
# Experiment script training and evaluating a model on Natural Questions
#

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/u/nlp/anaconda/main/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/u/nlp/anaconda/main/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

if [[ ${HOSTNAME} == *"jagupard28"* ]]; then
  source activate johnhew-blink37
elif
 [[ ${HOSTNAME} == *"jagupard29"* ]]; then
  source activate johnhew-blink37
else
  source activate johnhew-kqv
fi

mkdir -p logs/
mkdir -p results/

model_name=
for seed in 1 2; do
for model_name in gpt2 gpt2-medium gpt2-large EleutherAI/gpt-neo-125M EleutherAI/gpt-neo-1.3B; do
for method in avg_emb zeros default; do
  safe_model_name=${model_name//\//-}
  echo "Using $method"
  python lm_example.py \
    --do_block_texts True \
    --tokenizer_name $model_name \
    --do_wikitext_type_addition True \
    --do_train True \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --overwrite_output_dir True \
    --do_eval True \
    --model_name $model_name \
    --cache_dir /u/scr/nlp/johnhew/data/huggingface \
    --evaluation_strategy steps \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --log_level debug \
    --logging_strategy steps \
    --logging_first_step True \
    --save_total_limit 1 \
    --eval_steps 20 \
    --seed $seed \
    --avg_emb_epsilon 0.00001 \
    --new_token_method $method \
    --output_dir results/ppl-$safe_model_name-$method/$seed \
    --do_plot_logits True &> logs/ppl-$method-$safe_model_name-seed$seed
done 
done
done
