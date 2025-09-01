from datasets import load_dataset, DatasetDict, Dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import pandas as pd
import torch
from my_classes.summarize_by_chunk import by_chunk
from my_classes.my_model import prompt

import sys
sys.stdout.reconfigure(encoding='utf-8') #to avoid print error

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_json('data/lotm_clean_dataset.json')
dataset = Dataset.from_pandas(df)
dataset = dataset.remove_columns('__index_level_0__')

split = dataset.train_test_split(test_size=0.006, seed=42) 

#print(split)
train_dataset = split['train']
test_dataset = split['test'] # it's just to get a global idea of which model is more likely to perform well, so a small sample is enough (around ten chapter here)

df_train = train_dataset.to_pandas()

df_train.to_json('data/model_train', index=False)

print(test_dataset)
print("")

causal_model = ["gpt2-medium","microsoft/phi-3-mini-128k-instruct"] 

seq2seq_model = ["google/flan-t5-base","facebook/bart-large-cnn","google/pegasus-xsum","t5-large","allenai/led-base-16384"] 

model_name = causal_model + seq2seq_model

df_summary = pd.DataFrame(columns=['model','num_chp','summary'])


for model_ in model_name:
    torch.cuda.empty_cache()
    print(f"{model_} is being applied\n")
    if "gpt2" in model_ or "phi" in model_:
        tokenizer = AutoTokenizer.from_pretrained(model_, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_, trust_remote_code=True, 
                                                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_,local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_,local_files_only=True).to(device)

    model_max_length = tokenizer.model_max_length #determine model max length parameter
    if model_max_length < 1e9:
        max_token = model_max_length
    else:
        print(f"no specified input limit ({model_max_length})\n")
        max_token = 1024 #choice based on my selection of model, if a model, with an input of less than 1024 token, had unspecified max_length it would return an error
    max_output_length = 300

    if model_ in causal_model: #determine max_input_token based on the structure of the model
        max_input_tokens = max_token-max_output_length
    else:
        max_input_tokens = max_token

    for test_text in test_dataset: #summary all test_text and add them to the df
        text = test_text['text']
        #print(text)
        print("max_token = ",max_input_tokens)
        my_prompt = prompt.get_prompt(model_)
        summary = by_chunk.summarize_by_chunk(text, my_prompt, model, tokenizer, max_input_tokens=max_input_tokens, max_output_tokens=max_output_length, device=device)
        #print(summary)
        df_summary.loc[len(df_summary)] = [model_,test_text['num_chp'],summary]
        

print(df_summary)

df_summary.to_json('data/model_comparison', index=False)
