from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import pandas as pd
import torch
from my_classes.summarize_by_chunk import by_chunk
from my_classes.my_model import prompt

import sys
sys.stdout.reconfigure(encoding='utf-8') #to avoid print error

import os  
os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes")  

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_json('data/model_train.json')
dataset = Dataset.from_pandas(df)
print(df)
dataset = dataset.remove_columns(['__index_level_0__'])

dataset = dataset.shuffle(seed=42).select(range(150)) #reduce size to reduce time

#dataset = dataset.select([0,1,2,3,4]) #for testing code
print(dataset)

causal_model = ["microsoft/phi-3-mini-128k-instruct"] 

seq2seq_model = ["facebook/bart-large-cnn", "allenai/led-base-16384"] #"google/flan-t5-base"

model_name = causal_model + seq2seq_model

for model_ in model_name:
    torch.cuda.empty_cache()
    print(f"{model_} is being applied\n")
    df_summary_for_training = pd.DataFrame(columns=['num_chp','summary'])

    if "gpt2" in model_ or "phi" in model_:
        tokenizer = AutoTokenizer.from_pretrained(model_, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_, trust_remote_code=True, 
                                                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_).to(device) #,local_files_only=True

    model_max_length = tokenizer.model_max_length #determine model max length parameter
    if model_max_length < 1e9:
        max_token = model_max_length
    else:
        print(f"no specified input limit ({model_max_length})\n")
        max_token = 1024 #choice based on my selection of model, if a model, with an input of less than 1024 token, had unspecified max_length it would return an error

    max_output_token = 300 # the model will split the text into chunks and summarize each of them as to have partial summaries of length inferior to 300 tokens
                           # for causal model (of input size at least equal to 1024) it makes certain that it leaves enough space for context

    if model_ in causal_model: #determine max_input_token based on the structure of the model
        max_input_tokens = max_token-max_output_token
    else:
        max_input_tokens = max_token

    for test_text in dataset: #summary all test_text and add them to the df
        text = test_text['text']
        print("max_token = ",max_input_tokens)
        my_prompt = prompt.get_prompt(model_)
        summary = by_chunk.summarize_by_chunk(text, my_prompt, model, tokenizer, max_input_tokens=max_input_tokens, max_output_tokens=max_output_token, for_training=True, device=device) 
        #for_training=True joined summaries of chunk and return it, the number of tokens is close (and inferior) to the input size of the model

        print("len_summary : ",len(tokenizer.encode(summary, add_special_tokens=False)))
        df_summary_for_training.loc[len(df_summary_for_training)] = [test_text['num_chp'],summary]
        print(df_summary_for_training)
    
    model_tag = model_.replace("/", "_")
    df_summary_for_training.to_json(f"data/summary_for_training_{model_tag}.json", orient='records',lines=True, index=False)
