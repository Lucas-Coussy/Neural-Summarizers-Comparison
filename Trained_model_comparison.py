from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import pandas as pd
import torch
from my_classes.summarize_by_chunk import by_chunk
from my_classes.my_model import prompt

import sys
sys.stdout.reconfigure(encoding='utf-8') #to avoid print error

device = "cuda" if torch.cuda.is_available() else "cpu"

causal_model = [] #"microsoft/phi-3-mini-128k-instruct" #Could not be fine-tuned due to insufficient resources.

seq2seq_model = ["allenai/led-base-16384","facebook/bart-large-cnn"] #"google/flan-t5-base",

model_name = causal_model + seq2seq_model

df_summary = pd.DataFrame(columns=['model','num_chp','summary'])


for model_ in model_name:
    torch.cuda.empty_cache()
    model_tag = f"trained_{model_.replace('/', '_')}"

    test_file_name = f"data/test_dataset_for_{model_.replace('/', '_')}.json"

    df_test = pd.read_json(test_file_name, orient='columns')
    test_dataset = Dataset.from_pandas(df_test)

    print(f"{model_} is being applied\n")
    if "gpt2" in model_ or "phi" in model_:
        tokenizer = AutoTokenizer.from_pretrained(model_tag, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_tag, trust_remote_code=True, 
                                                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_tag,local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_tag,local_files_only=True).to(device)

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
        df_summary.loc[len(df_summary)] = [model_tag,test_text['num_chp'],summary]
        print(df_summary)



df_summary.to_json('data/Trained_model_comparison', index=False)
