from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1" #skip importing tensorflow cause unnecessary

from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoConfig,
)
import torch
from my_classes.run_training import train

import os  
os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes")


causal_model = ["microsoft/phi-3-mini-128k-instruct"] #"microsoft/phi-3-mini-128k-instruct"

seq2seq_model = ["facebook/bart-large-cnn","allenai/led-base-16384"] #"google/flan-t5-base"

model_name = causal_model + seq2seq_model

df_text = pd.read_json("data/lotm_clean_dataset.json")

for model_ in model_name:
    torch.cuda.empty_cache()
    
    file_name = f"data/summary_for_training_{model_.replace('/', '_')}.json"
    df = pd.read_json(file_name, lines=True) #import train_dataset from data file

    mask = df_text['num_chp'].isin(df['num_chp'])
    column_to_add = df_text.loc[mask, ["num_chp", "summary"]]

    df = df.merge(column_to_add, on='num_chp')
    df.rename(columns={"summary_x":"text","summary_y":"summary_target"}, inplace=True)
    print(df) 
    
    dataset = Dataset.from_pandas(df)
    split_datasets = dataset.train_test_split(test_size=0.2, seed=42) 

    dataset = DatasetDict({
        "train": split_datasets["train"],
        "validation": split_datasets["test"],  # test here is actually your validation split
    })

    test_dataset = dataset["validation"].to_pandas()
    test_file_name = f"data/test_dataset_for_{model_.replace('/', '_')}"
    test_dataset.to_json(test_file_name, index=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### find model's max input_length
    if "t5" in model_:
        tokenizer = AutoTokenizer.from_pretrained(model_, trust_remote_code=True) #t5 models config don't have max_position_embeddings attribute
        model_max_length = tokenizer.model_max_length
    elif "led" in model_:
        config = AutoConfig.from_pretrained(model_, trust_remote_code=True)
        model_max_length = config.max_encoder_position_embeddings 
    else:
        config = AutoConfig.from_pretrained(model_, trust_remote_code=True)
        model_max_length = config.max_position_embeddings
    
    if model_max_length < 1e9:
        max_input_len = model_max_length - 20
    else:
        print(f"no specified input limit ({model_max_length})\n")
        max_input_len = 512 - 20 #choice based on my selection of model, if a model, with an input of less than 512 token, had unspecified max_length it would return an error

    save_dir = f"trained_{model_.replace('/', '_')}"

    train.run_training(model_, save_dir=save_dir, dataset=dataset, max_input_len=max_input_len, device=device)
