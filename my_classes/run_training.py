from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1" #skip importing tensorflow to avoid conflicting versions

from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
)
import torch

import sys
sys.stdout.reconfigure(encoding='utf-8') #to avoid print error

class train:
    @staticmethod
    def run_training(model_name: str, save_dir: str, dataset, max_input_len: int, device):
        print(f"\n--- Fine-tuning {model_name} ---")

        # Load tokenizer & model
        if "gpt2" in model_name or "phi" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
                                                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name,local_files_only=True).to(device)
        
        # Tokenization function
        def tokenize_fn(batch):
            inputs = batch["text"]
            summaries = batch["summary_target"]
            #print("input : ", inputs)
            #print("summaries : ", summaries)

            model_inputs = tokenizer(
                inputs,
                padding="longest",
                truncation=True,
                max_length=max_input_len,
            )
            #print("model_inputs step 1: ",model_inputs)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    summaries,
                    padding="longest",
                    truncation=True,
                    max_length=500,
                )

            model_inputs["labels"] = labels["input_ids"]
            #print("model_inputs step 2: ",model_inputs)
            return model_inputs

        # Tokenize dataset
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "summary_target"])
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

        # Training args
        training_args = TrainingArguments(
            output_dir=save_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,
            num_train_epochs=3,
            fp16=True,
            save_total_limit=2,
            save_steps=5000,
            logging_steps=200,
            eval_steps=1000,
            evaluation_strategy="steps",   # ✅ add this
            save_strategy="steps",         # ✅ make explicit
            load_best_model_at_end=True,   # Load best checkpoint after training
            metric_for_best_model="eval_loss",  
            greater_is_better=False,
            report_to="none",
        )
        #print("train : ",tokenized["train"])
        #print("validation : ",tokenized["validation"])

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized ["train"],
            eval_dataset=tokenized ["validation"],  # Add validation here
            data_collator=data_collator,
            tokenizer=tokenizer,  # For automatic padding during eval
        )

        trainer.train()
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"✅ Model saved to {save_dir}\n")