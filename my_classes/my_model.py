from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyModel:
    
    @staticmethod
    def phi_apply(example_text, tokenizer, model, output_length=400, device=device):
        max_input_tokens = 7000
        inputs = tokenizer(example_text, truncation=True, max_length=max_input_tokens, return_tensors="pt").to(model.device)
        input_id = inputs["input_ids"][0]

        print("")
        print("number token input",input_id.size(0))
        # Génération
        with torch.no_grad():
            output = model.generate(**inputs,
                                    max_new_tokens=output_length,
                                    temperature=0.7,
                                    top_p=0.95,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id
                                )
        print("number total token summary",len(output[0]))
        print("number new generated token",len(output[0][inputs["input_ids"].shape[1]:]))
        # Affichage du résultat
        #print("unkept token : ",  tokenizer.decode(output[0][:inputs["input_ids"].shape[1]], skip_special_tokens=True))
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]  # slice off prompt tokens
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return summary
    
    def gpt2_apply(example_text, tokenizer, model, output_length=400, device=device):
        max_input_tokens = 7000
        inputs = tokenizer(example_text, truncation=True, max_length=max_input_tokens, return_tensors="pt").to(model.device)
        input_id = inputs["input_ids"][0]

        print("")
        print("number token input",input_id.size(0))
        # Génération
        with torch.no_grad():
            output = model.generate(**inputs,
                                    max_new_tokens=output_length,
                                    temperature=0.7,
                                    top_p=0.95,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id
                                )
        print("number total token summary",len(output[0]))
        print("number new generated token",len(output[0][inputs["input_ids"].shape[1]:]))
        # Affichage du résultat
        #print("unkept token : ",  tokenizer.decode(output[0][:inputs["input_ids"].shape[1]], skip_special_tokens=True))
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]  # slice off prompt tokens
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return summary
    
    def t5_apply(example_text, tokenizer, model, output_length=400, device=device):
        inputs = tokenizer(example_text, return_tensors="pt").to(device)
        input_id = inputs["input_ids"][0]
        print("")
        print("number token input",input_id.size(0))
        # Génération
        output = model.generate(**inputs,
                                max_length=output_length,
                                min_length=output_length - 10,
                                num_beams=4,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=False
                            )
        print("number token summary",len(output[0]))
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
    
    def seq2seq(example_text, tokenizer, model, output_length=400, device=device):
        inputs = tokenizer(example_text, return_tensors="pt").to(device)
        input_id = inputs["input_ids"][0]
        print("")
        print("number token input",input_id.size(0))
        # Génération
        output = model.generate(**inputs,
                                max_length=output_length,
                                min_length=output_length - 10,
                                num_beams=4,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=False
                            )
        print("number token summary",len(output[0]))
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
    
class prompt: 

    gpt2_prompt ="""
Summarize the following text as a factual, chronological report.
Only include explicit events, in order. Omit unclear details."""


    phi_prompt = """Carefully read the following text and provide a detailed, accurate summary of all events mentioned.

- Only include information explicitly stated in the text.  
- Do NOT add, guess, interpret, or infer any details that are not clearly described.  
- Maintain the exact meaning of the source without rewording in a way that changes it.  
- All events must be presented strictly in the order they occur.  
- Avoid descriptive embellishments, figurative language, or speculation.    
- If something is unclear in the source, omit it rather than guess.  

Write the summary as a single continuous paragraph of text, not as bullet points or a list."""

    t5_prompt = "Summarize the following text in a single coherent paragraph without bullet points:\n\n"

    def get_prompt(model_name):
        if model_name == "gpt2-medium":
            return prompt.gpt2_prompt
        elif model_name == "microsoft/phi-3-mini-128k-instruct":
            return prompt.phi_prompt
        elif "t5" in model_name:
            return prompt.t5_prompt
        else:
            return ""

