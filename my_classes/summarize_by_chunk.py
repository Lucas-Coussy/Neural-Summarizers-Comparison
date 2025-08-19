from my_classes.my_model import MyModel, prompt

class by_chunk : 
    @staticmethod


    def chunk_text_by_sentence_phi(text, prompt, tokenizer, max_tokens): ###distribute prompt to all created chunks
        sentences = text.split(". ")
        chunks = []
        current_chunk = [prompt + "\n\n### Text: "]

        len_prompt = len(tokenizer.encode(current_chunk[0], add_special_tokens=False))
        len_end_prompt = len(tokenizer.encode(". \n\n### Summary: ", add_special_tokens=False))

        current_length = len_prompt
        
        first_sentence = True

        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_length + len(sentence_tokens) + len_end_prompt + (1 if current_chunk else 0) <= max_tokens - 20:  # Account for potential joining space and special tokens
                if first_sentence:
                    current_chunk[-1] += sentence #to avoid adding unwanted dot 
                    current_length += len(sentence_tokens) + (1 if current_chunk else 0)
                    first_sentence = False
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence_tokens) + (1 if current_chunk else 0) #(1 if current_chunk else 0) add one to lenght if list not empty
            else:
                if current_chunk:
                    current_chunk[-1] += ". \n\n### Summary: "
                    chunks.append(". ".join(current_chunk))
                current_chunk = [prompt + "\n\n### Text: " + sentence]
                current_length = len_prompt + len(sentence_tokens) + 1
        if current_chunk:
            current_chunk[-1] += ". \n\n### Summary: "
            chunks.append(". ".join(current_chunk))     #if the last chunks wasn't added to the list of chunks, it add it  
        return chunks
    
    def chunk_text_by_sentence_gpt2(text, prompt, tokenizer, max_tokens): ###distribute prompt to all created chunks
        sentences = text.split(". ")
        chunks = []
        current_chunk = [prompt]

        len_prompt = len(tokenizer.encode(current_chunk[0], add_special_tokens=False))

        current_length = len_prompt
        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_length + len(sentence_tokens) + (1 if current_chunk else 0) <= max_tokens - 20:  # Account for potential joining space and special tokens
                current_chunk.append(sentence)
                current_length += len(sentence_tokens) + (1 if current_chunk else 0) #(1 if current_chunk else 0) add one to lenght if list not empty
            else:
                if current_chunk:
                    chunks.append(". ".join(current_chunk))
                current_chunk = [prompt + sentence]
                current_length = len_prompt + len(sentence_tokens) + 1
        if current_chunk:
            chunks.append(". ".join(current_chunk))     #if the last chunks wasn't added to the list of chunks, it add it  
        return chunks
    
    def chunk_text_by_sentence_t5(text, prompt, tokenizer, max_tokens): ###distribute prompt to all created chunks
        sentences = text.split(". ")
        chunks = []
        current_chunk = [prompt]

        len_prompt = len(tokenizer.encode(current_chunk[0], add_special_tokens=False))
        len_end_prompt = 0

        current_length = len_prompt
        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_length + len(sentence_tokens) + len_end_prompt + (1 if current_chunk else 0) <= max_tokens - 20:  # Account for potential joining space and special tokens
                current_chunk.append(sentence)
                current_length += len(sentence_tokens) + (1 if current_chunk else 0) #(1 if current_chunk else 0) add one to lenght if list not empty
            else:
                if current_chunk:
                    chunks.append(". ".join(current_chunk))
                current_chunk = [prompt + sentence]
                current_length = len_prompt + len(sentence_tokens) + 1
        if current_chunk:
            chunks.append(". ".join(current_chunk))     #if the last chunks wasn't added to the list of chunks, it add it  
        return chunks
    
    def chunk_text_by_sentence_seq2seq(text, tokenizer, max_tokens):
        sentences = text.split(". ")  # Simple sentence splitting
        chunks = []
        
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_length + len(sentence_tokens) + (1 if current_chunk else 0) <= max_tokens - 20:  # Account for potential joining space and special tokens
                current_chunk.append(sentence)
                current_length += len(sentence_tokens) + (1 if current_chunk else 0) #(1 if current_chunk else 0) add one to lenght if list not empty
            else:
                if current_chunk:
                    chunks.append(". ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence_tokens) + 1
        if current_chunk:
            chunks.append(". ".join(current_chunk))     #if the last chunks wasn't added to the list of chunks, it add it  
        return chunks
    
    def summarize_by_chunk(text, prompt, model, tokenizer, max_input_tokens=512, max_output_tokens=300, detailed_summary = False,device="cpu"):
        if model.name_or_path == "gpt2-medium": #if model is gpt2
            chunks = by_chunk.chunk_text_by_sentence_gpt2(text, prompt, tokenizer, max_tokens=max_input_tokens)

            final_prompt = prompt + " #### "
            len_final_prompt = len(tokenizer.encode(final_prompt, add_special_tokens=False))
            output_length = min((max_input_tokens-len_final_prompt-5)//len(chunks),max_output_tokens)

            my_model = MyModel.gpt2_apply

        elif model.name_or_path == "microsoft/phi-3-mini-128k-instruct":
            chunks = by_chunk.chunk_text_by_sentence_phi(text, prompt, tokenizer, max_tokens=max_input_tokens)
            
            final_prompt = prompt + "\n\n### Text: " + " #### " + "\n\n### Summary:"
            output_length = max_output_tokens

            my_model = MyModel.phi_apply
             
        elif "t5" in model.name_or_path:
            chunks = by_chunk.chunk_text_by_sentence_t5(text, prompt, tokenizer, max_tokens=max_input_tokens)
           
            final_prompt = prompt + " #### " + " \n\n### Summary:"

            len_final_prompt = len(tokenizer.encode(final_prompt, add_special_tokens=False))
            output_length = min((max_input_tokens-len_final_prompt-5)//len(chunks),max_output_tokens)

            my_model = MyModel.t5_apply

        else:
            chunks = by_chunk.chunk_text_by_sentence_seq2seq(text, tokenizer, max_tokens=max_input_tokens)

            final_prompt = " #### "
            output_length = min((max_input_tokens-5)//len(chunks),max_output_tokens)

            my_model = MyModel.seq2seq

        print("number of chunks = ", len(chunks))
        print("max_input_tokens = ",max_input_tokens)
        print("number of token per chunks: ", output_length)
        print("")

         
        partial_summaries = []
        
        for chunk in chunks:
            print("output_length = ", output_length)
            summary = my_model(chunk, tokenizer, model, output_length=output_length, device=device) #we further limit the output length so that the summary of each 
            partial_summaries.append(summary)    
                                                                                                 #chunk can be concatenated and pass into the model without truncating
        combined_summary = " ".join(partial_summaries)
        if len(partial_summaries) == 1 or detailed_summary == True:
            return combined_summary
        else:
            final_summary = my_model(final_prompt.replace("####",combined_summary), tokenizer, model, output_length=max_output_tokens, device=device)
        return final_summary
        
    