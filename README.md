# Neural-Summarizers-Comparison

The objective of this project was to get familiar with **neural networks for language analysis** and to compare the performance of several models on a **summarization task**.  

The comparison was done in two stages:  
1. **Without fine-tuning** – testing how well pre-trained models perform out of the box.  
2. **With fine-tuning** – evaluating improvements after adapting the models specifically to the task.

## The Data

To test the summarization abilities of different models, I chose to work with chapters from the novel *Lord of the Mysteries*.  

- I first **scraped the chapters** from [novelfull](https://novelfull.net/lord-of-the-mysteries.html) and **extracted the summaries** from [dragneelclub](https://dragneelclub.com/category/chapters/lord-of-the-mysteries/).  
  Both were then exported as a JSON file named **`lotm_dataset.json`** (see **`scrap_chapter.py`** for the code).  

- Next, the dataset was **cleaned**:  
  - Translator and editor notes were removed from the chapters.  
  - Advertisements were removed from the summaries.  

  The cleaned version was exported as **`lotm_clean_dataset.json`** (see **`clean_data_lotm.py`**).

## Applying the models to our chapters

- I started by importing and splitting my cleaned dataset (**`clean_data_lotm.py`**).  
  To speed up testing, I kept about **10 texts** for applying the models and getting a rough idea of which one would be the most effective for summarization.  

- The first run was done with **70 texts** (see **`model_comparison_previous_version.json`**) and gave similar results.  
  Later, when I added a seed to the split, I reduced the set to 10 to save time. Since the results were consistent, I kept this smaller set.  
  The rest of the dataset was exported as a JSON file (**`model_train.json`**).  

- For texts that exceeded the input size of the models, I looked up different strategies such as :  
  - Using extractive models to shrink the text to fit the input size.  
  - Splitting the text into chunks matching the input size, summarizing each chunk, concatenating the partial summaries, and then summarizing the result.  
  I chose the **chunk-based approach** (see **`summarize_by_chunk.py`** in the **`my_classes`** folder).  

- Finally, I created a DataFrame with three columns:  
  - **Model name**  
  - **Chapter number**  
  - **Generated summary**  

  The results were then stored in the JSON file **`model_comparison.json`**.  

## Grading the generated summaries

Using the previous results, two main DataFrames were created:

- **`df_benchmark`** that contains ROUGE and BERTScore results comparing the generated summaries with the **original text**.  
- **`df_grading`** that contains ROUGE and BERTScore results comparing the generated summaries with the **human-made summaries**.  

To get an overall view of model performance, the mean scores were computed and stored in two additional DataFrames:

- **`df_global_score`** that average scores from `df_grading`.  
- **`df_benchmark_score`** that average scores from `df_benchmark`.

Next, the generated summaries from the first text in the dataset are displayed for comparison across the different models.  

## Training the model

Next, I trained the **three best-performing models** (**phi-3-mini-128k-instruct**,**bart-large-cnn** and **led-base-16384**) using my dataset (`model_train.json`):  

- The texts were **split into chunks** and summarized in the same way as when the models were applied.  
- The **concatenated partial summaries** (shorter than the model's input size) were exported to files named:  
  `summary_for_training_{model_name}`.  
- These concatenated summaries were then used as **input** to fine-tune the models, with the **human-made summaries** serving as the target (for the code, see the file `Train_model.py`).

## Applying the trained model

Here, I slightly adjusted the code from **`model_comparison.py`** (see `Trained_model_comparison.py`) so that it could apply the **fine-tuned models**, just like it did with the pre-trained ones.  

- The process is the same, and it exports a file similar to **`model_comparison.json`** but for the fine-tuned models, it is named:  
  **`Trained_model_comparison.json`**  

At the beginning of the for loop, the **`model_tag`** can be set depending on whether you want to run a **locally saved trained model** obtained from running the file `Train_model.py`, or the **fine-tuned model hosted on Hugging Face** trained for this project.

```python
for model_ in model_name:
    torch.cuda.empty_cache()
    # Use this option to load the locally saved model (from `Train_model.py`)
    # model_tag = f"trained_{model_.replace('/', '_')}"

    # Use this option to load the model uploaded on Hugging Face
    model_tag = f"Lambda-ck/{model_.split('/')[1]}-lotm-fine-tuned"
```

## Grading the trained models

The fine-tuned models are compared in the **same way** as the pre-trained ones.  

- The results are presented in the notebook **`Results_presentation.ipynb`**.  
- To switch between pre-trained and fine-tuned model results, simply change the data source:  

```python
# For pre-trained models
df_predict = pd.read_json('data/model_comparison.json')

# For fine-tuned models
df_predict = pd.read_json('data/Trained_model_comparison.json')
```