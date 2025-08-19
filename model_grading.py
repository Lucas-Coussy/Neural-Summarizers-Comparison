import pandas as pd

from datasets import load_dataset, DatasetDict, Dataset

#from evaluate import load
from rouge_score import rouge_scorer

from bert_score import BERTScorer

from sacrebleu.metrics import BLEU
bleu_scorer = BLEU()

import sys
sys.stdout.reconfigure(encoding='utf-8') #to avoid print error

pd.set_option('display.max_columns', None)

df_test = pd.read_json('data/lotm_clean_dataset.json')

df_predict = pd.read_json('data/model_test.json')

df_grading = pd.DataFrame()

df_grading['num_chp'] = df_predict['num_chp']
df_grading['model'] = df_predict['model']
df_grading['predicted_summary'] = df_predict['summary']
df_grading['expected_summary'] = df_grading.apply(lambda row: df_test.iloc[row.num_chp]["summary"], axis=1) 

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# Extract lists of predictions and original summary
predicted_summary = df_grading["predicted_summary"].tolist()
expected_summary = df_grading["expected_summary"].tolist()

P, R, F1 = scorer.score(predicted_summary, expected_summary)

# Add results to df
df_grading["bertscore_P"] = P.tolist()
df_grading["bertscore_R"] = R.tolist()
df_grading["bertscore_F1"] = F1.tolist()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_rouge(row):
    scores = scorer.score(
        row["expected_summary"],   # reference
        row["predicted_summary"]   # candidate
    )
    return (
        scores["rouge1"].fmeasure,
        scores["rouge2"].fmeasure,
        scores["rougeL"].fmeasure,
    )

# apply row by row
df_grading[["rouge1", "rouge2", "rougeL"]] = df_grading.apply(
    compute_rouge, axis=1, result_type="expand"
)

df_global_score = df_grading.groupby(["model"]).agg(
    bertscore_P=('bertscore_P', 'mean'),
    bertscore_R=('bertscore_R', 'mean'),
    bertscore_F1=('bertscore_F1', 'mean'),
    rouge1=('rouge1', 'mean'),
    rouge2=('rouge2', 'mean'),
    rougeL=('rougeL', 'mean')
)
print(df_global_score)
print("")

#take a look at the summaries of the first text
test_chp = min(df_predict['num_chp'])

my_df = df_predict[df_predict['num_chp']==test_chp]

for model_ in my_df['model']:
    print(f"--{model_} summary--")
    print(my_df.loc[my_df['model']== model_, 'summary'].iloc[0])
    print("")