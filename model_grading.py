import pandas as pd

#from evaluate import load
from rouge_score import rouge_scorer

from bert_score import BERTScorer

import sys
sys.stdout.reconfigure(encoding='utf-8') #to avoid print error

pd.set_option('display.max_columns', None)

df_test = pd.read_json('data/lotm_clean_dataset.json')
print(df_test.columns)

#df_predict = pd.read_json('data/model_comparison.json') #untrained models

df_predict = pd.read_json('data/Trained_model_comparison.json') #trained models

### df for comparing summaries with text
df_benchmark = pd.DataFrame()

df_benchmark['num_chp'] = df_predict['num_chp']
df_benchmark['model'] = df_predict['model']
df_benchmark['predicted_summary'] = df_predict['summary']

for index, row in df_test[df_test['num_chp'].isin(df_predict['num_chp'])].iterrows():
    df_benchmark.loc[len(df_benchmark)] = [row['num_chp'],"handmade_summary",row['summary']]

df_benchmark['original_text'] = df_benchmark.apply(lambda row: df_test.iloc[row.num_chp]["text"], axis=1)
print(df_benchmark)

original_text = df_benchmark["original_text"].tolist()
generated_summary = df_benchmark["predicted_summary"].tolist()

### df for comparing generated summaries with handmade summary
df_grading = pd.DataFrame()

df_grading['num_chp'] = df_predict['num_chp']
df_grading['model'] = df_predict['model']
df_grading['predicted_summary'] = df_predict['summary']
df_grading['expected_summary'] = df_grading.apply(lambda row: df_test.iloc[row.num_chp]["summary"], axis=1) 

scorer = BERTScorer(lang="en", rescale_with_baseline=False)

# Extract lists of predictions and references
predicted_summary = df_grading["predicted_summary"].tolist()
expected_summary = df_grading["expected_summary"].tolist()

P, R, F1 = scorer.score(predicted_summary, expected_summary)

# Add results to DataFrame
df_grading["bertscore_P"] = P.tolist()
df_grading["bertscore_R"] = R.tolist()
df_grading["bertscore_F1"] = F1.tolist()

P_b, R_b, F1_b = scorer.score(original_text, generated_summary)

# Add results to DataFrame
df_benchmark["bertscore_P"] = P_b.tolist()
df_benchmark["bertscore_R"] = R_b.tolist()
df_benchmark["bertscore_F1"] = F1_b.tolist()

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

# Apply row by row
df_grading[["rouge1", "rouge2", "rougeL"]] = df_grading.apply(
    compute_rouge, axis=1, result_type="expand"
)

def compute_rouge_for_benchmark(row):
    scores = scorer.score(
        row['original_text'],   # reference
        row['predicted_summary']   # candidate
    )
    return (
        scores["rouge1"].fmeasure,
        scores["rouge2"].fmeasure,
        scores["rougeL"].fmeasure,
    )

# Apply row by row
df_benchmark[["rouge1", "rouge2", "rougeL"]] = df_benchmark.apply(
    compute_rouge_for_benchmark, axis=1, result_type="expand"
)

df_global_score = df_grading.groupby(["model"]).agg(
    bertscore_P=('bertscore_P', 'mean'),
    bertscore_R=('bertscore_R', 'mean'),
    bertscore_F1=('bertscore_F1', 'mean'),
    rouge1=('rouge1', 'mean'),
    rouge2=('rouge2', 'mean'),
    rougeL=('rougeL', 'mean')
)
print("score : summary vs handmade summary")
print(df_global_score)
print("")

df_benchmark_score = df_benchmark.groupby(["model"]).agg(
    bertscore_P=('bertscore_P', 'mean'),
    bertscore_R=('bertscore_R', 'mean'),
    bertscore_F1=('bertscore_F1', 'mean'),
    rouge1=('rouge1', 'mean'),
    rouge2=('rouge2', 'mean'),
    rougeL=('rougeL', 'mean')
)
print("score : summary vs text")
print(df_benchmark_score)
print("")

#take a look at the summaries of the first text
test_chp = min(df_predict['num_chp'])

my_df = df_predict[df_predict['num_chp']==test_chp]

print("expected summary : ")
print(df_test[df_test["num_chp"] == test_chp]["summary"].tolist()[0])
print("")

for model_ in my_df['model']:
    print(f"--{model_} summary--")
    print(my_df.loc[my_df['model']== model_, 'summary'].iloc[0])
    print("")
