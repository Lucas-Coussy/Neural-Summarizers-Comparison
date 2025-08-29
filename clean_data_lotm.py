import pandas as pd
import ast

import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_json('data/lotm_dataset.json')
pd.set_option('display.max_rows', None)

shape = df.shape

#remove the editor on the first row (other pages' editor are treated in the scraping code)
to_remove = "Editor:AtlasStudios\n"
df["text"].iloc[0][0] = df["text"].iloc[0][0].replace(to_remove,"")

## clean dataframe from unsupported symbol and ads
for num_chp in range(shape[0]):
    df["summary"].iloc[num_chp][1] = df["summary"].iloc[num_chp][1].replace('“', '"').replace('”', '"').replace("’","'").replace("—","-").replace('…', '...')

    if (num_chp + 1) % 15 == 0:
        text_to_remove = """About AllSmilesOn Hello Web novels, Manhwa and Dramas are my source of Smiles, so you will be seeing a lot of posts about them. View all posts by AllSmilesOn Comment Save my name, email, and website in this browser for the next time I comment."""
        df["summary"].iloc[num_chp][1] = df["summary"].iloc[num_chp][1].replace(text_to_remove,"")
    if num_chp % 15 == 0:
        text_to_remove = """Unlock Global Stories with Noyaku! Tired of waiting? Instantly translate web novels, manhwa, and manhua from their original sources! Noyaku provides Smart AI and Basic translation engines, plus a custom Glossary for consistent terms."""
        df["summary"].iloc[num_chp][1] = df["summary"].iloc[num_chp][1].replace(text_to_remove,"")
    
    df["summary"].iloc[num_chp] = df["summary"].iloc[num_chp][1]
    df["text"].iloc[num_chp] = df["text"].iloc[num_chp][0]
    df["num_chp"] = df.index + 1 

df = df.iloc[:, [2, 0, 1]]
#print(df)
df.to_json("data/lotm_clean_dataset", index=False)
