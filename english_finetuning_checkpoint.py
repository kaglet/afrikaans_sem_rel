from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
from deep_translator import GoogleTranslator

english_df = pd.read_csv('sem_text_rel_ranked.csv')
afr_df_dev = pd.read_parquet("dev.parquet")
afr_df_test = pd.read_parquet("test.parquet")
afr_df = pd.concat([afr_df_dev, afr_df_test], ignore_index=True)

afr_df_dev.to_csv("afr_dev.csv", index=False)
afr_df_test.to_csv("afr_test.csv", index=False)

english_df[['sentence1', 'sentence2']] = english_df['Text'].str.split('\n', n=1, expand=True)

english_df['label'] = english_df['Score']

english_df_proc = english_df[['sentence1', 'sentence2', 'label']]

translator = GoogleTranslator(source='en', target='af')
english_df['sentence1'] = translator.translate_batch(english_df['sentence1'].tolist())
english_df['sentence2'] = translator.translate_batch(english_df['sentence2'].tolist())

english_df.to_csv("translated_afrikaans.csv", index=False)

df_combined = pd.concat([english_df_proc, afr_df], ignore_index=True)

df_combined.to_csv("combined_dataset.csv", index=False)
