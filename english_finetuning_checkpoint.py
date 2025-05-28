import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
from deep_translator import (GoogleTranslator, batch_detection)

def chunk_list(lst, chunk_size):
    """Yield successive chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

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

chunk_size = 100

sentence1_chunks = chunk_list(english_df_proc['sentence1'].tolist(), chunk_size)
sentence2_chunks = chunk_list(english_df_proc['sentence2'].tolist(), chunk_size)

translated_sent1_chunks = []
translated_sent2_chunks = []

for i, chunk in enumerate(sentence1_chunks):
    if i < 6:
        continue
    try:
        translated = translator.translate_batch(chunk)
        translated_sent1_chunks.extend(translated)
        print(f"✅ Translated chunk {i + 1}")
        df_chunk = pd.DataFrame(translated, columns=['translated_sentence'])
        df_chunk.to_csv(f"translated_chunk_{i + 1}.csv", index=False)
        time.sleep(5)  # add a short delay (1-2 sec) to be polite to the server
    except Exception as e:
        print(f"❌ Failed on chunk {i + 1}: {e}")
        translated_sent1_chunks.extend(["[ERROR]"] * len(chunk))

for i, chunk in enumerate(sentence2_chunks):
    if i < 6:
        continue
    try:
        translated = translator.translate_batch(chunk)
        translated_sent2_chunks.extend(translated)
        print(f"✅ Translated chunk {i + 1}")
        df_chunk = pd.DataFrame(translated, columns=['translated_sentence'])
        df_chunk.to_csv(f"translated_chunk_{i + 1}.csv", index=False)
        time.sleep(5)  # add a short delay (1-2 sec) to be polite to the server
    except Exception as e:
        print(f"❌ Failed on chunk {i + 1}: {e}")
        translated_sent2_chunks.extend(["[ERROR]"] * len(chunk))

translated_sent1 = [item for sublist in sentence1_chunks for item in sublist]
translated_sent2 = [item for sublist in sentence2_chunks for item in sublist]

english_df['sentence1'] = translated_sent1
english_df['sentence2'] = translated_sent2

english_df.to_csv("translated_afrikaans.csv", index=False)

# df_combined = pd.concat([english_df_proc, afr_df], ignore_index=True)

# df_combined.to_csv("combined_dataset.csv", index=False)
