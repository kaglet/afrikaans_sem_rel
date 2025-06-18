import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
import os
from deep_translator import (GoogleTranslator, batch_detection)

def chunk_list(lst, chunk_size):
    """Yield successive chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# Get english sem rel data
english_df = pd.read_csv('sem_text_rel_ranked.csv', encoding='utf-8')

# Get afrikaans sem rel data (combined)
if not os.path.exists("afr.csv"):
    afr_df_dev = pd.read_parquet("dev.parquet")
    afr_df_test = pd.read_parquet("test.parquet")
    afr_df = pd.concat([afr_df_dev, afr_df_test], ignore_index=True)
    afr_df.to_csv("afr.csv", index=False, encoding='utf-8')
else: 
    afr_df = pd.read_csv('afr.csv', encoding='utf-8')

# Place english data in new format (same as afr version)
if not os.path.exists("eng_processed.csv"):
    english_df[['sentence1', 'sentence2']] = english_df['Text'].str.split('\n', n=1, expand=True)
    english_df['label'] = english_df['Score']
    english_df_proc = english_df[['sentence1', 'sentence2', 'label']]
    english_df_proc.to_csv("eng_processed.csv", encoding='utf-8')
else: 
    english_df_proc = pd.read_csv('eng_processed.csv', encoding='utf-8')

# Translate chunked lists in the main list
translator = GoogleTranslator(source='en', target='af')

chunk_size = 50

sentence1_chunks = chunk_list(english_df_proc['sentence1'].tolist(), chunk_size)
sentence2_chunks = chunk_list(english_df_proc['sentence2'].tolist(), chunk_size)

translated_sent1_chunks = []
translated_sent2_chunks = []

for i, chunk in enumerate(sentence1_chunks):
    if os.path.exists(f"sent1_translated_chunk_{i + 1}.csv"):
        chunk_df = pd.read_csv(f"sent1_translated_chunk_{i + 1}.csv", encoding='utf-8')
        translated_sent1_chunks.extend(chunk_df['translated_sentence'].tolist())        
        continue
    try:
        translated = translator.translate_batch(chunk)
        translated_sent1_chunks.extend(translated)
        print(f"✅ Translated chunk {i + 1}")
        df_chunk = pd.DataFrame(translated, columns=['translated_sentence'])
        df_chunk.to_csv(f"sent1_translated_chunk_{i + 1}.csv", index=False, encoding='utf-8')
        time.sleep(5)  # add a short delay (1-2 sec) to be polite to the server
    except Exception as e:
        print(f"❌ Failed on chunk {i + 1}: {e}")
        translated_sent1_chunks.extend(["[ERROR]"] * len(chunk))

for i, chunk in enumerate(sentence2_chunks):
    if os.path.exists(f"sent2_translated_chunk_{i + 1}.csv"):
        chunk_df = pd.read_csv(f"sent2_translated_chunk_{i + 1}.csv", encoding='utf-8')
        translated_sent2_chunks.extend(chunk_df['translated_sentence'].tolist())      
        continue
    try:
        translated = translator.translate_batch(chunk)
        translated_sent2_chunks.extend(translated)
        print(f"✅ Translated chunk {i + 1}")
        df_chunk = pd.DataFrame(translated, columns=['translated_sentence'])
        df_chunk.to_csv(f"sent2_translated_chunk_{i + 1}.csv", index=False, encoding='utf-8')
        time.sleep(5)  # add a short delay (1-2 sec) to be polite to the server
    except Exception as e:
        print(f"❌ Failed on chunk {i + 1}: {e}")
        translated_sent2_chunks.extend(["[ERROR]"] * len(chunk))

if not os.path.exists("translated_eng.csv"):
    translated_sent1 = np.array(translated_sent1_chunks)
    translated_sent2 = np.array(translated_sent2_chunks)
    
    translated_sent1 = translated_sent1.flatten()
    translated_sent2 = translated_sent2.flatten()

    print(len(translated_sent1), len(translated_sent2))
    translated_eng_df = pd.DataFrame({
    'sentence1': translated_sent1,
    'sentence2': translated_sent2
    })
    translated_eng_df['label'] = english_df_proc['label']
    translated_eng_df.to_csv("translated_eng.csv", encoding='utf-8')
else:
    translated_eng_df = pd.read_csv('translated_eng.csv', encoding='utf-8')

# Augment afrikaans dataset with translated english dataset (Final dataset for fine tuning)
if not os.path.exists("combined_dataset_cleaned.csv"):
    df_combined = pd.concat([afr_df, translated_eng_df], ignore_index=True)
    df_combined.to_csv("combined_dataset.csv", index=False, encoding='utf-8')
else:
    df_combined = pd.read_csv("combined_dataset.csv", encoding='utf-8')