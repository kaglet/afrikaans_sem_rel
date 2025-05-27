from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd

english_df = pd.read_csv('sem_text_rel_ranked.csv')
afr_df_dev = pd.read_parquet("dev.parquet")
afr_df_test = pd.read_parquet("test.parquet")

afr_df_dev.to_csv("afr_df_dev.csv", index=False)
afr_df_test.to_csv("afr_df_test.csv", index=False)

