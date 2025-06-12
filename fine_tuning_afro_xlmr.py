!pip install sentence-transformers torch torchvision scikit-learn scipy pandas numpy

import json
import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
import random
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except ImportError:
    print("Not running in Google Colab or Drive mount not needed")

drive_root = "/content/drive/MyDrive/COS760/project"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU not available - make sure to enable GPU in Runtime > Change runtime type")

# Configuration
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

# Load dataset
csv_path = os.path.join(drive_root, "combined_dataset_cleaned.csv")
df = pd.read_csv(csv_path, encoding='ISO-8859-1')
print("✅ Loaded dataset")
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
dataset = Dataset.from_pandas(df)
print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {df.columns}")
print(f"Sample data:")
print(df.head(3))

model_name = "Davlan/afro-xlmr-large"

# Preprocess (tokenize)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(ds):
    return tokenizer(ds["sentence1"], ds["sentence2"], truncation=True, padding="max_length", max_length = 128)

"""Set seeds for all random number generators to ensure reproducibility"""
def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All seeds set to {seed}")

set_all_seeds(SEED)

def compute_metrics(preds):
    predictions, labels = preds
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    pearson = pearsonr(labels, predictions)[0]
    spearman = spearmanr(labels, preds)
    return {"mse": mse, "mae": mae, "pearson": pearson, "spearman": spearman}

def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, seed=SEED):
    # First split: separate train from (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=seed,
        shuffle=True
    )

    # Second split: separate val from test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio,
        random_state=seed,
        shuffle=True
    )

    return train_df, val_df, test_df
   
train_df, eval_df, test_df = split_dataset(df)
train_ds = Dataset.from_pandas(train_df)
eval_ds = Dataset.from_pandas(eval_df)
test_ds = Dataset.from_pandas(test_df)
print("✅ Dataset splitting complete")
tokenized_train = train_ds.map(preprocess, batched=True)
tokenized_eval = test_ds.map(preprocess, batched=True)
tokenized_test = test_ds.map(preprocess, batched=True)
print("✅ Tokenization complete")
print("✅ Saving tokenized splits of dataset to disk")    
tokenized_train.save_to_disk("train_afro-xlmr-final-ft")
tokenized_eval.save_to_disk("train_afro-xlmr-final-ft")
tokenized_test.save_to_disk("train_afro-xlmr-final-ft")
print("✅ Save complete!")  

# Get model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression").to(device)

print(device)

def full_train(model_ckpt, tokenized_train, tokenized_eval, tokenized_test, drive_root):
    args = TrainingArguments(
        output_dir="afro-xlmr-final-ft",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        # weight_decay=0.01,
        num_train_epochs=4, 
        save_strategy="epoch",
        report_to="none",
        logging_dir="./logs",
        logging_steps=10,  # Show logging every 10 steps
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics
    )

    print("✅ Training started")
    trainer.train()
    print("✅ Training finished")

    print("✅ Evaluating by metrics...")
    results = trainer.evaluate()
    print("✅ Evaluation by metrics done!")
 
    print("✅ Saving model and metrics to disk")    
    trainer.save_to_disk(os.path.join(drive_root, "checkpoint"))
    trainer.save_to_disk(os.path.join(drive_root, "checkpoint"))
    results_path = os.path.join(drive_root, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Save complete!")    

    return model, tokenizer, results

model, tokenizer, results = full_train(model, tokenized_train, tokenized_eval, tokenized_test, drive_root)
print("Results:")
print(results)