import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
import random
import os
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Configuration
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

# Load dataset
df = pd.read_csv('combined_dataset_cleaned.csv', encoding='ISO-8859-1')
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

tokenized = dataset.map(preprocess, batched=True)
print("✅ Tokenization complete")

# Get model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression").to(device)

print(device)

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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    pearson = pearsonr(labels, predictions)[0]
    return {"mse": mse, "mae": mae, "pearson": pearson}

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
   

def quick_train(model_ckpt, tokenized):
    tokenized  = tokenized.train_test_split(test_size=0.2, seed=2)
    print("✅ Saving tokenized dataset to disk")    
    tokenized.save_to_disk("afro-xlmr-final-ft")
    print("✅ Save complete!")  
    train_ds, eval_ds, test_ds = train_test_split(
        df
    )
    train_ds = tokenized["train"].shuffle(seed=2).select(range(300))  # small subset
    eval_ds = tokenized["test"].shuffle(seed=2).select(range(100))

    args = TrainingArguments(
        output_dir="afro-xlmr-final-ft",
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=16,
        num_train_epochs=3, #TODO: try 2
        save_strategy="no",
        report_to="none",
        logging_dir="./logs",
        logging_steps=10,  # Show logging every 10 steps
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )

    print("✅ Training started")
    trainer.train()
    print("✅ Training finished")

    print("✅ Evaluating by metrics...")
    results = trainer.evaluate()
    print("✅ Evaluation by metrics done!")

    output_dir = "saves"

    print("✅ Saving model and metrics to disk")    
    trainer.save_model("checkpoint")
    # trainer.save_metrics(args.output_dir)
    # tokenizer.save_pretrained(output_dir)
    print("✅ Save complete!")    

    return model, tokenizer, results

def full_train(model_ckpt, tokenized):
    tokenized  = tokenized.train_test_split(test_size=0.2, seed=2)
    print("✅ Saving tokenized dataset to disk")    
    tokenized.save_to_disk("afro-xlmr-final-ft")
    print("✅ Save complete!")  
    train_ds = tokenized["train"].shuffle(seed=2)
    eval_ds = tokenized["test"].shuffle(seed=2)

    args = TrainingArguments(
        output_dir="afro-xlmr-final-ft",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5, #TODO: try 2
        save_strategy="epoch",
        report_to="none",
        logging_dir="./logs",
        logging_steps=10,  # Show logging every 10 steps
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )

    print("✅ Training started")
    trainer.train()
    print("✅ Training finished")

    print("✅ Evaluating by metrics...")
    results = trainer.evaluate()
    print("✅ Evaluation by metrics done!")

    output_dir = "saves"

    print("✅ Saving model and metrics to disk")    
    trainer.save_model("checkpoint")
    # trainer.save_metrics(args.output_dir)
    # tokenizer.save_pretrained(output_dir)
    print("✅ Save complete!")    

    return model, tokenizer, results

# model, tokenizer, results = quick_train(model, tokenized)
# print(results)

model, tokenizer, results = full_train(model, tokenized)


# args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train()