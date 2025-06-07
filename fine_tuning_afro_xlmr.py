import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import pandas as pd
import os
import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

df = pd.read_csv('combined_dataset_cleaned.csv', encoding='ISO-8859-1')
dataset = Dataset.from_pandas(df)
print(df.columns)
print("✅ Loaded dataset")

model_name = "Davlan/afro-xlmr-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(ds):
    return tokenizer(ds["sentence1"], ds["sentence2"], truncation=True, padding="max_length", max_length = 128)

tokenized = dataset.map(preprocess, batched=True)
print("✅ Tokenization complete")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression").to(device)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    pearson = pearsonr(labels, predictions)[0]
    return {"mse": mse, "mae": mae, "pearson": pearson}

def quick_train(model_ckpt):
    tokenized = tokenized.train_test_split(test_size=0.2, seed=2)
    print("✅ Saving tokenized dataset to disk")    
    tokenized.save_to_disk(f"{model_ckpt}-tokenized")
    print("✅ Save complete!")  
    train_ds = tokenized["train"].shuffle(seed=2).select(range(300))  # small subset
    eval_ds = tokenized["test"].shuffle(seed=2).select(range(100))

    args = TrainingArguments(
        output_dir=f"{model_ckpt}-quick-ft",
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=16,
        num_train_epochs=1, #TODO: try 2
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
    trainer.save_model(args.output_dir)
    trainer.save_metrics(args.output_dir)
    # tokenizer.save_pretrained(output_dir)
    print("✅ Save complete!")    

    return model, tokenizer, results

def full_train(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=1, problem_type="regression"
    ).to(device)

    tokenized = dataset.map(preprocess, batched=True)

    tokenized = tokenized.train_test_split(test_size=0.2, seed=42)
    train_ds = tokenized["train"]
    val_ds = tokenized["test"]

    args = TrainingArguments(
        output_dir=f"{model_ckpt}-final-ft",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        save_strategy="epoch",
        report_to="none",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer.evaluate()

model, tokenizer, results = quick_train(model)
print(results)

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