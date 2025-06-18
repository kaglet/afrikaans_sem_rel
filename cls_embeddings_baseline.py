# CLS Embeddings to Logistic Regression Baseline
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

csv_path = os.path.join(drive_root, "combined_dataset_cleaned.csv")
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Load AfroXLM-R tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")
model = AutoModel.from_pretrained("Davlan/afro-xlmr-base").to(device)
model.eval()

# Get CLS summary embeddings for batches of sentences
def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeds)
    return np.vstack(all_embeddings)

# Generate embeddings for both sentence columns
embeddings_1 = get_embeddings(df["sentence1"].fillna("").tolist(), tokenizer, model)
embeddings_2 = get_embeddings(df["sentence2"].fillna("").tolist(), tokenizer, model)

# Combine embeddings
X = np.hstack([embeddings_1, embeddings_2])

# Target labels
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# After fitting your model
results_path = os.path.join(drive_root, "regression_model.joblib")
joblib.dump(regressor, results_path)
# Later, to load it back
# clf_loaded = joblib.load(model_path)

# Predict on test set
y_pred = regressor.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"R2 Score: {r2:.4f}")
