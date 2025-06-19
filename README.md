# Evaluating Cross-Lingual Semantic Relatedness in African Languages

This project investigates the effectiveness of multilingual pre-trained language models for measuring semantic relatedness in low-resource African languages, with a focus on Afrikaans. We benchmark traditional statistical approaches and embedding-based baselines against fine-tuned transformer models like AfroXLM-R and LaBSE. The study also utilizes interpretability methods, including LIME, to examine model reasoning.

## Run Instructions

1.  **Fork and Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. In root directory run 
```bash
pip install -r requirements.txt 
```

3.  **Run Baselines (Training and Metrics Output):**
    ```bash
    python baselines/cls_embeddings_baseline.py
    python baselines/tfidf_baseline.py
    ```

4.  **Run Fine-Tuned Models (Training and Metrics Output):**
    * Open and run the Jupyter notebooks:
        ```bash
        jupyter notebook finetune_models/finetuning_afroxlmr.ipynb
        jupyter notebook finetune_models/finetuning_labse.ipynb
        ```

## Problem Statement

* Which multilingual pre-trained language models (AfroXLM-R, LaBSE) achieve the best performance in identifying semantic relatedness in African language texts when fine-tuned using transfer learning?
* Can interpretability tools such as LIME help explain how these models arrive at their predictions?

## Datasets

The project uses a combination of:

* **SemRel2024 (Afrikaans Subset):** 751 human-annotated sentence pairs from SemEval-2024 Task 1.
* **SemRel2022 (English):** 5499 English sentence pairs, back-translated into Afrikaans using Google Translate for data augmentation.

The combined dataset is split into 70% for training and 30% for testing using stratified sampling.

## Models

### Baseline Models

* **TF-IDF to Linear Regression:** A traditional statistical approach.
* **CLS Embeddings to Linear Regression:** Utilizes frozen AfroXLMR embeddings.

### Fine-Tuned Models

* **AfroXLMR:** An adaptation of the multilingual XLM-RoBERTa transformer specifically designed for African languages, fine-tuned with a regression head.
    * **Training Parameters:** Seed: 42, Batch size: 32, Epochs: 4, Learning rate: $2 \times 10^{-5}$
    * **Loss Function:** Mean Squared Error (MSE)
* **LaBSE (Language-Agnostic BERT Sentence Embeddings):** A multilingual model for sentence embeddings supporting 109 languages, fine-tuned with a regression head.
    * **Training Parameters:** Seeds: 42, 1042, ..., 9042, Batch size: 32, Epochs: 4, Learning rate: $2 \times 10^{-5}$
    * **Loss Function:** Cosine similarity loss

## Evaluation Metrics

* Pearson Correlation
* Spearman Correlation
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

## Interpretability

LIME (Local Interpretable Model-agnostic Explanations) is used to understand model predictions by highlighting influential words or phrases in the input sentences.


## Results and Interpretability

All relevant results, including metrics and LIME interpretability images, will be saved in the `saves/eval_results` directory.

## Performance Comparison

| Model     | Avg Pearson | Avg Spearman | Avg MSE  | Avg MAE  |
| :-------- | :---------- | :----------- | :------- | :------- |
| AfroXLM-R | 0.6508      | 0.6532       | 0.0324   | 0.1282   |
| LaBSE     | 0.8253      | 0.8252       | 0.0166   | 0.1015   |
| TF-IDF    | 0.1559      | 0.1691       | 0.0841   | 0.2303   |
| CLS       | 0.1982      | 0.2014       | 0.0668   | 0.2062   |

LaBSE outperforms other models, showing stronger alignment with human-annotated similarity scores for Afrikaans.

## Team Members
* Kago Motlhabane
* Nevin Thomas
* Jaimen Govender
