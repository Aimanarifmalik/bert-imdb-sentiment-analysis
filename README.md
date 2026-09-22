# bert-imdb-sentiment-analysis
Fine-tuned BERT for IMDb movie review sentiment classification using Hugging Face Transformers.
# BERT IMDb Sentiment Analysis

A Natural Language Processing project that fine-tunes **BERT (`bert-base-uncased`)** for binary sentiment classification of IMDb movie reviews using Hugging Face Transformers.

## Project Overview

This project demonstrates how a pretrained Transformer model can be adapted to a downstream NLP classification task.

The workflow includes:

* Loading the IMDb dataset using Hugging Face Datasets
* Tokenizing movie reviews using a BERT tokenizer
* Fine-tuning `bert-base-uncased`
* Training with Hugging Face `Trainer`
* Evaluating model performance
* Generating predictions for custom movie reviews
* Uploading the trained model to Hugging Face Hub

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Hub
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter / Google Colab

## Model

**Base Model:** `bert-base-uncased`

**Task:** Binary Sentiment Classification

**Classes:**

* Negative
* Positive

The IMDb dataset contains movie reviews labeled as positive or negative.

## Dataset

The project uses the IMDb dataset available through Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
```

For efficient experimentation on CPU, a smaller subset of the dataset was used for fine-tuning.

## Methodology

### 1. Dataset Loading

The IMDb dataset was loaded using the Hugging Face Datasets library.

### 2. Tokenization

Reviews were converted into BERT-compatible input tokens using:

```python
AutoTokenizer.from_pretrained("bert-base-uncased")
```

Reviews were truncated/padded to a maximum length of 128 tokens.

### 3. Model Fine-Tuning

A pretrained BERT model was loaded using:

```python
AutoModelForSequenceClassification
```

The model was fine-tuned for two classes:

```text
0 → Negative
1 → Positive
```

### 4. Training

Training was performed using Hugging Face's `Trainer` API.

The final experimental run used:

* 500 training samples
* 200 evaluation samples
* 1 epoch
* Batch size: 16
* Learning rate: `2e-5`

The smaller subset was intentionally used because the experiment was performed on CPU.

## Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Custom movie reviews were also passed through the trained model to test its predictions.

## Hugging Face Model

The trained model is available on Hugging Face Hub:

**Aimaaann/model-push**

## Project Structure

```text
bert-imdb-sentiment-analysis/
│
├── BERT_IMDB_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## What I Learned

This project helped me understand the practical workflow of Transformer-based NLP:

```text
Raw Text
   ↓
IMDb Dataset
   ↓
BERT Tokenizer
   ↓
Tokenized Inputs
   ↓
Pretrained BERT
   ↓
Fine-Tuning
   ↓
Sentiment Prediction
   ↓
Evaluation
   ↓
Hugging Face Hub
```

The project also provided hands-on experience with pretrained Transformer models, tokenization, fine-tuning, evaluation, and model deployment using the Hugging Face ecosystem.
