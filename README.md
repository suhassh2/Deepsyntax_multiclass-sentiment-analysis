# Multiclass Sentiment Analysis using BERT

## Overview

This project demonstrates transfer learning by fine-tuning a pre-trained BERT model for multiclass sentiment classification using PyTorch.

The goal is to classify text into three sentiment categories using a BERT-based model.

## Dataset

We used the Hugging Face dataset:

Sp1786/multiclass-sentiment-analysis-dataset

The dataset contains text samples labeled into three classes:
- Negative
- Neutral
- Positive

## Exploratory Data Analysis

We analyzed the dataset and visualized the class distribution using a bar chart.  
The dataset shows slight class imbalance but is relatively balanced overall.

## Model Details

- Model: bert-base-uncased
- Framework: PyTorch
- Optimizer: AdamW
- Epochs: 2
- Batch size: 16
- Learning rate: 2e-5

We implemented a custom PyTorch training loop (HuggingFace Trainer API was not used).

## Results

- Accuracy: 76.7%
- Weighted F1-score: 0.77

The confusion matrix shows that most predictions lie on the diagonal, indicating correct classifications.

## Inference

A custom `predict_text()` function was implemented to classify raw input text and return:
- Predicted sentiment label
- Confidence score

## Conclusion

This project shows that transfer learning with BERT provides good performance on multiclass sentiment classification tasks.
