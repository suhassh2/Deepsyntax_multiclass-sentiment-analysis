# Multiclass Sentiment Analysis using BERT

## Overview

In this project, we performed transfer learning by fine-tuning a pre-trained BERT model for multiclass sentiment classification using PyTorch.

The objective was to classify text into three sentiment categories using a BERT-based architecture.

---

## Dataset

We used the Hugging Face dataset:

Sp1786/multiclass-sentiment-analysis-dataset

The dataset consists of text samples labeled into three sentiment classes:
- Negative
- Neutral
- Positive

The dataset contains separate train and test splits.

---

## Approach

1. Loaded the dataset using the HuggingFace `datasets` library.
2. Performed Exploratory Data Analysis (EDA) to check:
   - Dataset size
   - Class distribution
3. Visualized class distribution using a bar chart.
4. Used `bert-base-uncased` as the pre-trained model.
5. Tokenized the text using `BertTokenizer`.
6. Converted the dataset into PyTorch format.
7. Created custom DataLoaders for training and testing.
8. Implemented a custom PyTorch training loop (HuggingFace Trainer API was NOT used).
9. Fine-tuned the model for 2 epochs using AdamW optimizer.
10. Evaluated the model using:
    - Accuracy
    - Precision
    - Recall
    - F1-score (weighted)
    - Confusion Matrix
11. Implemented a `predict_text()` function for inference on custom input text.

---

## Assumptions

- Maximum sequence length was set to 128 tokens.
- Batch size was set to 16 to balance performance and GPU memory usage.
- Training was performed for 2 epochs due to time and resource constraints.
- Weighted F1-score was used because slight class imbalance was observed.

---

## Model Details

- Model: bert-base-uncased
- Framework: PyTorch
- Optimizer: AdamW
- Epochs: 2
- Batch Size: 16
- Learning Rate: 2e-5

---

## Results

- Accuracy: 76.7%
- Weighted F1-score: 0.77

The confusion matrix shows that most predictions lie on the diagonal, indicating correct classifications.

The model performs slightly better on the positive class compared to other classes.

---

## Observations

- Loss decreased significantly from Epoch 1 to Epoch 2, indicating successful fine-tuning.
- The model generalizes reasonably well to the test set.
- Some misclassification occurs between neutral and negative classes, possibly due to similarity in sentiment expression.
- Transfer learning with BERT provides strong performance even with limited training epochs.

---

## Inference

A custom `predict_text(text: str)` function was implemented to classify raw text input and return:
- Predicted sentiment label
- Confidence score

Example:

Input: "I absolutely loved this movie!"
Output: Positive (Confidence ~0.98)

---

## Conclusion

This project demonstrates that transfer learning using BERT is highly effective for multiclass sentiment classification tasks. Even with limited fine-tuning, the model achieves strong performance.

