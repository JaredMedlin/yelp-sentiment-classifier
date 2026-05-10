# Yelp Review Sentiment Classifier

**WGU MSDA D213 Task 2 | Jared Medlin | March 2026**

## Overview

A feedforward neural network that classifies Yelp restaurant reviews as 
positive or negative sentiment at the sentence level, built as a proof-of-concept 
for automated voice-of-customer monitoring.

## Research Question

Can a neural NLP model accurately classify sentence-level sentiment from 
Yelp restaurant reviews?

## Key Results

- Test accuracy: **83.33%** (correctly classifies 5 out of every 6 reviews)
- Model: Embedding → GlobalAveragePooling1D → Dense(64, ReLU) → Dense(1, Sigmoid)
- Early stopping triggered at epoch 12, restoring best weights from epoch 9
- Training accuracy reached 98% vs. 83% test accuracy — overfitting present,
  attributed to small dataset size (700 training samples)

## Tools & Methods

- **Language:** Python 3
- **Libraries:** TensorFlow 2.15, Keras, pandas, numpy
- **Techniques:** TextVectorization, learned word embeddings, binary 
  cross-entropy loss, Adam optimizer, early stopping, stratified train/val/test split

## Dataset

Yelp Reviews dataset — 1,000 labeled sentences (500 positive, 500 negative),  
split 700 train / 150 validation / 150 test with stratification.

## Repository Contents

| File | Description |
|------|-------------|
| `JaredMedlinD213Task2.pdf` | Full written analysis report |
| `D213 Task 2(2).ipynb` | Jupyter Notebook with code and outputs |
| `data/` | Labeled Yelp review dataset, prepaired training data, and keras sentiment model|
