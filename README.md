## Sentiment Analysis
Author: Andrew Kwon

## Description
Model training, evaluation, and comparison for classification task using sentiment analysis. The language processing libraries/techniques and models covered in this project are as follows:
- NLTK + TF-IDF vectorization (Logisitc Regression)
- spaCy + TF-IDF vectorization (Logisitc Regression)
- spaCy + TF-IDF vectorization (LightGBM Classifier)
- BERT (Logsitic Regression)

These model and text processor combinations were chosen for runtime considerations. The BERT embeddings generation incorporates GPU/CUDA support; users can expect significantly longer runtimes if run on CPU only. Additional tasks covered in this project include exploratory data analsysis and visualization. Code solution and analysis conducted in Jupyter notebook.

## Introduction
In this project, we'll develop a system for filtering and categorizing movie reviews using sentiment analysis. The goal is to train a model to automatically detect negative reviews. We'll be using a dataset of IMBD movie reviews with polarity labeling to build a model for classifying positive and negative reviews. Our evaluation goal is to achieve an F1 score of at least 0.85.

## Dataset
**imdb_reviews.tsv**:
- File size too large for upload, users will need to extract the 7zip archive to the appropriate directory prior to loading the data
- Full dataset described in notebook
- Only columns used for main task are review (text processing) and pos (target variable) 

## Required Libraries
- math
- pandas
- numpy
- re
- spacy
- torch
- transformers
- matplotlib.pyplot
- matplotlib.dates
- seaborn
- sklearn.metrics
- sklearn.linear_model
- sklearn.dummy
- sklearn.feature_extraction.text
- lightgbm
- nltk.tokenize
- nltk.stem
- nltk.corpus
- tqdm.auto

## Screenshots

![5e6e2648-b129-4f87-9ed7-96f9f4bd6c6b](https://github.com/adkwn1/sentiment_analysis/assets/119823114/5bca37c1-6385-4e8c-98ec-e72124a87f4d)
![cc1ea493-4dc5-4ad0-b762-f01a538d7ab1](https://github.com/adkwn1/sentiment_analysis/assets/119823114/30d3ce0e-010a-4530-8bb6-45114d31ecab)
![5687aeaa-953b-44cc-b00d-5a5988a247c4](https://github.com/adkwn1/sentiment_analysis/assets/119823114/57540f3c-0a7b-43f3-8849-4a7aac9df907)
