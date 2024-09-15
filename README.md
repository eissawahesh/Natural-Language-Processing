# Natural-Language-Processing
This repository contains four exercises focusing on Natural Language Processing (NLP) tasks using Hebrew texts from the Knesset protocols. The exercises guide you through various stages of NLP, including corpus creation, text classification, language modeling, and the use of word embeddings.
Exercise 1: Corpus Creation and Text Processing
•	Description: This exercise involves building a text corpus from the Knesset protocols, stored as Word documents. It covers the extraction of textual data, sentence splitting, tokenization, and cleaning the data by removing non-Hebrew and malformed sentences. Additionally, it involves implementing Zipf's law to analyze the word frequency distribution within the corpus.
•	Algorithms and Techniques Used:
o	Tokenization
o	Zipf's Law (word frequency analysis)
Exercise 2: N-grams Language Modeling
•	Description: This task focuses on building language models using trigrams. It involves constructing separate trigram models for the committee and plenary protocols, calculating sentence probabilities using methods like Maximum Likelihood Estimation (MLE), and identifying frequent collocations within the corpus using the Pointwise Mutual Information (PMI) metric.
•	Algorithms and Techniques Used:
o	Trigram Language Models
o	Smoothing Techniques (Laplace, Linear Interpolation)
o	Pointwise Mutual Information (PMI) for collocation extraction
Exercise 3: Text Classification
•	Description: This exercise explores text classification using sentences extracted from the Knesset protocols. The goal is to classify each sentence into one of two categories: "committee" or "plenary." The exercise covers feature vector creation using methods like Bag of Words (BoW) and TF-IDF, and training classifiers like K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). Evaluation methods such as 10-fold cross-validation and train-test split are also included.
•	Algorithms and Techniques Used:
o	K-Nearest Neighbors (KNN)
o	Support Vector Machines (SVM)
o	Feature Extraction (Bag of Words, TF-IDF)
o	Cross-Validation and Train-Test Split
Exercise 4: Word Embeddings and Advanced NLP
•	Description: This task delves into word embeddings, focusing on training a Word2Vec model using the Knesset corpus. It includes tasks such as finding similar words, computing sentence embeddings, and replacing words in context. The latter part introduces the use of large language models like DictaBERT for masked language modeling.
•	Algorithms and Techniques Used:
o	Word2Vec
o	Cosine Similarity
o	Sentence Embeddings
o	Masked Language Modeling with DictaBERT

