import random
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

random.seed(42)
np.random.seed(42)


def read_file(file_path):
    df = pd.read_csv(file_path)
    list_protocol_plenary = df[df['protocol_type'] == 'plenary']['sentence_text'].tolist()
    list_protocol_committee = df[df['protocol_type'] == 'committee']['sentence_text'].tolist()
    return list_protocol_plenary, list_protocol_committee


def divide_into_chunks(chunk_size, sentence_list):
    chunk_num = len(sentence_list) // chunk_size
    chunks = []
    for i in range(chunk_num):
        first = i * chunk_size
        last = first + chunk_size
        chunk = sentence_list[first:last]
        chunks.append(chunk)
    return chunks


def chunks_to_text(chunks):
    list = []
    for chunk in chunks:
        list2 = []
        for sentence in chunk:
            list2.append(sentence)
        list.append(' '.join(list2))
    return list


def downs_sampling(chunks, target_size):
    return random.sample(chunks, target_size)


def testing_split(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(" Train-Test Split:")
    print(classification_report(y_test, y_pred))


def cross_validation(X, y, K_fold, classifier):
    # cv = StratifiedKFold(n_splits=K_fold, shuffle=True, random_state=42)
    y_pred = cross_val_predict(classifier, X, y, cv=K_fold, n_jobs=-1)
    print("Cross - Validation:")
    print(classification_report(y, y_pred))


def read_data_to_predict(file_path):
    chunks = []
    with open(file_path, "r", encoding="utf-8") as file:
        for sentence in file:
            chunks.append(sentence.strip())
    return chunks


def print_predict(victorized_data, classifier,file_path):
    y_pred = classifier.predict(victorized_data)
    with open(file_path, 'w', encoding='utf-8') as file:
        for y in y_pred:
            file.write(y + "\n")


# def _vectorizer(texts):
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(texts)
# return X, vectorizer
#
def _vectorizer(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def get_top_features(matrix, vectorizer, n):
    sums = np.array(matrix.sum(axis=0)).flatten()
    words_sum = []
    for i, word in enumerate(vectorizer.get_feature_names_out()):
        words_sum.append((word, sums[i]))
    sorted_words_sums = sorted(words_sum, key=lambda x: x[1], reverse=True)
    top_n_words = []
    for i in range(n):
        word = sorted_words_sums[i][0]
        top_n_words.append(word)
    return top_n_words

def creating_feature_vector(all_texts,top_words):
    most_appeared_words=top_words
    punctuation_marks=".,:!?"
    vector = []
    for chunk in (all_texts):
        words = chunk.split()
        words_lengths = [len(word) for word in words]
        words_count = [chunk.count(word) for word in most_appeared_words]
        punctuation_count = sum(chunk.count(p) for p in punctuation_marks)
        number_count = sum(char.isdigit() for char in chunk)
        words_lengths_avg = np.mean(words_lengths) if words_lengths else 0
        num_words = len(words)
        words_lengths_var = np.var(words_lengths) if words_lengths else 0
        vector.append(words_count + [words_lengths_avg, num_words, words_lengths_var,punctuation_count, number_count])

    return np.array(vector)

def testing(X,y,svm_classifier,knn_classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42, shuffle=True)
    print("SVM")
    testing_split(X_train, X_test, y_train, y_test, svm_classifier)
    print("KNN")
    testing_split(X_train, X_test, y_train, y_test, knn_classifier)
    k_fold = 10
    print("SVM")
    cross_validation(X, y, k_fold, svm_classifier)
    print("KNN")
    cross_validation(X, y, k_fold, knn_classifier)


def main():
    file_path = "example_knesset_corpus.csv"
    list_protocol_plenary, list_protocol_committee = read_file(file_path)
    chunk_size=5
    plenary_chunks = divide_into_chunks(chunk_size, list_protocol_plenary)
    committee_chunks = divide_into_chunks(chunk_size, list_protocol_committee)

    target_size = min(len(plenary_chunks), len(committee_chunks))

    plenary_chunks = downs_sampling(plenary_chunks, target_size)
    committee_chunks = downs_sampling(committee_chunks, target_size)

    plenary_texts = chunks_to_text(plenary_chunks)
    committee_texts = chunks_to_text(committee_chunks)

    all_texts = plenary_texts + committee_texts
    all_labels = ["plenary"] * len(plenary_texts) + ["committee"] * len(committee_texts)

    X, vectorizer = _vectorizer(all_texts)
    top_features = get_top_features(X,vectorizer,100)
    X_feature_vector = creating_feature_vector(all_texts,top_features)
    y = np.array(all_labels)

    svm_classifier = SVC(kernel="linear", random_state=42)
    knn_classifier = KNeighborsClassifier(n_neighbors=51, n_jobs=-1)

    print("TfidfVector:")
    testing(X,y, svm_classifier, knn_classifier)
    print("Our Veature Vector")
    testing(X_feature_vector,y,svm_classifier, knn_classifier)

    to_be_predicted = read_data_to_predict("knesset_text_chunks.txt")
    svm_classifier.fit(X, y)
    print_predict(vectorizer.transform(to_be_predicted), svm_classifier,file_path="classification_results.txt")


if __name__ == '__main__':
    main()
