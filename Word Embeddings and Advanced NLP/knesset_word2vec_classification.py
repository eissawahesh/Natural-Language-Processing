import argparse
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
def tokenization(sentences):
    tokenized_sentences=[]
    for sentence in sentences:
        tokenized_sentence=sentence.split()
        clear_sentence=[token for token in tokenized_sentence if any(check_hebrewchar(char) for char in token )]
        tokenized_sentences.append(clear_sentence)
    return tokenized_sentences
def check_hebrewchar(char):
    return '\u0590' <= char <= '\u05FF'
def sentence_embedding(sentence_tokens, model):
    embedding = np.mean([model.wv[token] for token in sentence_tokens], axis=0)
    return embedding


def generate_embeddings(texts, model):
    tokenized_texts = tokenization(texts)
    embeddings = np.array([sentence_embedding(tokens, model) for tokens in tokenized_texts])
    return embeddings

def main(input_path1,input_path2):
    file_path = input_path1
    list_protocol_plenary, list_protocol_committee = read_file(file_path)
    model=Word2Vec.load(input_path2)
    chunk_sizes=[1,3,5]
    for chunk_size in chunk_sizes:
        plenary_chunks = divide_into_chunks(chunk_size, list_protocol_plenary)
        committee_chunks = divide_into_chunks(chunk_size, list_protocol_committee)

        target_size = min(len(plenary_chunks), len(committee_chunks))

        plenary_chunks = downs_sampling(plenary_chunks, target_size)
        committee_chunks = downs_sampling(committee_chunks, target_size)

        plenary_texts = chunks_to_text(plenary_chunks)
        committee_texts = chunks_to_text(committee_chunks)

        all_texts = plenary_texts + committee_texts
        all_labels = ["plenary"] * len(plenary_texts) + ["committee"] * len(committee_texts)

        X=generate_embeddings(all_texts,model)
        y = np.array(all_labels)
        knn_classifier = KNeighborsClassifier(n_neighbors=51)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42, shuffle=True)
        print("KNN-chunk size",chunk_size)
        testing_split(X_train, X_test, y_train, y_test, knn_classifier)


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path1")
    parser.add_argument("input_path2")
    args = parser.parse_args()
    main(args.input_path1,args.input_path2)
if __name__ == '__main__':
    arg()
