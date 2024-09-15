import argparse
import random
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def read_data(file_path):
    df = pd.read_csv(file_path)
    sentences = df["sentence_text"].tolist()
    return sentences
def tokenization(sentences):
    tokenized_sentences=[]
    for sentence in sentences:
        tokenized_sentence=sentence.split()
        clear_sentence=[token for token in tokenized_sentence if any(check_hebrewchar(char) for char in token )]
        tokenized_sentences.append(clear_sentence)
    return tokenized_sentences
def check_hebrewchar(char):
    return '\u0590' <= char <= '\u05FF'

def calculate_top_similarities(model, word_list, top_n=5):
    list_of_top_similarities=[]
    for word in word_list:
        if word in model.wv:
            similarities = []
            for other_word in model.wv.index_to_key:
                if other_word != word:
                    similarity_score = model.wv.similarity(word, other_word)
                    similarities.append((other_word, similarity_score))
            top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
            list_of_top_similarities.append(top_similarities)
        else:
            list_of_top_similarities.append("the word is not on the vocabulary")

    return list_of_top_similarities
def sentence_embedding(sentence_tokens, model):
    embedding = np.mean([model.wv[token] for token in sentence_tokens], axis=0)
    return embedding


def replace_word_in_sentences(sentences, model):
    replced_sentences=[]
    target_word="לחדר"
    most_similar_words = model.wv.most_similar(positive=[target_word,"פנימה","להיכנס"], topn=3)
    replaced_sentence=sentences[0].replace(target_word,most_similar_words[1][0])
    replced_sentences.append(replaced_sentence)
    target_word="מוכנה"
    most_similar_words = model.wv.most_similar(positive=[target_word,"אישה"],negative=["גבר"], topn=3)
    replaced_sentence=sentences[1].replace(target_word,most_similar_words[2][0])
    target_word="ההסכם"
    most_similar_words=model.wv.most_similar(positive=[target_word,"ההסדר"], topn=3)
    replaced_sentence=replaced_sentence.replace(target_word,most_similar_words[2][0])
    replced_sentences.append(replaced_sentence)
    target_word="טוב"
    most_similar_words=model.wv.most_similar(positive=[target_word,"בוקר","נהדר","משמח"], topn=3)
    replaced_sentence=sentences[2].replace(target_word,most_similar_words[2][0])
    target_word="פותח"
    most_similar_words = model.wv.most_similar(positive=[target_word,"אישה"],negative=["גבר"], topn=3)
    replaced_sentence=replaced_sentence.replace(target_word,most_similar_words[1][0])
    replced_sentences.append(replaced_sentence)
    target_word="שלום"
    most_similar_words = model.wv.most_similar(positive=[target_word,"נכבדי","גבירותי","רבותי"], topn=3)
    replaced_sentence=sentences[3].replace(target_word,most_similar_words[2][0])
    target_word="היקר"
    most_similar_words=model.wv.most_similar(positive=[target_word,"הישראלי","היפה"],negative=["אישה"],topn=3)
    replaced_sentence=replaced_sentence.replace(target_word,most_similar_words[0][0])
    target_word="בשנה"
    most_similar_words=model.wv.most_similar(positive=[target_word], topn=3)
    replaced_sentence=replaced_sentence.replace(target_word,most_similar_words[0][0])
    replced_sentences.append(replaced_sentence)
    return replced_sentences
def calc_cosine_similarity(target_sentences, all_sentence_embeddings,model):
    most_similar_indexes = []
    for target_sentence in target_sentences:
        target_embedding = sentence_embedding(target_sentence, model).reshape(1, -1)
        scores = cosine_similarity(target_embedding, all_sentence_embeddings)[0]
        most_similar_index = scores.argsort()[-2]
        most_similar_indexes.append(most_similar_index)
    return most_similar_indexes
def save_file_cosine_similarity(most_similar_indexes,list_of_sentences,sentences,file_path):
    with open(file_path+"\\Knesset_similar_sentences.txt", 'w', encoding='utf-8') as file:
        for i in range(len(list_of_sentences)):
            file.write(list_of_sentences[i]+": most similar sentence: "+sentences[most_similar_indexes[i]]+"\n")
def save_file_top_similarities(words_list,list_of_top_similarities,file_path):

    with open(file_path+"\\knesset_similar_words.txt", 'w', encoding='utf-8') as file:
        for i in range(len(words_list)):
            file.write(words_list[i]+": ")
            if(isinstance(list_of_top_similarities[i],list)):
                last_index=len(list_of_top_similarities[i])-1
                for other_word, score in list_of_top_similarities[i]:
                    if last_index:
                        file.write("("+other_word+","+str(score)+"),")
                    else:
                        file.write("(" + other_word + "," + str(score) + ")\n")
                    last_index -= 1
            else:
                file.write("the word is not on the vocabulary")

def save_file_red_words(sentences_to_rep,replaced_senteces,file_path):
    with open(file_path+"\\red_words_sentences.txt", 'w', encoding='utf-8') as file:
        for i in range(len(sentences_to_rep)):
            file.write(sentences_to_rep[i]+":"+replaced_senteces[i]+"\n")


def main(input_path,output_path):
    file_path=input_path
    sentences=read_data(file_path)
    tokenized_sentences=tokenization(sentences)
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
    model.save(output_path+"\\knesset_word2vec.model")
    model=Word2Vec.load(output_path+"\\knesset_word2vec.model")

    words_list=["ישראל","ממשלה","כנסת","חבר","שלום","שולחן"]
    list_of_top_similarities=calculate_top_similarities(model,words_list)
    save_file_top_similarities(words_list,list_of_top_similarities,output_path)
    sentence_embeddings = [sentence_embedding(sentence, model) for sentence in tokenized_sentences]
    list_of_sentences=['תודה רבה לחברת הכנסת זהבה גלאון .',"יש לנו גם את נציגי המסגרות , בטח נשמע אותם ."
     ,'לא אמרתי ששמעתי אותך .',"התופעה החלה לפני כעשור , בהיקף מאוד מאוד נרחב , בגלל הפער המאוד - גדול בין הרמה הכלכלית באפריקה , שהיא על הפנים , לבין הרמה הכלכלית בישראל , שהיא טובה .",
    "אדוני היושב - ראש , אדוני שר האוצר , בסקר החברתי על רווחת האוכלוסייה בישראל שפרסמה לפני כשבועיים הלשכה המרכזית לסטטיסטיקה , עולה ש- 46 % – כמחצית מאזרחי המדינה – אינם מצליחים לכסות את ההוצאות החודשיות .",
    "זה לא נושא של קואליציה - אופוזיציה .","הוא מימן אבל הוא לא רוצה להמשיך ולממן .","כנגד זה אנחנו נותנים לו שירות ב-  700 אלף שקל בשנה ."
    ,'להצעת החוק יש הסתייגויות , ואני מבקש את תמיכתכם בדחיית ההסתייגויות ובאישור ההצעה בקריאה שנייה וקריאה שלישית .',"רבותי חברי הכנסת , אנחנו עוברים להצעתו של חן רשף , שמבקש לבטל את החסינות הדיונית ."]
    list_of_tokenized_sentences=tokenization(list_of_sentences)
    most_similar_indexes=calc_cosine_similarity(list_of_tokenized_sentences,sentence_embeddings,model)
    save_file_cosine_similarity(most_similar_indexes,list_of_sentences,sentences,output_path)


    distance_score = model.wv.distance("רע", "טוב")
    similarity_score = model.wv.similarity("רע", "טוב")
    distance_score = model.wv.distance("סגירת", "פתיחת")
    similarity_score = model.wv.similarity("סגירת", "פתיחת")


    sentences_rep=["ברוכים הבאים , הכנסו בבקשה לחדר ."," אני מוכנה להאריך את ההסכם באותם תנאים.","בוקר טוב , אני פותח את הישיבה .","שלום , הערב התבשרנו שחברינו היקר לא ימשיך איתנו בשנה הבאה . "]
    replaced_senteces=replace_word_in_sentences(sentences_rep,model)
    save_file_red_words(sentences_rep,replaced_senteces,output_path)


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    main(args.input_path,args.output_path)
if __name__ == '__main__':
    arg()




