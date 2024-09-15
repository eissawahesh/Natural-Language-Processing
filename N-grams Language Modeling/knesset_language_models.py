import math

import pandas as pd

class M_Trigram:
    def __init__(self,protocol_type):
        df = pd.read_csv("example_knesset_corpus.csv")
        if protocol_type=="committee":
            committee_df = df[df["protocol_type"] == "committee"]
            self.sentences = committee_df["sentence_text"].tolist()

        else:
            plenary_df = df[df["protocol_type"] == "plenary"]
            self.sentences = plenary_df["sentence_text"].tolist()
        self.unigrams={}
        self.bigrams={}
        self.trigrams={}
        self.collocations={}
    def pre_process(self):
        for sentence in self.sentences:
            tokens=sentence.split()

            for i in range(len(tokens)):
                if(tokens[i]in self.unigrams):
                    self.unigrams[tokens[i]]+=1
                else:
                    self.unigrams[tokens[i]] = 1

            for i in range(len(tokens)-1):
                bigram=' '.join([tokens[i],tokens[i+1]])
                if (bigram in self.bigrams):
                    self.bigrams[bigram] += 1
                else:
                    self.bigrams[bigram] = 1

            for i in range(len(tokens)-2):
                trigram=' '.join([tokens[i],tokens[i+1],tokens[i+2]])
                if (trigram in self.trigrams):
                    self.trigrams[trigram] += 1
                else:
                    self.trigrams[trigram] = 1
    def calculate_prob_of_sentence(self,sentence,smoothing_type):
        if(smoothing_type=="Linear"):
            return self.calculate_prob_of_sentence_linear(sentence)
        if(smoothing_type=="Laplace"):
            return self.calculate_prob_of_sentence_laplace(sentence)


    def calculate_prob_of_sentence_laplace(self,sentence):
        log_prob=0
        v=len(self.unigrams)
        tokens=sentence.split()
        for i in range(len(tokens)-2):
            trigram=' '.join([tokens[i],tokens[i+1],tokens[i+2]])
            bigram=' '.join([tokens[i],tokens[i+1]])
            trigram_count = self.trigrams.get(trigram, 0)
            bigram_count = self.bigrams.get(bigram, 0)
            laplace_prob=math.log2((trigram_count+1)/(bigram_count+v))
            log_prob+=laplace_prob
        return float("{:.3f}".format(log_prob))

    def calculate_prob_of_sentence_linear(self,sentence):
        linear_prob=0
        v = len(self.unigrams)
        tokens_count=sum(self.unigrams.values())
        tokens=sentence.split()
        lamda1,lamda2,lamda3=0.7,0.2,0.1
        for i in range(len(tokens)-2):


            unigram=tokens[i+2]
            unigram_count = self.unigrams.get(unigram, 0)
            unigram_prob=(unigram_count+1)/(tokens_count+v)


            bigram=' '.join([tokens[i+1],tokens[i+2]])
            unigram=tokens[i+1]
            bigram_count=self.bigrams.get(bigram, 0)
            bigram_prob=0
            if(bigram_count!=0):
                unigram_count=self.unigrams.get(unigram,0)
                bigram_prob=(bigram_count)/(unigram_count)


            trigram=' '.join([tokens[i],tokens[i+1],tokens[i+2]])
            bigram=' '.join([tokens[i],tokens[i+1]])
            trigram_count=self.trigrams.get(trigram,0)
            trigram_prob=0
            if trigram_count!=0:
                bigram_count = self.bigrams.get(bigram, 0)
                trigram_prob=(trigram_count)/(bigram_count)


            linear_prob+=math.log2(lamda1*trigram_prob+lamda2*bigram_prob+lamda3*unigram_prob)
        print("sentnce probability :",float("{:.3f}".format(linear_prob)))
        return float("{:.3f}".format(linear_prob))
    def generate_next_token(self,sequence):
        tokens = sequence.split()
        max_prob=float("-inf")
        best_token=None

        for token in self.unigrams.keys():
            seq=' '.join([tokens[-2],tokens[-1],token])
            prob=self.calculate_prob_of_sentence(seq,"Linear")

            if(prob>max_prob):
                max_prob=prob
                best_token=token
        print("The best token:",best_token)

        return best_token

def filling_words(list_of_sentences,protocol_type):
    sentences = list_of_sentences[:]
    missing_words=[]
    for i in range(len(sentences)):
        tokens=[]
        while "[*]" in sentences[i]:
            parts = sentences[i].split("[*]", 1)
            seq = parts[0].strip()
            if(protocol_type=="committee"):
                missing_word = committee_model.generate_next_token(seq)
            elif(protocol_type=="plenary"):
                missing_word = pleanary_model.generate_next_token(seq)
            parts[0] = seq + " " + missing_word
            sentences[i] = ''.join(parts)
            tokens.append(missing_word)
        sentences[i]=sentences[i][8:]
        missing_words.append(tokens)
    return sentences ,missing_words
def calc_prob(sentences):
    pleanary_prob=[]
    committee_prob = []
    for sentence in sentences:
        pleanary_prob.append(pleanary_model.calculate_prob_of_sentence(sentence,"Linear"))
        committee_prob.append(committee_model.calculate_prob_of_sentence(sentence,"Linear"))
    return pleanary_prob,committee_prob

pleanary_model=M_Trigram("plenary")
pleanary_model.pre_process()
committee_model=M_Trigram("committee")
committee_model.pre_process()

sentences=[]
with open("masked_sentences.txt","r",encoding="utf-8") as file:
    for sentence in file:
        sentences.append("<s> <s> "+sentence.strip())



plenary_sentences,plenary_missing_words=filling_words(sentences,"plenary")
committee_sentences,committee_missing_words=filling_words(sentences,"committee")

plenary_sentences_pleanary_prob,plenary_sentences_committee_prob=calc_prob(plenary_sentences)
committee_sentences_pleanary_prob,committee_sentences_committee_prob=calc_prob(committee_sentences)

for i in range(len(sentences)):
    sentences[i]=sentences[i][8:]
with open('sentences_results.txt', 'w', encoding='utf-8') as file:
    for i in range(len(sentences)):
        file.write("Original sentence: "+str(sentences[i])+"\n")
        file.write("Committee sentence: "+str(committee_sentences[i])+"\n")
        file.write("Committee tokens: "+str(committee_missing_words[i])+"\n")
        file.write("Probability of committee sentence in committee corpus: "+str(committee_sentences_committee_prob[i])+"\n")
        file.write("Probability of committee sentence in plenary corpus: "+str(committee_sentences_pleanary_prob[i])+"\n")
        if(committee_sentences_committee_prob[i]>committee_sentences_pleanary_prob[i]):
            file.write("This sentence is more likely to appear in corpus: committee\n")
        else:
            file.write("This sentence is more likely to appear in corpus: plenary\n")
        file.write("Plenary sentence: "+str(plenary_sentences[i])+"\n")
        file.write("Plenary tokens: "+str(plenary_missing_words[i])+"\n")
        file.write("Probability of plenary sentence in plenary corpus: "+str(plenary_sentences_pleanary_prob[i])+"\n")
        file.write("Probability of plenary sentence in committee corpus: "+str(plenary_sentences_committee_prob[i])+"\n")
        if(plenary_sentences_pleanary_prob[i]>plenary_sentences_committee_prob[i]):
            file.write("This sentence is more likely to appear in corpus: plenary\n")
        else:
            file.write("This sentence is more likely to appear in corpus: committee\n")
        file.write("\n")
def get_k_n_collocations(k ,n, protocol_type):
    df = pd.read_csv("example_knesset_corpus.csv")
    if(protocol_type == "committee"):
        committee_df = df[df["protocol_type"] == "committee"]
        sentences = committee_df["sentence_text"].tolist()
    if(protocol_type == "plenary"):
        plenary_df = df[df["protocol_type"] == "plenary"]
        sentences = plenary_df["sentence_text"].tolist()

    unigrams={}
    pmi_dict={}
    collocations={}
    for sentence in sentences:
        tokens =  sentence.split()

        for i in range(len(tokens)):
            if (tokens[i] in unigrams):
                unigrams[tokens[i]] += 1
            else:
                unigrams[tokens[i]] = 1


    N = sum(unigrams.values())


    for sentence in sentences:
        ngrams =[]
        tokens =sentence.split()
        for i in range(len(tokens ) - n +1):
            ngrams.append(' '.join(tokens[i: i +n]))
        for ngram in ngrams:
            if ngram in collocations:
                collocations[ngram] += 1
            else:
                collocations[ngram] = 1


    N_collactions=sum(collocations.values())
    for key ,value in collocations.items():
        tokens = key.split()
        prob2 = 1
        for token in tokens:
            prob2 *= (unigrams[token] / N)
        pmi_dict[key] = math.log2(((value /N_collactions)) / prob2)

    sorted_collocations = dict(sorted(pmi_dict.items(), key=lambda item: item[1], reverse=True))
    k_collocations = dict(list(sorted_collocations.items())[:k])
    return k_collocations
with open('knesset_collocations.txt', 'w', encoding='utf-8') as file:
    k_dict = {}

    for i in range(2,5):
        if (i == 2):
            file.write("Two-gram collocations:\n")
        elif (i == 3):
            file.write("Three-gram collocations:\n")
        elif (i == 4):
            file.write("Four-gram collocation:\n")
        k_dict=get_k_n_collocations(10,i,"committee")
        file.write("Committee corpus:\n")
        for key in k_dict.keys():
            file.write(key+"\n")
        file.write("\n")
        k_dict=get_k_n_collocations(10, i, "plenary")
        file.write("Plenary corpus:\n")
        for key in k_dict.keys():
            file.write(key+"\n")
        file.write("\n")



