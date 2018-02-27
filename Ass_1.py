import numpy as np
import nltk
from collections import OrderedDict
from nltk.corpus import gutenberg
from nltk.corpus import brown
import operator
import math
import string
#files = gutenberg.fileids()
vocab = {}
train = []
test =[]
unknown = {}
pred_count_dict = {}
succ_count_dict = {}

    
def fetch_train_test(corpora, test_corpus):
    train = []
    test = []
    unknown.clear()
    vocab.clear()
    pred_count_dict.clear()
    succ_count_dict.clear()
    for corpus in corpora:
        if corpus == 'brown':
            files = brown.fileids()
        elif corpus == 'gutenberg':
            files = gutenberg.fileids()
        else:
            print("config Error")
        for file in files:
            if corpus == 'brown':
                sentences = brown.sents(file)
            elif corpus == 'gutenberg':
                sentences = gutenberg.sents(file)
            else:
                print("config Error")
            permute = np.ones(len(sentences))
            if corpus == test_corpus:
                permute[:int(len(sentences) * 0.2)] = 0
            np.random.shuffle(permute)
            for index in range(len(sentences)):
                if permute[index] == 0:
                    test.append(sentences[index])
                else:
                    train.append(sentences[index])
        return [train, test]
    
def remove_less_freq(dictionary):
   dictionary = {k: v for k,v in dictionary.items() if (v >= 2)} 
   return dictionary;
            
def less_freq(dictionary):
   dictionary = {k: v for k,v in dictionary.items() if (v <= 1)} 
   return dictionary;


def replace_unk():
    for sentence in train:
        for idx, item in enumerate(sentence):
            if item in unknown:
                sentence[idx] = 'unk'
    for sentence in test:
        for idx, item in enumerate(sentence):
            if item not in vocab:
                sentence[idx] = 'unk'
            
    

def buildNGram(sentences, N = 2):
    vocabulary = {}
    for sentence in sentences:
        tokens = nltk.word_tokenize("starttoken " +" ".join(sentence) + " stoptoken")
        for i in range(N, len(tokens)):
            token = ""
            for j in range(N, 0, -1):
                if j != N:
                    token = token + " " +tokens[i-j].lower()
                else:
                    token = token + tokens[i-j].lower()
            if token in vocabulary:
                vocabulary[token] += 1
            else:
                vocabulary[token] = 1
    return vocabulary;


def buildVocabulary(sentences):
    vocabulary = {}
    for sentence in sentences:
        tokens = nltk.word_tokenize("starttoken " +" ".join(sentence) + " stoptoken")
        for token in tokens:
            token = token.lower()
            if token in vocabulary:
                vocabulary[token] += 1
            else:
                vocabulary[token] = 1
    #For Sorting the Dictionary
    sorted_vocabulary = OrderedDict(sorted(vocabulary.items(), key=operator.itemgetter(1)))
    #print(type(sorted_vocabulary))
    return sorted_vocabulary;


def whichGram(num):
    if num == 1:
        return vocab
    elif num == 2:
        return biGram
    elif num == 3:
        return triGram
    #elif num == 4:
    #    return fourGram
    else:
        print("First build the model", num, " and approach me")

def count_succ(sentence, N):
    """
    return uniq count and also count
    """
    if sentence in succ_count_dict:
        return succ_count_dict[sentence]
    count = 0
    uniqcount = 0
    gram = whichGram(N + 1)
    for word in vocab:
        search = word + ' ' + sentence
        if search in gram:
            count += gram[search]
            uniqcount += 1
    succ_count_dict[sentence] = [uniqcount, count]
    return [uniqcount,  count]

def count_pred(sentence, N):
    """
    return uniq count and also count
    """
    if sentence in pred_count_dict:
        return pred_count_dict[sentence]
    count = 0
    uniqcount = 0
    gram = whichGram(N + 1)
    for word in vocab:
        search = sentence + ' ' + word
        if search in gram:
            count += gram[search]
            uniqcount += 1
    pred_count_dict[sentence] = [uniqcount, count]
    return [uniqcount,  count]


def kneser_nay_prob(sentence, N):
    if N <= 2:
        delta = 1
    else:
        delta = 0.75
    gram = whichGram(N)
    if sentence in gram:
        count = gram[sentence]
    else:
        count = 0
        
    tokens = nltk.word_tokenize(sentence)
    premise = ""
    for i in range(len(tokens) - 1):
        if i == 0:
            premise = premise + tokens[i].lower()
        else:
            premise = premise + " " + tokens[i].lower()
    
    hypothesis = tokens[len(tokens) - 1]
    
    premise_pred_count = count_pred(premise, N-1)
    hypo_succ_count = count_succ(hypothesis, 1)
    """
    if count == 0:
        return hypo_succ_count[0] / len(biGram)
    """
    term1 = max(count - delta, 0) / vocab[premise]
    
    lambda_term = delta * premise_pred_count[0] / premise_pred_count[1]
    return (term1 + (lambda_term * (hypo_succ_count[0] / len(biGram))))


def calculate_perplexity(N):
    perplexity = 0
    count = 0
    for sentence in test:
        count += 1
        prob = 0
        tokens = nltk.word_tokenize("starttoken " +" ".join(sentence) + " stoptoken")
        for i in range(N, len(tokens)):
            token = ""
            for j in range(N, 0, -1):
                if j != N:
                    token = token + " " + (tokens[i-j]).lower()
                else:
                    token = token + tokens[i-j].lower()
            try:
                prob = prob + math.log(kneser_nay_prob(token, N))
            except:
                prob = prob + math.log(1/len(vocab))
                #print("problem : ", token)
        if count%2000 == 0:
            print("Perplexity for ", count, " : ", perplexity/len(test))
        prob = -(prob)/len(tokens)
        perplexity += math.exp(prob)
    return perplexity/len(test)

def generate_sentence():
    sentence = ""
    count = 0
    candidate = ""
    used = {}
    for k,v in triGram.items():
        if v > count:
            ignore = False
            tokens = nltk.word_tokenize(k)
            for token in tokens:
                if not token.isalpha():
                    ignore = True
            if ignore == True:
                continue
            candidate = k
            #print(k, " : ", v)
            count = v
    used[candidate] = 1
    sentence = sentence + candidate
    #print("sentence : " , sentence)
    """
    begining three words are fixed
    """
    changed = True
    
    while True:
        if changed == False:
            break
        changed = False
        candidate = ""
        count = 0
        tokens = nltk.word_tokenize(sentence)
        pred = tokens[len(tokens) - 2] + " " +tokens[len(tokens) - 1]
        for word in vocab:
            check = pred + " " + word
            tokens = nltk.word_tokenize(check)
            """
            for token in tokens:
                if not token.isalpha():
                    ignore = True
            if ignore == True:
                continue
            """
            if check in used:
                continue
            if check in triGram:
                changed = True
                if count < triGram[check]:
                    count = triGram[check]
                    candidate = check
        if changed == True:
            tokens = nltk.word_tokenize(candidate)
            sentence = sentence + " " + tokens[len(tokens) - 1]
            used[candidate] = 1
        #print("sentence : " , sentence)
    final_sent = ""
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token == 'starttoken' or token == 'brownstoptoken':
            continue
        elif token == 'unk':
            final_sent = final_sent + " " + vocab[np.random.randint(0, len(vocab))]
        else:
            final_sent = final_sent + " " + token
    print("sentence is : ", final_sent)
    
##############################################
            
print('Configuration : train = Brown, Test = Brown')

train, test = fetch_train_test(['brown'], 'brown')
print("train : ", len(train), "test : ", len(test))
vocab = buildVocabulary(train)
infrequent_words = less_freq(vocab);
permute = np.ones(len(infrequent_words))
permute[:int(len(infrequent_words) * 0.03)] = 0

for i in range(int(len(infrequent_words) * 0.9)):
    infrequent_words.popitem()
"""
Just some emperical handling
"""
if 'a.m.' in vocab:
    infrequent_words['a.m.'] = 1
if 'p.m.' in vocab:
    infrequent_words['p.m.'] = 1

for k,v in infrequent_words.items():
    if k in infrequent_words:
        vocab.pop(k)
        unknown[k] = 1
vocab['unk'] = len(unknown)

replace_unk();

biGram = buildNGram(train, 2)
triGram = buildNGram(train, 3)

#perp = calculate_perplexity(2)
#print("perplexity is : ", perp) 
generate_sentence()



print("***********************************************************");
    
print('Configuration : train = Gutenberg, Test = Gutenberg')
train, test = fetch_train_test(['gutenberg'], 'gutenberg')
print("train : ", len(train), "test : ", len(test))
vocab = buildVocabulary(train)
infrequent_words = less_freq(vocab);
permute = np.ones(len(infrequent_words))
permute[:int(len(infrequent_words) * 0.03)] = 0

for i in range(int(len(infrequent_words) * 0.9)):
    infrequent_words.popitem()
"""
#Just some emperical handling
"""
if 'a.m.' in vocab:
    infrequent_words['a.m.'] = 1
if 'p.m.' in vocab:
    infrequent_words['p.m.'] = 1

for k,v in infrequent_words.items():
    if k in infrequent_words:
        vocab.pop(k)
        unknown[k] = 1
vocab['unk'] = len(unknown)

replace_unk();

biGram = buildNGram(train, 2)
triGram = buildNGram(train, 3)

#perp = calculate_perplexity(2)
#print("perplexity is : ", perp) 
generate_sentence()
print("***********************************************************");

print('Configuration : train = Brown + Gutenberg, Test = Brown')
train, test = fetch_train_test(['brown', 'gutenberg'], 'brown')
print("train : ", len(train), "test : ", len(test))
vocab = buildVocabulary(train)
infrequent_words = less_freq(vocab);
permute = np.ones(len(infrequent_words))
permute[:int(len(infrequent_words) * 0.03)] = 0

for i in range(int(len(infrequent_words) * 0.9)):
    infrequent_words.popitem()
"""
#Just some emperical handling
"""
if 'a.m.' in vocab:
    infrequent_words['a.m.'] = 1
if 'p.m.' in vocab:
    infrequent_words['p.m.'] = 1

for k,v in infrequent_words.items():
    if k in infrequent_words:
        vocab.pop(k)
        unknown[k] = 1
vocab['unk'] = len(unknown)

replace_unk();

biGram = buildNGram(train, 2)
triGram = buildNGram(train, 3)

#perp = calculate_perplexity(2)
#print("perplexity is : ", perp) 
generate_sentence()
print("***********************************************************");

print('Configuration : train = Brown + Gutenberg, Test = Gutenberg')
train, test = fetch_train_test(['brown', 'gutenberg'], 'gutenberg')
print("train : ", len(train), "test : ", len(test))
vocab = buildVocabulary(train)
infrequent_words = less_freq(vocab);
permute = np.ones(len(infrequent_words))
permute[:int(len(infrequent_words) * 0.03)] = 0

for i in range(int(len(infrequent_words) * 0.9)):
    infrequent_words.popitem()
"""
#Just some emperical handling
"""
if 'a.m.' in vocab:
    infrequent_words['a.m.'] = 1
if 'p.m.' in vocab:
    infrequent_words['p.m.'] = 1

for k,v in infrequent_words.items():
    if k in infrequent_words:
        vocab.pop(k)
        unknown[k] = 1
vocab['unk'] = len(unknown)

replace_unk();

biGram = buildNGram(train, 2)
triGram = buildNGram(train, 3)

#perp = calculate_perplexity(2)
#print("perplexity is : ", perp) 
generate_sentence()
print("***********************************************************");

