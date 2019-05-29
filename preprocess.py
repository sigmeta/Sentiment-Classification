import pandas as pd
from bs4 import BeautifulSoup as bs
import re
import json
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_review(raw_review: str) -> str:
    # 1. Remove HTML
    review_text = bs(raw_review, "lxml").get_text()
    # 3. Convert to lower case
    lowercase_letters = review_text.lower()
    return lowercase_letters


# not used
def lemmatize(tokens: list) -> list:
    # 1. Lemmatize
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # 2. Remove stop words
    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return meaningful_words


def preprocess(review: str) -> list:
    # 1. Clean text
    review = clean_review(review)
    # 2. Split into individual words
    tokens = word_tokenize(review)
    # 3. Lemmatize
    #lemmas = lemmatize(tokens)
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return tokens

if __name__=='__main__':
    REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    train = []
    test = []
    mono = []
    df = pd.read_csv("data/labeledTrainData.tsv", delimiter='\t')
    for i in range(len(df)):
        txt=preprocess(df.loc[i, 'review'])
        train.append({'label': int(df.loc[i, 'sentiment']), 'text': txt})
        mono.append(txt)

    dftest = pd.read_csv("data/testData.tsv", delimiter='\t')
    for i in range(len(dftest)):
        txt=preprocess(dftest.loc[i, 'review'])
        test.append(txt)
        mono.append(txt)

    with open("data/unlabeledTrainData.tsv", encoding='utf8') as f:
        tlist=f.read().split('\n')
        for t in tlist[1:]:
            if len(t.split('\t'))==2:
                mono.append(preprocess(t.split('\t')[1]))
            else:
                print(t)

    with open("data/train.json", 'w') as f:
        f.write(json.dumps(train))
    with open("data/test.json", 'w') as f:
        f.write(json.dumps(test))
    with open("data/mono.json",'w') as f:
        f.write(json.dumps(mono))

