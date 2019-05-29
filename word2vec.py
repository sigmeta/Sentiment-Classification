from gensim.models import Word2Vec
import json
import multiprocessing

corpus_path="data/mono.json"
if __name__ == '__main__':
    with open(corpus_path) as f:
        corpus=json.loads(f.read())
    model = Word2Vec(corpus, size=128, window=5, min_count=3, workers=multiprocessing.cpu_count(), iter=10)
    model.save("data/word2vec.model")
    model.wv.most_similar('galaxy')
