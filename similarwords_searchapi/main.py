import MeCab
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Dict
from pydantic import BaseModel
from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch

app = FastAPI()
w2v_model_path = "/app/wiki/entity_vector.model.bin"
print('loading w2v model file on memory...')
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
es = Elasticsearch("http://elasticsearch:9200", request_timeout=100)
target_index = "jawiki-swem"

class Item(BaseModel):
    text: str

class MeCabTokenizer():
    def __init__(self, mecab_args=""):
        self.tagger = MeCab.Tagger(mecab_args)
        self.tagger.parse("")

    def tokenize(self, text):
        return self.tagger.parse(text).strip().split(" ")

class SWEM():
    """
    Simple Word-Embeddingbased Models (SWEM)
    https://arxiv.org/abs/1805.09843v1
    """

    def __init__(self, w2v, tokenizer, oov_initialize_range=(-0.01, 0.01)):
        self.w2v = w2v
        self.tokenizer = tokenizer
        self.vocab = set(self.w2v.vocab.keys())
        self.embedding_dim = self.w2v.vector_size
        self.oov_initialize_range = oov_initialize_range

        if self.oov_initialize_range[0] > self.oov_initialize_range[1]:
            raise ValueError("Specify valid initialize range: "
                             f"[{self.oov_initialize_range[0]}, {self.oov_initialize_range[1]}]")

    def get_word_embeddings(self, text):
        np.random.seed(abs(hash(text)) % (10 ** 8))

        vectors = []
        for word in self.tokenizer.tokenize(text):
            if word in self.vocab:
                vectors.append(self.w2v[word])
            else:
                vectors.append(np.random.uniform(self.oov_initialize_range[0],
                                                 self.oov_initialize_range[1],
                                                 self.embedding_dim))
        return np.array(vectors)

    def average_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]

    def hierarchical_pooling(self, text, n):
        word_embeddings = self.get_word_embeddings(text)

        text_len = word_embeddings.shape[0]
        if n > text_len:
            raise ValueError(f"window size must be less than text length / window_size:{n} text_length:{text_len}")
        window_average_pooling_vec = [np.mean(word_embeddings[i:i + n], axis=0) for i in range(text_len - n + 1)]

        return np.max(window_average_pooling_vec, axis=0)

tokenizer = MeCabTokenizer("-Owakati")
swem = SWEM(w2v_model, tokenizer)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/similarwords/")
def search_similarwords(item: Item):
    similars = w2v_model.most_similar(u'[{}]'.format(item.text))
    if len(similars) > 0:
        similarwords = [
            row[0].strip('[').strip(']')
            for row in similars
        ]
        querywords = similarwords[0:2] if len(similars) > 1 else similarwords[0:1]
        querywords.insert(0, item.text)
        responses = [
        {
            "query": word,
            "responses": es.search(
                index=target_index,
                size=2,
                query={
                    "multi_match": {
                        "fields": [ "title", "text" ],
                        "query": word
                    }
                }
            )
        }
        for word in querywords
        ]

        results = [
        {
            'query': response['query'],
            'title': row['_source']['title'], 
            'text': row['_source']['text'], 
            'score': row['_score'],
        }
        for response in responses
            for row in response['responses']['hits']['hits']
        ]

        return results
    else:
        return {"Result": "nothing similar words..."}

@app.post("/recommends/")
def propose_recommend(slackPost: Dict):
    data = jsonable_encoder(slackPost)
    query_vector = swem.average_pooling(data['text']).tolist()
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    response = es.search(
        index=target_index,
        size=2,
        query=script_query
    )

    result = [
        {
            'title': row['_source']['title'], 
            'text': row['_source']['text'], 
            'score': row['_score'],
        }
        for row in response['hits']['hits']
    ]
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)