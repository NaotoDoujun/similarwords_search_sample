import MeCab
import numpy as np
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from typing import Dict
from pydantic import BaseModel
from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch
import slackweb

app = FastAPI()
logger = logging.getLogger('uvicorn')
w2v_model_path = "/app/wiki/entity_vector.model.bin"
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
es = Elasticsearch("http://elasticsearch:9200", request_timeout=100)
target_index = "jawiki-swem"
similarwordbot_url = "http://mattermost:8065/hooks/uysbwqyteirxpy9co96mwk1tty"
recommendbot_url = "http://mattermost:8065/hooks/xsugb5pke7yfbg8jf88hx4ieua"
max_size = 2

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
    try:
        similars = w2v_model.most_similar(u'[{}]'.format(item.text))
        if len(similars) > 0:
            similarwords = [
                row[0].strip('[').strip(']')
                for row in similars
            ]

            if item.text in similarwords:
                similarwords.remove(item.text)

            unique_similarwords = []
            for similarword in similarwords:
                if similarword not in unique_similarwords:
                    unique_similarwords.append(similarword)

            if len(unique_similarwords) > 0:
                querywords = unique_similarwords[0:max_size] if len(unique_similarwords) > 1 else unique_similarwords[0:1]
                querywords.insert(0, item.text)
                responses = [
                {
                    "query": word,
                    "responses": es.search(
                        index=target_index,
                        size=max_size,
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

                if len(results) > 0:
                    try:
                        attachments = [
                            {
                            "mrkdwn_in": ["text"],
                                "color": "#36a64f",
                                "pretext": "",
                                "author_name": "similarwordbot",
                                "title": "Similar Words Search Result",
                                "text": "search word [{}]".format(item.text),
                                "fields": [
                                    {
                                        "title": "{} (query:{} score:{})".format(row['title'], row['query'], row['score']),
                                        "value": row['text'],
                                        "short": "false"
                                    }
                                    for row in results
                                ]
                            }
                        ]
                        bot = slackweb.Slack(url=similarwordbot_url)
                        bot.notify(attachments=attachments)
                    except:
                        logger.warning("Similarword Bot notify error. Check url.")
                        pass

                return results
            else:
                raise HTTPException(status_code=404, detail="Similar words not found. (empty unique_similarwords)")
        else:
            raise HTTPException(status_code=404, detail="Similar words not found. (empty similars)")
    except:
        raise HTTPException(status_code=404, detail="Similar words not found. (except)")

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
        size=max_size,
        query=script_query
    )

    results = [
        {
            'title': row['_source']['title'], 
            'text': row['_source']['text'], 
            'score': row['_score'],
        }
        for row in response['hits']['hits']
    ]

    if len(results) > 0:
        try:
            attachments = [
                {
                "mrkdwn_in": ["text"],
                    "color": "#36a64f",
                    "pretext": "",
                    "author_name": "recommendbot",
                    "title": "Recommends from Wikipedia",
                    "text": "{} said [{}]".format(data['user_name'], data['text']),
                    "fields": [
                        {
                            "title": "{} (score:{})".format(row['title'], row['score']),
                            "value": row['text'],
                            "short": "false"
                        }
                        for row in results
                    ]
                }
            ]
            bot = slackweb.Slack(url=recommendbot_url)
            bot.notify(attachments=attachments)
        except:
            logger.warning("Recommend Bot notify error. Check url.")
            pass

        return results
    else:
        raise HTTPException(status_code=404, detail="Recommends not found. (empty results)")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)