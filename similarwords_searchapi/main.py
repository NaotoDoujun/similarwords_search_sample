import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch

app = FastAPI()
w2v_model_path = "/app/wiki/entity_vector.model.bin"
print('loading w2v model file on memory...')
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
es = Elasticsearch("http://elasticsearch:9200", request_timeout=100)
target_index = "jawiki"

class Item(BaseModel):
    text: str

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)