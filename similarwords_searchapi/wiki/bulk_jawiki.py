# -*- coding: utf-8 -*-
import MeCab
import numpy as np
import json, gzip, datetime, sys, math
from elasticsearch import Elasticsearch, helpers
from gensim.models import KeyedVectors
es = Elasticsearch("http://elasticsearch:9200", request_timeout=100)
target_file = "/app/wiki/jawiki-20220516-cirrussearch-content.json.gz"
target_index = "jawiki-swem"
target_mapping = "/app/wiki/jawiki_swem_mapping.json"
target_setting = "/app/wiki/jawiki_swem_setting.json"
w2v_model_path = "/app/wiki/entity_vector.model.bin"
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)

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

def progress(current, pro_size):
    return print('\r making bulk data {0}% {1}/{2}'.format(
        math.floor(current / pro_size * 100.), 
        current, 
        pro_size), end='')

def convert_size(size, unit="B"):
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB")
    i = units.index(unit.upper())
    size = round(size / 1024 ** i, 2)
    return f"{size} {units[i]}"

def do_bulk_import(import_data, count):
    if len(import_data) > 0:
        size = sys.getsizeof(import_data)
        count += 1
        print('\r ****** bulk_import {} [{}] started at {} *****'.format(
            count, 
            convert_size(size, "KB"),
            datetime.datetime.now()))
        helpers.bulk(es, import_data)
        print('\r ****** bulk_import {} [{}]    done at {} *****'.format(
            count, 
            convert_size(size, "KB"), 
            datetime.datetime.now()))
    return count

def create_wiki_doc(curid, json_line):
    return {
        'curid': curid,
        'title': json_line['title'], 
        'text': json_line['text'],
        'text_vector': swem.average_pooling(json_line['text']), 
        'category': json_line['category'],
        'outgoing_link': json_line['outgoing_link'],
        'timestamp': json_line['timestamp']
    }

def open_cirrussearch_file(cirrussearch_file, index_name, bulk_articles_limit, import_limit):
    with gzip.open(cirrussearch_file) as f:
        data, curid, count, import_count = [], 0, 1, 0
        for line in f:
            if not line:
                import_count = do_bulk_import(data, import_count)
                data = []
                break
            else:
                json_line = json.loads(line)
                if "index" not in json_line:
                    progress(count, bulk_articles_limit)
                    data.append({'_index': index_name, '_source':create_wiki_doc(curid, json_line)})
                    if count % bulk_articles_limit == 0:
                        import_count = do_bulk_import(data, import_count)
                        data, count = [], 1
                        if import_limit > 0 and import_count >= import_limit:
                            break
                    else:
                        count += 1
                else:
                    curid = json_line['index']['_id']

def bulk_import_wiki(bulk_articles_limit=1000, import_limit=0):
    open_cirrussearch_file(target_file, target_index, bulk_articles_limit, import_limit)

def make_index():
    if es.indices.exists(index=target_index):
        es.indices.delete(index=target_index)
    
    with open (target_setting) as fs:
        setting = json.load(fs)
        with open(target_mapping) as fm:
            mapping = json.load(fm)
            es.indices.create(index=target_index, mappings=mapping, settings=setting)

def check_recreate_index():
    while True:
        inp = input('Re-create index[{}] before bulk import? [Y]es/[N]o? >> '.format(target_index)).lower()
        if inp in ('y', 'yes', 'n', 'no'):
            inp = inp.startswith('y')
            break
        print('Error! Input again.')
    return inp

def main():
    if check_recreate_index():
        make_index()
    bulk_import_wiki(1000, 10)
    
if __name__ == '__main__':
    main()
    es.close()