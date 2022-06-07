# similarwords_search_sample

## download w2v model
Go http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/ and download 20170201.tar.bz2  
after done, decompress it n put 'entity_vector.model.bin' under 'similarwords_searchapi/wiki'

## Run containers
```bash
docker-compose up -d --build
```

## create jawiki index and import articles
Go https://dumps.wikimedia.org/other/cirrussearch/ and download content.json.gz  
I used jawiki-20220516-cirrussearch-content.json.gz. after done, put it under 'similarwords_searchapi/wiki'  
then, Run below command in similarwords_searchapi container
```python
python3 /app/wiki/bulk_jawiki.py
```

## Swagger screen
Open http://localhost:8000/docs