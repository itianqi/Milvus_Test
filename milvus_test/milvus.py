import numpy as np
import random
from milvus import Milvus
from milvus import Status
from milvus import DataType

_HOST = '192.168.xx.xx'
_PORT = 19530

milvus = Milvus(_HOST, _PORT)

_DIM = 8  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index

collection_name = 'example_collection'
partition_tag = 'demo_tag'
segment_name= ''

ivf_param = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 4096}}
milvus.create_index(collection_name, "embedding", ivf_param)

vectors = [[random.random() for _ in range(_DIM)] for _ in range(10)]
ids = [i for i in range(10)]

collection_param = {
    "fields": [
        #  Milvus doesn't support string type now, but we are considering supporting it soon.
        #  {"name": "title", "type": DataType.STRING},
        {"name": "duration", "type": DataType.INT32, "params": {"unit": "minute"}},
        {"name": "release_year", "type": DataType.INT32},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8}},
    ],
    "segment_row_limit": 4096,
    "auto_id": False
}

The_Lord_of_the_Rings = [
    {
        "title": "The_Fellowship_of_the_Ring",
        "id": 1,
        "duration": 208,
        "release_year": 2001,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "The_Two_Towers",
        "id": 2,
        "duration": 226,
        "release_year": 2002,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "title": "The_Return_of_the_King",
        "id": 3,
        "duration": 252,
        "release_year": 2003,
        "embedding": [random.random() for _ in range(8)]
    }
]

ids = [k.get("id") for k in The_Lord_of_the_Rings]
durations = [k.get("duration") for k in The_Lord_of_the_Rings]
release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

hybrid_entities = [
    # Milvus doesn't support string type yet, so we cannot insert "title".
    {"name": "duration", "values": durations, "type": DataType.INT32},
    {"name": "release_year", "values": release_years, "type": DataType.INT32},
    {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
]

for _ in range(2000):
    ids = milvus.insert(collection_name, hybrid_entities, ids, partition_tag="American")
    print("\n----------insert----------")
    print("Films are inserted and the ids are: {}".format(ids))

milvus.flush([collection_name])


films = milvus.get_entity_by_id(collection_name, ids=[1, 200])
for film in films:
    if film is not None:
        print(" > id: {},\n > duration: {}m,\n > release_years: {},\n > embedding: {}"
              .format(film.id, film.duration, film.release_year, film.embedding))

query_embedding = [random.random() for _ in range(8)]
query_hybrid = {
    "bool": {
        "must": [
            {
                "term": {"release_year": [2002, 2003]}
            },
            {
                # "GT" for greater than
                "range": {"duration": {"GT": 250}}
            },
            {
                "vector": {
                    "embedding": {"topk": 3, "query": [query_embedding], "metric_type": "L2"}
                }
            }
        ]
    }
}

results = milvus.search(collection_name, query_hybrid, fields=["duration", "release_year", "embedding"])
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("- id: {}".format(topk_film.id))
        print("- distance: {}".format(topk_film.distance))

        print("- release_year: {}".format(current_entity.release_year))
        print("- duration: {}".format(current_entity.duration))
        print("- embedding: {}".format(current_entity.embedding))






