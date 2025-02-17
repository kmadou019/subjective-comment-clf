#!/usr/bin/env python3

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import os

#The embedding function
emb_fn = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#Loading data set
data = pd.read_csv("data/comments.csv")

#Chromadb creation
chroma_client = chromadb.PersistentClient("./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="Comment",
    embedding_function=emb_fn,
    metadata={
        "hnsw:space":"cosine",
        "hnsw:search_ef":10
    })

batch_size = 100
ids = data["id"].astype(str).tolist()
for i in range(0, len(data["content"]), batch_size):
    batch_comments = data["content"][i:i+batch_size]
    batch_ids      = ids[i:i+batch_size]

    collection.upsert(
        documents=batch_comments,
        ids=batch_ids
    )
