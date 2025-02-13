#!/usr/bin/env python3

import chromadb
import pandas as pd
from mistralai import Mistral
from chromadb.utils import embedding_functions

#The embedding function
emb_fn = embedding_functions.DefaultEmbeddingFunction()
#Loading data set
data = pd.read_csv("comments.csv")
#Merging comment and tag
comments = [f"""{content} : {tag} """ for content,tag in zip(data["content"], data["tag"])]

#Chromadb creation
chroma_client = chromadb.PersistentClient("./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="Comment",
    embedding_function=emb_fn,
    metadata={
        "hnsw:space":"cosine",
        "hnsw:search_ef":10
    })


collection.upsert(
    documents=comments,
    ids=data["id"].astype(str).tolist()
)

