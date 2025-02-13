#!/usr/bin/env python3

import chromadb
import pandas as pd
from mistralai import Mistral

API_KEY = "YDjpRuQdjQDiE2z0hjBfn0G65TYRg80i"

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings.create(model=model, inputs=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

#Loading data set
data = pd.read_csv("comments.csv")
#Merging comment and tag
comments = [f"""{content} : {tag} """ for content,tag in zip(data["content"], data["tag"])]

#Getting embedding of each comment
model = "mistral-embed"
client = Mistral(api_key=API_KEY)
embeddings = get_embeddings_by_chunks(comments, 50)

#Chromadb creation

chroma_client = chromadb.PersistentClient("./chroma_db")

collection = chroma_client.get_or_create_collection(name="Comment")
collection.add(
    ids=data["id"],
    embeddings=embeddings
)

