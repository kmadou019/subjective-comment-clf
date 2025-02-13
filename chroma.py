#!/usr/bin/env python3

import chromadb
import pandas as pd
from mistralai import Mistral
from chromadb.utils import embedding_functions

API_KEY = "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"
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#Loading data set
data = pd.read_csv("comments.csv")
#Merging comment and tag
comments = [f"""{content} : {tag} """ for content,tag in zip(data["content"], data["tag"])]

#Getting embedding of each comment
embeddings = huggingface_ef(comments[1])

#Chromadb creation

chroma_client = chromadb.PersistentClient("./chroma_db")

collection = chroma_client.get_or_create_collection(name="Comment")
collection.add(
    ids=data["id"],
    embeddings=embeddings
)

