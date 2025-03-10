#!/usr/bin/env python3

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
import os

emb_fn = OllamaEmbeddings(model="mistral")

with open("./comment_glob.txt") as f:
    text = f.read()

text_splitter = SemanticChunker(
    emb_fn,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=1,)

split_ = text_splitter.create_documents([text])

for doc in split_:
    print(doc)
    print("\n")


