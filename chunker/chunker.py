#!/usr/bin/env python3

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import ast
emb_fn = OllamaEmbeddings(model="mistral")

true_chunks = pd.read_csv("comment_chunk.csv")
true_chunks["chunks"] = true_chunks["chunks"].apply(ast.literal_eval)
text = true_chunks['chunks'][0]

text_splitter = SemanticChunker(
    emb_fn,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=1,)

split_ = text_splitter.create_documents([text])

for doc in split_:
    print(doc)
    print("\n")