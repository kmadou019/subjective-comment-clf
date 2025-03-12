#!/usr/bin/env python3

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import ast

def compare_chunks(true_chunk, split_chunks):
   for i in range(min(len(true_chunk), len(split_chunks))):
    print("True: ", true_chunk[i])
    print("--------------------")
    print("from: ", split_chunks[i])
    print("#############")

    return 0 # metric

emb_fn = OllamaEmbeddings(model="mistral")
df = pd.read_csv("comment_chunk.csv")
df["chunks"] = df["chunks"].apply(ast.literal_eval)
comments = df["comment"]
true_chunks = df["chunks"]
metrics = []
for threshold in range(1, 100):
    text_splitter = SemanticChunker(
        emb_fn,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,)
    metric = 0
    for comment,true_chunk in zip(comments, true_chunks):
        comment_split = text_splitter.create_documents([comment])
        comment_split = [doc.page_content for doc in comment_split]
        metric += compare_chunks(true_chunk, comment_split)
    metrics.append(metric)

print(metrics)

