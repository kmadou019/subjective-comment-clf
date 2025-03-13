#!/usr/bin/env python3

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import ast
from Levenshtein import distance
import numpy as np

def compare_chunks(true_chunks, split_chunks):
   
   for unity_ia in split_chunks:
    distances = []
    chunks_similarity = 0
    for unity_human in true_chunks:
        distances.append(distance(unity_ia, unity_human))   
    chunks_similarity += min(distances)
    
    return chunks_similarity

def print_chunks(true_chunk, comment_split):
    for i in range(min(len(true_chunk), len(comment_split))):
        print("True chunk: ", true_chunk[i])
        print("Comment split: ", comment_split[i])
        print("--------------------------------")
    print("#############################")
  


emb_fn = OllamaEmbeddings(model="mistral")
df = pd.read_csv("comment_chunk.csv")
df["chunks"] = df["chunks"].apply(ast.literal_eval)
comments = df["comment"]
true_chunks = df["chunks"]
metrics = []

threshold = 0.8
text_splitter = SemanticChunker(
    emb_fn,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=threshold,)

metric = 0
for comment,true_chunk in zip(comments, true_chunks):
    comment_split = text_splitter.create_documents([comment])
    comment_split = [doc.page_content for doc in comment_split]
    print_chunks(true_chunk, comment_split)
    metric += compare_chunks(true_chunk, comment_split)


#The best is 0.8
