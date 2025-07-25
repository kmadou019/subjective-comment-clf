#!/usr/bin/env python3
from .graph_builder import graph as graphSplit 
from rag.graph_builder import graph as graphRagClf
import pandas as pd
import ast
from Levenshtein import distance
import os
import re
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


def compare_chunks(true_chunks, split_chunks):
    if (len(true_chunks) + len(split_chunks) != 0):
        for unity_ia in split_chunks:
            distances = []
            chunks_similarity = 0
            for unity_human in true_chunks:
                distances.append(distance(unity_ia, unity_human))   
            chunks_similarity += min(distances)
        
        recall = chunks_similarity/len(true_chunks)
        precision = chunks_similarity / len(split_chunks)

        return 2 * (precision*recall)/(precision+recall) if precision + recall != 0 else 0

def print_chunks(true_chunk, comment_split):
    for i in range(min(len(true_chunk), len(comment_split))):
        print("True chunk: ", true_chunk[i])
        print("Comment split: ", comment_split[i])
        print("--------------------------------")
    print("#############################")
  
def extract_json(text):
    start = text.find("{")
    end = text.find("}")
    text = text[start:end+1].replace("true",'True').replace("false",'False').replace('"True"',"True").replace('"False"',"False")
    return ast.literal_eval(text)

def compute_score(original, founded):
    len_original, len_founded = len(original), len(founded) 
    correct, miss = 0,0
    print(original)
    print(founded)
    while founded:
        f = founded[0]
        if f in original:
            correct += 1
        else:
            miss += 1
        founded = [item for item in founded if item != f]
    print(f"{correct}/{len_original} ; {miss}/{len_founded}")
    return (correct, miss)

def main():
    df_path = os.path.join(os.path.dirname(__file__), '../data/', 'comment_chunk_test.csv')
    df = pd.read_csv(df_path)
    df["chunks"] = df["chunks"].apply(ast.literal_eval)
    global_comments = df["comment"]

    df_path = os.path.join(os.path.dirname(__file__), '../../rag/data', 'comments.csv')
    ####Repair here
    original_comment_path = df_path = os.path.join(os.path.dirname(__file__), '../data', 'real.csv')
    original_comment = pd.read_csv(original_comment_path)["real"]
    
    emb_fn = OllamaEmbeddings(model="mistral")
    text_splitter = SemanticChunker(
        emb_fn,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=8,)
    correct,miss = 1,1
    for comment,original in zip(global_comments, original_comment):
        comment_split = graphSplit.invoke({"comment" : comment})["chunks"]
        #comment_split = text_splitter.create_documents([comment])
        #comment_split = [doc.page_content for doc in comment_split]

        classes = []
        for chunk in comment_split:
            #classify all the chunks
            response = graphRagClf.invoke({"comment": chunk, "llm" : "phi4"})
            classe = extract_json(response["predictedClass"])["classe"]
            classes.append(classe)

        original_classes = re.findall(r'\[([^\[\]]+)\]', original)
        c, m = compute_score(original_classes, classes)
        correct += c
        miss    += m
    print("====> ",correct,";", miss)


if __name__ == "__main__":
    main()


# F1-score for rag :  1811.7612554112548
