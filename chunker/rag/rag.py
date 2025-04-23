#!/usr/bin/env python3
from graph_builder import graph 
import pandas as pd
import ast
from Levenshtein import distance

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
  

def main():
    df = pd.read_csv("../data/comment_chunk_test.csv")
    df["chunks"] = df["chunks"].apply(ast.literal_eval)
    comments = df["comment"]
    true_chunks = df["chunks"]

    sum_F1_s = 0
    for comment,true_chunk in zip(comments, true_chunks):
        comment_split = graph.invoke({"comment" : comment})["chunks"]
        print_chunks(true_chunk, comment_split)
        sum_F1_s += compare_chunks(true_chunk, comment_split)

    print('F1-score for rag : ', sum_F1_s)

if __name__ == "__main__":
    main()


# F1-score for rag :  1811.7612554112548
