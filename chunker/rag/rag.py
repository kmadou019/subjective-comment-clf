#!/usr/bin/env python3
from graph_builder import graph 
from chunker.rag.graph_builder import graph as graphRAG
import pandas as pd
import ast
from Levenshtein import distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


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
    print(text)
    return ast.literal_eval(text)


def main():
    df = pd.read_csv("../data/comment_chunk_test.csv")
    df["chunks"] = df["chunks"].apply(ast.literal_eval)
    global_comments = df["comment"]
    true_chunks = df["chunks"]

    data_path = os.path.join(os.path.dirname(__file__), '../../rag/data', 'comments.csv')
    comments = pd.read_csv(data_path)

    sum_F1_s = 0
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(collection_name="Comment",persist_directory="../../rag/chroma_db" ,embedding_function=embeddings)
    for comment,true_chunk in zip(global_comments, true_chunks):
        comment_split = graph.invoke({"comment" : comment})["chunks"]
        for chunk in comment_split:
            #find chunk's categories
            #chunk_in_db = vector_store.similarity_search(chunk, k=1)
            #categorie = comments["tag"][int(chunk_in_db[0].id)]
            #print("chunk",chunk)
            #print("chunk in db",chunk_in_db)
            #print("cat",categorie)
            #classify all the chunks
            categorie_of_chunk = graphRAG.invoke({"comment": chunk, "llm" : "phi4"})["classe"]
            print(categorie_of_chunk)


        #print_chunks(true_chunk, comment_split)
        #sum_F1_s += compare_chunks(true_chunk, comment_split)

    print('F1-score for rag : ', sum_F1_s)

if __name__ == "__main__":
    main()


# F1-score for rag :  1811.7612554112548
