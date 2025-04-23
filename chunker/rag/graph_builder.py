#!/usr/bin/env python3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import pandas as pd
from langchain_ollama.llms import OllamaLLM
import json

model = "llama3.3"
llm = OllamaLLM(model=model)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(collection_name="Chunks", persist_directory="./chroma_db", embedding_function=embeddings)

true_chunks = pd.read_csv("../data/comment_chunk_train.csv", index_col="comment")

template = """
Tu es un expert dans l’analyse sémantique et le découpage de texte en unités de sens cohérentes et exploitables. Ton objectif est de segmenter un texte donné en morceaux courts, chacun représentant une unité de sens distincte.

Voici des exemples de texte et leur découpage en unités de sens :

{context}

En te basant sur ces exemples, découpe le texte suivant en unités de sens :

{comment}

Contraintes importantes :

Chaque unité doit exprimer une idée ou information complète et autonome."['La majorité de mes réponses sont correctes et dans un savoir certain ce qui est un point satisfaisant ', '. Cependant je dirai que je suis imprudent puisque j’ai fait 8 erreurs dangereuses. ', '']"

Les unités doivent être suffisamment courtes pour être utilisées individuellement dans un système de RAG.

Respecte rigoureusement le format de sortie suivant (JSON array) :

["unité de sens 1", "unité de sens 2", "unité de sens 3", ...]

Ne produis aucun autre texte en dehors de ce format de sortie.

"""



prompt = PromptTemplate(
    input_variables=["commentaire", "context"],
    template=template
)

class State(TypedDict):
    comment     : str
    context     : List[Document]
    chunks      : str
    true_chunks : List[str]
    
def retrieve(state : State):
    #Retrieve
    retrieved_docs = vector_store.similarity_search(state["comment"], k=2)
    return {"context": retrieved_docs}

def generate(state : State):
    context = ""
    for comment in state["context"]:
        context += comment.page_content + "---->" + true_chunks.loc[comment.page_content]
        context += "\n"
    
    formatted_prompt = prompt.invoke({"comment": state["comment"], "context": context})
    response = llm.invoke(formatted_prompt)
    print(response)
    return {"chunks" : json.loads(response), "true_chunks" : "Have to interact with test_set" }


# Compile application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

