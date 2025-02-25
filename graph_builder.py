#!/usr/bin/env python3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import pandas as pd
from langchain_ollama.llms import OllamaLLM

# Define the LLM
model = input("Enter the model name: ")
llm = OllamaLLM(model=model)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(collection_name="Comment",persist_directory="./chroma_db" ,embedding_function=embeddings)
#Load the dataset
comments = pd.read_csv("data/comments.csv")
# Define prompt for classification
template = """
Tu es un expert en classification de commentaires.

Tu disposes des classes suivantes, chacune avec sa définition et une liste de mots-clés pertinents pouvant faciliter leur identification :

1.MonCal.Interpret : Interprétations des résultats du test allant au-delà des observations factuelles, reflétant les interprétations subjectives des étudiants sur leur performance.

2.MonCal.Facts.RF : Observations factuelles sur les résultats du test, incluant des remarques sur le nombre/quantité de réponses correctes et incorrectes.

3.MonCal.Facts.DC : Observations factuelles sur les savoirs fragiles, savoirs certains, erreurs dangereuses et erreurs présumées.

4.BDS.Emotions : Commentaires exprimant les émotions ressenties par l’étudiant pendant et après le test.

5.MotOrient.Value : Commentaires sur l'intérêt des étudiants pour les types de tests et leur valeur perçue pour l’apprentissage.

6.MotOrient.SelfEff : Commentaires sur la confiance et l’auto-efficacité des étudiants.

7.DomKldg : Commentaires identifiant les concepts/disciplines manquants et leur degré d’acquisition.

8.StratKldg : Commentaires sur les stratégies d’apprentissage utilisées pendant le test et l’analyse du comportement des étudiants.

9.CTRL : Commentaires sur les comportements futurs des apprenants.

En te basant sur les commentaires suivants, classifiés par un expert humain sous le format (commentaire : classe) :

{context}

Classifie ce commentaire :

{comment}

Format de sortie (sans explication ni caractères spéciaux comme "<,>,',...") :
<classe>

"""

prompt = PromptTemplate(
    input_variables=["commentaire", "context"],
    template=template
)

# Define state for application
class State(TypedDict):
    comment       : str
    context       : List[Document]
    initialClass  : str
    predictedClass: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["comment"], k=10)
    return {"context": retrieved_docs}


def generate(state: State):
    #To have something like : 'je me sens nul ':BDS.Emotions 
    context = "\n".join(f""" '{ctx.page_content}':{comments["tag"][int(ctx.id)]} """ for ctx in state["context"])
    formatted_prompt = prompt.invoke({"comment": state["comment"], "context": context})
    response = llm.invoke(formatted_prompt)
    response = ''.join(c for c in response.strip() if c not in "<>'")
    return {"predictedClass": response}


# Compile application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
