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

Mots-clés : "prudent", "imprudent", "doute", etc.

2.MonCal.Facts.RF : Observations factuelles sur les résultats du test, incluant des remarques sur le nombre/quantité de réponses correctes et incorrectes.

Mots-clés : "correcte", "incorrecte", "ignorance", "nombres", "juste", "faux".

3.MonCal.Facts.DC : Observations factuelles sur les savoirs fragiles, savoirs certains, erreurs dangereuses et erreurs présumées.

Mots-clés : "savoirs fragiles", "savoirs certains", "erreurs dangereuses", "nombre", "erreurs présumées", "courbe en J".

4.BDS.Emotions : Commentaires exprimant les émotions ressenties par l’étudiant pendant et après le test.

Mots-clés : "stress", "anxiété", "peur", "déception", "satisfaction", "fierté", "fatigue".

5.MotOrient.Value : Commentaires sur l'intérêt des étudiants pour les types de tests et leur valeur perçue pour l’apprentissage.

Mots-clés : "permet", "évaluation", "utilité", "test".

6.MotOrient.SelfEff : Commentaires sur la confiance et l’auto-efficacité des étudiants.

Mots-clés : "confiance", "je suis sûr de moi", "mes compétences", "mes connaissances".

7.DomKldg : Commentaires identifiant les concepts/disciplines manquants et leur degré d’acquisition.

Mots-clés : "lacunes", "domaines disciplinaires", "concepts", "améliorer", "apprendre", "revoir", "comprendre", "chapitres", "leçon", "cours", "matières", "physique-chimie".

8.StratKldg : Commentaires sur les stratégies d’apprentissage utilisées pendant le test et l’analyse du comportement des étudiants.

Mots-clés : "oublis", "mauvaise lecture", "dû à", "avoir du mal", "loupé", "temps", "trop rapide", "attention", "inattention".

9.CTRL : Commentaires sur les comportements futurs des apprenants.

Mots-clés : verbes au futur, "prochain", "prochain test", "devenir", "devrais", "il faut que", "être plus comme ça...".

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
