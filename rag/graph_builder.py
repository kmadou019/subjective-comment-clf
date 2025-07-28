#!/usr/bin/env python3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import pandas as pd
from langchain_ollama.llms import OllamaLLM
import os

# Initialize sentence-transformer embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup Chroma vector store path and instantiate vector store
chroma_path = os.path.join(os.path.dirname(__file__), '', 'chroma_db')
vector_store = Chroma(collection_name="Comment", persist_directory=chroma_path, embedding_function=embeddings)

# Load comments dataset with class labels
data_path = os.path.join(os.path.dirname(__file__), 'data', 'comments.csv')
comments = pd.read_csv(data_path)

# Define classification prompt template for the LLM, including classes, keywords, and output format
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

Tu peux donner deux classes si tu estimes que le commentaire peut correspondre à deux ou plusieurs classes. Tu donnes dans ce cas les deux classes que tu penses les plus pertinantes.

Faudra également que pour chaque classe tu donnes ton dégré de certitude par rapport à cette classe et cela en pourcentage.
Et aussi la somme de tes certitudes doit faire 100.

Format de sortie :
N'OUBLIE PAS LES GUILLEMETS DANS LES LABELS DES CLASSES (e.i : ["classe1"]).
{{
    "classe": ['classe1', 'classe2'],
    "certitude": [degré de certitude pour classe1, degré de certitude pour classe2],  
    "justification": "..."
}}

(Il faut une justification courte et concise. N'oublie pas les guillements autour de la justification), n'invente aucune classe prend bien une(pas deux ou plus) des classes ci-dessus
"""

# Create LangChain prompt template with inputs "commentaire" and "context"
prompt = PromptTemplate(
    input_variables=["commentaire", "context"],
    template=template
)

# Define application state schema with TypedDict for type safety
class State(TypedDict):
    comment       : str
    context       : List[Document]
    initialClass  : str
    predictedClass: str
    llm           : str
    comment_db    : str
    tag_db        : str

# Step 1: Retrieve relevant documents from vector store based on comment similarity
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["comment"], k=1)
    return {"context": retrieved_docs}

# Step 2: Generate classification prediction using LLM with prompt and retrieved context
def generate(state: State):
    model = state["llm"]
    llm = OllamaLLM(model=model)

    # Format context as string with comment:class for prompt input
    context = "\n".join(f""" '{ctx.page_content}':{comments["tag"][int(ctx.id)]} """ for ctx in state["context"])

    # Fill prompt template with comment and context
    formatted_prompt = prompt.invoke({"comment": state["comment"], "context": context})

    # Get prediction from LLM and clean unwanted characters
    response = llm.invoke(formatted_prompt)
    response = ''.join(c for c in response.strip() if c not in "<>'")

    # Retrieve original comment and tag from dataset for baseline comparison
    ctx = state["context"][0]
    comment_db = ctx.page_content
    tag_db     = comments["tag"][int(ctx.id)]

    return {"predictedClass": response,
            "comment_db"    : comment_db,
            "tag_db"        : tag_db}

# Build the LangGraph state machine pipeline: retrieve -> generate
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
