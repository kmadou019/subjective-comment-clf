#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import pandas as pd


llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)

#Load the comments
data = pd.read_csv("comments.csv")
#Merging comment and tag
comments = [Document(f"""{content} : {tag} """) for content,tag in zip(data["content"], data["tag"])]

#Store the comments in the chromadb by indexing them
vector_store.add_documents(documents=comments)

# Define prompt for classification
template = """
Tu es un expert en classification de commentaire. 
Voici un commentaire : 
{comment}

Classe ce commentaire dans l'une des catégories suivantes :
1. ...
2. ...

En te basant sur ces classifications existantes (commentaire : classe):
{context}

Classe du commentaire : 
"""

prompt = PromptTemplate(
    input_variables=["commentaire", "context"],
    template=template
)

# Define state for application
class State(TypedDict):
    comment: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["comment"])
    return {"context": retrieved_docs}

def generate(state: State):
    context = "\n".join(ctx.page_content for ctx in state["context"])
    formatted_prompt = prompt.invoke({"comment": state["comment"], "context": context})
    response = llm.invoke(formatted_prompt)
    return {"answer": response.content}




def main():
    # Compile application
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    # and test application
    response = graph.invoke({"comment": "Il y a eu beaucoup de réponse incorrecte et pas mal d'ignorance"})
    print(response["answer"]) 

if __name__ == "__main__":
    main()