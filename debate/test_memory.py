#!/usr/bin/env python3

from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

name_orchestrator = "mistral"
model_orchestrator =  OllamaLLM(model=name_orchestrator)
memory_orchestrator = ConversationBufferMemory()
orchestrator = ConversationChain(llm=model_orchestrator, memory = memory_orchestrator)

response = orchestrator.invoke({"input" : "What is the capital of France ?"})

print("response 1:", response)

response = orchestrator.invoke({"input" : "What was the previous question ?"})

print("response 2:", response)

