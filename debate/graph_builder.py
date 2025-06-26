#!/usr/bin/env python3
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import START, END, StateGraph
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing_extensions import TypedDict
import logging
import ast
from rag.graph_builder import graph as RAG

MAX_TURN = 4

# Define agents
name_orchestrator = "llama3.3"
name_debater1 =     "llama3.3"
name_debater2 =     "llama3.3"

# Initialize the LLMs
orchestrator_llm = OllamaLLM(model=name_orchestrator)
debater1_llm = OllamaLLM(model=name_debater1)
debater2_llm = OllamaLLM(model=name_debater2)

# Initialize memories and conversation chains
memory_orchestrator = ConversationBufferMemory(return_messages=True)
memory_debater1 = ConversationBufferMemory(return_messages=True)
memory_debater2 = ConversationBufferMemory(return_messages=True)

orchestrator = ConversationChain(llm=orchestrator_llm, memory=memory_orchestrator)
debater1 = ConversationChain(llm=debater1_llm, memory=memory_debater1)
debater2 = ConversationChain(llm=debater2_llm, memory=memory_debater2)

# Initialize the log file
log_file = "debate_log.txt"
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_file, mode="a")
logger.setLevel(logging.DEBUG)
handler.setLevel(logging.INFO)
handler.addFilter(lambda record: record.levelno == logging.INFO)
logger.addHandler(handler)

def extract_json(text):
    print(text)
    start = text.find("{")
    end = text.find("}")
    text = text[start:end+1].replace("true",'True').replace("false",'False').replace('"True"',"True").replace('"False"',"False")
    return ast.literal_eval(text)

class State(TypedDict):
    turn : int
    comment: str
    agreement: bool
    debater1_response: dict
    debater2_response: dict
    final_evaluation: str

def Orchestrator(state: State):
    if state['turn'] == 0:
        prompt = f"""
        Vous êtes un orchestrateur supervisant un débat entre deux classificateurs de commentaire d'étudiants.
        Le commentaire à classifier est le suivant : {state['comment']}

        Il doit être classifié dans l'une des neufs classes suivantes :

        MonCal.Interpret : Interprétations des résultats du test allant au-delà des observations factuelles, reflétant les interprétations subjectives des étudiants sur leur performance.

        MonCal.Facts.RF : Observations factuelles sur les résultats du test, incluant des remarques sur le nombre/quantité de réponses correctes et incorrectes.

        MonCal.Facts.DC : Observations factuelles sur les savoirs fragiles, savoirs certains, erreurs dangereuses et erreurs présumées.

        BDS.Emotions : Commentaires exprimant les émotions ressenties par l’étudiant pendant et après le test.

        MotOrient.Value : Commentaires sur l'intérêt des étudiants pour les types de tests et leur valeur perçue pour l’apprentissage.

        MotOrient.SelfEff : Commentaires sur la confiance et l’auto-efficacité des étudiants.

        DomKldg : Commentaires identifiant les concepts/disciplines manquants et leur degré d’acquisition.

        StratKldg : Commentaires sur les stratégies d’apprentissage utilisées pendant le test et l’analyse du comportement des étudiants.

        CTRL : Commentaires sur les comportements futurs des apprenants.

        Ton but sera de vérifier s'il y a un consensus entre les débatteurs.
        """
        orchestrator.predict(input=prompt)
        return {"agreement": False, "turn": state['turn'] + 1}
    else:
        prompt = f"""
        Voici ce qu'a dit le classificateur 1:
        {state["debater1_response"]}
        Et voici ce qu'a dit le classificateur 2:
        {state["debater2_response"]}

        Fait un résumé de la situation, et dis-nous s'il y a un consensus entre les classificateurs ou pas.

        Veuillez bien à ce que ta réponse soit dans le format suivant (Écrit bien True/False avec la première lettre en maj sans quotes) :

        {{
            "turn": {state['turn']},
            "agreement":"True/False",
            "summary": "..."
        }}
        """
        response = extract_json(orchestrator.predict(input=prompt))
        print("Orchestrator : ", response)
        logger.info("\n")
        return {"agreement": response["agreement"], "turn": state["turn"]+1}

def Debater1(state: State):
    if state['turn'] == 1:
        response = extract_json(RAG.invoke({"comment": state["comment"], "llm": name_debater1})["predictedClass"])
        logger.info(f"Réponse du classificateur 1 ({name_debater1}) (RAG) : {response}\n")
    else:
        prompt = f"""
        Dans le tour précédent du débat, tu as donné la classe suivante :
        {state['debater1_response']["classe"]}

        Cependant, le second classificateur a exprimé un désaccord. Voici sa réponse :
        {state['debater2_response']}

        Réexamine ta position à la lumière de sa justification.

        Sois confiant dans ton raisonnement initial, mais reste **ouvert à la discussion** : si l’autre classificateur apporte des **arguments solides**, n’hésite pas à adapter ta position.

        Ta réponse doit être fournie dans le format JSON suivant :
        {{
            "classe": "...",
            "justification": "..."
        }}
        Sois bref dans ta justification.
        """
        response = extract_json(debater1.predict(input=prompt))
        logger.info(f"Réponse du classificateur 1 ({name_debater1}) : {response}\n")

    print("Debater 1 :", response)
    return {"debater1_response": response}

def Debater2(state: State):
    if state['turn'] == 1:
        response = extract_json(RAG.invoke({"comment": state["comment"], "llm": name_debater2})["predictedClass"])
        logger.info(f"Réponse du classificateur 2 ({name_debater2}) (RAG) : {response}\n")
    else:
        prompt = f"""
        Dans le tour précédent du débat, tu as donné la classe suivante :
        {state['debater2_response']["classe"]}

        Cependant, le second classificateur a exprimé un désaccord. Voici sa réponse :
        {state['debater1_response']}

        Réexamine brièvement ta propre réponse à la lumière de cet avis contradictoire.

        Attention :
        Tu dois rester **ferme et confiant** dans ta décision initiale, sauf si les arguments adverses sont **clairement fondés** et plus convaincants.

        Dans tous les cas, justifie soigneusement ta décision.

        Réponds au format JSON :
        {{
            "classe": "...",
            "justification": "..."
        }}
        Sois bref dans ta justification.
        """
        response = extract_json(debater2.predict(input=prompt))
        logger.info(f"Réponse du classificateur 2 ({name_debater2}) : {response}\n")

    print("Debater 2 :", response)
    return {"debater2_response": response}

def end(state: State):
    if state['agreement'] == True or (state['turn'] == MAX_TURN and state['agreement'] == False):
        return "END"
    else:
        return ["debater1", "debater2"]

def END_fnc(state: State):
    if state['turn'] == MAX_TURN and state['agreement'] == False:
        prompt = f"""
        Voici les avis finaux des classificateurs :
        classificateur 1 : {state['debater1_response']}.
        classificateur 2 : {state['debater2_response']}.
        Le débat a atteint le nombre maximal de tours ({MAX_TURN}).
        Résumez le débat et donnez une conclusion.

        Tranche entre les deux si nécessaire.

        Format attendu :
        {{
            "classe" : "...",
            "justification": "..."
        }}
        """
    else:
        prompt = f"""
        Voici les avis finaux des classificateurs :
        classificateur 1 : {state['debater1_response']}.
        classificateur 2 : {state['debater2_response']}.
        Un consensus a été atteint.

        Format attendu :
        {{
            "classe" : "...",
            "justification": "..."
        }}
        """
    final_evaluation = extract_json(orchestrator.predict(input=prompt))
    logger.info(f"Résumé final du débat par l'orchestrateur ({name_orchestrator}) : {final_evaluation}\n")

    # Clean memory
    memory_orchestrator.clear()
    memory_debater1.clear()
    memory_debater2.clear()

    return {"agreement": True, "turn": state['turn'], "final_evaluation": final_evaluation}

# Build the LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("Orchestrator", Orchestrator)
graph_builder.add_node("debater1", Debater1)
graph_builder.add_node("debater2", Debater2)
graph_builder.add_node("END", END_fnc)

graph_builder.add_edge(START, "Orchestrator")
graph_builder.add_edge("debater1", "Orchestrator")
graph_builder.add_edge("debater2", "Orchestrator")
graph_builder.add_conditional_edges("Orchestrator", end, ["debater1", "debater2", "END"])

graph = graph_builder.compile()

if __name__ == "__main__":
    img = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)
    print("Image enregistrée sous 'graph.png'")
