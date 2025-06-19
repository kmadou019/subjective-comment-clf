#!/usr/bin/env python3
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import START,END, StateGraph
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing_extensions import TypedDict
import logging
import ast
from rag.graph_builder import graph as RAG
MAX_TURN = 4

#Define agents
name_orchestrator = "mistral"
name_debater1 = "phi4"
name_debater2 = "llama3.3"
# Initialize the LLMs
orchestrator =  OllamaLLM(model=name_orchestrator)
debater1 = OllamaLLM(model=name_debater1)
debater2 = OllamaLLM(model=name_debater2)
#Intitialize memories


#Intialize the log file
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

        il doit être classifier dans l'une des neufs classes suivantes dont la description est également fournie : 

        MonCal.Interpret : Interprétations des résultats du test allant au-delà des observations factuelles, reflétant les interprétations subjectives des étudiants sur leur performance.

        MonCal.Facts.RF : Observations factuelles sur les résultats du test, incluant des remarques sur le nombre/quantité de réponses correctes et incorrectes.

        MonCal.Facts.DC : Observations factuelles sur les savoirs fragiles, savoirs certains, erreurs dangereuses et erreurs présumées.

        BDS.Emotions : Commentaires exprimant les émotions ressenties par l’étudiant pendant et après le test.

        MotOrient.Value : Commentaires sur l'intérêt des étudiants pour les types de tests et leur valeur perçue pour l’apprentissage.

        MotOrient.SelfEff : Commentaires sur la confiance et l’auto-efficacité des étudiants.

        DomKldg : Commentaires identifiant les concepts/disciplines manquants et leur degré d’acquisition.

        StratKldg : Commentaires sur les stratégies d’apprentissage utilisées pendant le test et l’analyse du comportement des étudiants.

        CTRL : Commentaires sur les comportements futurs des apprenants.

        Ton but sera de vérifier s'il y a un consensus entre les debatteurs.

        """
        orchestrator.invoke(prompt)
        return {"agreement": False, "turn": state['turn'] + 1}
    else:
        prompt = f"""
        Voici ce qu'à dit le classificateur 1:
        {state["debater1_response"]}
        et voici ce qu'à dit le classificateur 2
        {state["debater2_response"]}

        fait un résumé de la situation, et dit nous s'il y a un consensus entre les classificateurs ou pas. 
        Veuillez bien à ce que ta reponse soit dans le format suivant(Ecrit bien True/False avec la première lettre en maj sans quotes):
                
        {{
            "turn": {state['turn']},
            "agreement": "True/False",
            "summary": "..."
        }}

        """
        response = extract_json(orchestrator.invoke(prompt))
        print("Orchestrator : ", response)
        logger.info("\n")
        return {"agreement": response["agreement"], "turn": state["turn"]+1}

def Debater1(state: State):
    response = ""
    if state['turn'] == 1:
        response = extract_json(RAG.invoke({"comment":state["comment"], "llm" : name_debater1})["predictedClass"])
        logger.info(f"Réponse du classificateur 1 ({name_debater1}) (RAG) : {response}\n")
    else:
        prompt = f"""
        Dans le tour précédent du débat, tu as donné la classe suivante : 
        {state['debater1_response']["classe"]}

        Cependant, le second classificateur a exprimé un désaccord. Voici sa réponse : 
        {state['debater2_response']}

        Réexamine ta position à la lumière de sa justification. 

        Sois confiant dans ton raisonnement initial, mais reste **ouvert à la discussion** : si l’autre classificateur apporte des **arguments solides** ou met en lumière une **faiblesse réelle** dans ta réponse, n’hésite pas à adapter ta position.

        Ne modifie pas ta réponse sans raison valable, mais sois prêt à reconnaître une meilleure justification si elle est clairement fondée.

        Quelle que soit ta décision — maintien ou changement de classe — explique ton raisonnement de manière claire et rigoureuse.

        Ta réponse doit être fournie dans le format JSON suivant :
        {{
            "classe": "...",
            "justification": "..."
        }}
        Soit bref dans ta justification.
        """
        response = extract_json(debater1.invoke(prompt))
        logger.info(f"Réponse du classificateur 1 ({name_debater1}) : {response}\n")
       
    print("Debator 1 :", response)
    return {"debater1_response": response }

def Debater2(state: State):
    response = ""
    if state['turn'] == 1:
        response = extract_json(RAG.invoke({"comment":state["comment"], "llm": name_debater2})["predictedClass"])
        logger.info(f"Réponse du classificateur 2 ({name_debater2}) (RAG) : {response}\n")
    else:
        prompt = f"""
        Dans le tour précédent du débat, tu as donné la classe suivante : 
        {state['debater2_response']["classe"]}

        Cependant, le second classificateur a exprimé un désaccord. Voici sa réponse : 
        {state['debater1_response']}

        Réexamine brièvement ta propre réponse à la lumière de cet avis contradictoire.

        Attention : 
        Tu dois rester **ferme et confiant** dans ta décision initiale, sauf si les arguments adverses sont **clairement fondés et plus convaincants que les tiens**.

        Tu ne dois **pas changer d’avis à la légère**. Un changement de réponse ne doit se produire que si :
        - L’argumentation adverse est **factuellement correcte**,
        - Elle met en évidence une **erreur explicite** ou une **faiblesse importante** dans ta propre justification.

        Dans tous les cas, justifie soigneusement ta décision, en soulignant **pourquoi tu maintiens ou modifies ta position.**

        Ta réponse doit être au format JSON suivant :
        {{
            "classe": "...",
            "justification": "..."
        }}
        Soit bref dans ta justification.

        """
        response = extract_json(debater2.invoke(prompt))
        logger.info(f"Réponse du classificateur 2 ({name_debater2}) : {response}\n")
    print("Debator 2 :", response)
    return {"debater2_response": response }

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
        Vous êtes l'orchestrateur.
        Le débat a atteint le nombre maximal de tours ({MAX_TURN}).
        Résumez le débat et donnez une conclusion.
        Si consensus n'a pas été atteint entre les classificateurs alors tranche entre eux et donne la bonne classe et une seule en produisant la justification adéquate.
        Veuille à me fournir ta reponse dans le format json suivant :
        {{
            "classe" : "...",
            "justification": "..."
        }}
        """
    if state['agreement'] == True:
        prompt = f"""
        Voici les avis finaux des classificateurs :
        classificateur 1 : {state['debater1_response']}.
        classificateur 2 : {state['debater2_response']}.
        Un consensus a été atteint.
        Fournissez la reponse sous ce format, veuille bien à me retourner qu'une seule classe : 
         {{
            "classe" : "...",
            "justification": "..."
        }}
  
        """
    final_evaluation = extract_json(orchestrator.invoke(prompt))
    logger.info(f"Résumé final du débat par l'orchestrateur ({name_orchestrator}) : {final_evaluation}\n")
    #clean buffer
    return {"agreement": True, "turn": state['turn'], "final_evaluation": final_evaluation}

graph_builder = StateGraph(State)
graph_builder.add_node("Orchestrator", Orchestrator)
graph_builder.add_node("debater1", Debater1)
graph_builder.add_node("debater2", Debater2)
graph_builder.add_node("END", END_fnc)

graph_builder.add_edge(START, "Orchestrator")
graph_builder.add_edge("debater1", "Orchestrator")
graph_builder.add_edge("debater2", "Orchestrator")
graph_builder.add_conditional_edges("Orchestrator", end,["debater1","debater2", "END"])
graph = graph_builder.compile()

if __name__=="__main__":
    img = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)
    print("Image enregistrée sous 'graph.png'")