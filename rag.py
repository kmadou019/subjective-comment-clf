#!/usr/bin/env python3
from graph_builder import graph
import pandas as pd
import logging

def main():
    #Load the test set
    comments_test = pd.read_csv("data/test.csv")
    # Créer un logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Handler pour les erreurs
    error_handler = logging.FileHandler('log/misclassification.log',mode="w")
    error_handler.setLevel(logging.ERROR)

    # Handler pour les informations
    info_handler = logging.FileHandler('log/accuracy.log',mode="w")
    info_handler.setLevel(logging.INFO)

    # Ajouter les handlers au logger
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)

    # Test application
    true_prediction = 0
    mistakes_output = ""
    for comment,tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment})
        if response["predictedClass"] == tag:
            true_prediction += 1
        else : 
            logger.error(f""" [{comment}] ; human({tag}) ; IA({response["predictedClass"]}) \n""")
    
    #Accuracy
    accuracy = true_prediction/len(comments_test)
    print("Accuracy = ", accuracy)
    logger.info("Accuracy = %d", accuracy)

    print(mistakes_output)
if __name__ == "__main__":
    main()