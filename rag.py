#!/usr/bin/env python3
from graph_builder import graph
import pandas as pd

def main():
    #Load the test set
    comments_test = pd.read_csv("./test.csv")
    # Test application
    true_prediction = 0
    mistakes_output = ""
    for comment,tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment})
        if response["predictedClass"] == tag:
            true_prediction += 1
        else : 
            mistakes_output +=  f""" [{comment}] ; human({tag}) ; IA({response}) \n"""
    
    #Accuracy
    accuracy = true_prediction/len(comments_test)
    print("Accuracy = ", accuracy)

    print(mistakes_output)
if __name__ == "__main__":
    main()