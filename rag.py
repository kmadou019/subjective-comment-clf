#!/usr/bin/env python3
from graph_builder import graph
import pandas as pd
import numpy as np
import logging

def kappa_cohen(matrix, n):
    matrix.loc["SumCol",:] = matrix.sum(axis=0) # Sum of the column
    matrix.loc[matrix.index[:-1], "SumRow"] = matrix.sum(axis=1) # Sum of the row
    sum_diag = np.trace(matrix.iloc[:-1,:-1])

    num = sum_diag
    denom = n
    for i in range(len(matrix)-1):
        ef = (matrix.iloc[i,-1] * matrix.iloc[-1,i]) / n
        num -= ef
        denom -= ef
    kappa = num / denom
    return kappa

def main():
    # Load the test dataset
    comments_test = pd.read_csv("data/test.csv")

    # Create a dedicated logger for this script
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Handler for error logs
    error_handler = logging.FileHandler('log/error_phi4_keywords.log', mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(lambda record: record.levelno == logging.ERROR)

    # Handler for info logs
    info_handler = logging.FileHandler('log/success_phi4_keywords.log', mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)

    # Add handlers to the logger
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
    # Initialize counters
    true_prediction = 0
<<<<<<< HEAD
=======
    
    matrix = pd.DataFrame(data=0,columns=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"],
                           index=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"])
>>>>>>> 9e08062768fa97d880e7334026e39bd45123c892
    # Iterate through the test dataset and evaluate predictions
    for comment, tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment})
        matrix[tag,response["predictedClass"]] += 1
        if response["predictedClass"] == tag:
            true_prediction += 1
        else:
            logger.error(f"[{comment}] ; human({tag}) ; IA({response['predictedClass']})")
    # Calculate accuracy
    accuracy = true_prediction / len(comments_test)
    kappa = kappa_cohen(matrix, len(comments_test))
    #export the matrix as excel file
    matrix["kappa"] = kappa
    matrix.to_excel("excel/matrix_cohen.xlsx")

    print("Accuracy = ", accuracy)
    print("Kappa = ", kappa)
    logger.info("Accuracy = %f", accuracy)
    logger.info("Kappa = %f", kappa)


if __name__ == "__main__":
    main()
