#!/usr/bin/env python3
from graph_builder_no import graph
import pandas as pd
import numpy as np
import logging
import itertools
import os


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

def read_excel_or_create(filename):
    try:
        return pd.read_excel(filename, index_col=[0,1])
    except FileNotFoundError:
        tuple = list(itertools.product(["mistral", "phi4", "llama3.3"], ["No_rag", "Rag", "Rag_keyword"] ))
        index = pd.MultiIndex.from_tuples(tuples=tuple, names=["Model", "Version"])
        performance = pd.DataFrame(columns=["Accuracy", "Kappa"], index=index)
        return performance


def main():
    # Load the test dataset
    comments_test = pd.read_csv("data/test.csv")
    performance = read_excel_or_create("excel/performance.xlsx")

    # Create a dedicated logger for this script
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Handler for error logs
    out = input("Enter the log name : ")
    error_handler = logging.FileHandler(f'log/error_{out}.log', mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(lambda record: record.levelno == logging.ERROR)

    # Handler for info logs
    info_handler = logging.FileHandler(f'log/success_{out}.log', mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)

    # Add handlers to the logger
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
    # Define the model and version
    version = input("Enter the version: ") 
    model = input("Enter the model name: ")

    # Initialize counters
    true_prediction = 0
    
    matrix = pd.DataFrame(data=0,columns=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"],
                           index=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"])
    # Iterate through the test dataset and evaluate predictions
    for comment, tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment})
        predicted_class = response["predictedClass"]
        if predicted_class == tag:
            true_prediction += 1
        else:
            logger.error(f"[{comment}] ; human({tag}) ; IA({response['predictedClass']})")
        if predicted_class in matrix.columns:
            matrix.loc[tag,predicted_class] += 1

    # Calculate accuracy
    accuracy = true_prediction / len(comments_test)
    kappa = kappa_cohen(matrix, len(comments_test))
    #export the matrix as excel file

    filename = "excel/matrix_cohen.xlsx"
    sheet_name = f"{model}_{version}"

    matrix["kappa"] = kappa
    # Create the file if it does not exist
    if not os.path.exists(filename):
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer)  # Create an empty sheet

    with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        matrix.to_excel(writer, sheet_name=sheet_name)

    # Performance
    performance.loc[(model, version), "Accuracy"] = round(accuracy, 2)
    performance.loc[(model, version), "Kappa"] = round(kappa, 2)
    performance.to_excel("excel/performance.xlsx")
    # Log the results
    print("Accuracy = ", accuracy)
    print("Kappa = ", kappa)
    logger.info("Accuracy = %f", accuracy)
    logger.info("Kappa = %f", kappa)


if __name__ == "__main__":
    main()
