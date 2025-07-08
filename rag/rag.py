#!/usr/bin/env python3
from graph_builder import graph
import pandas as pd
import numpy as np
import logging
import itertools
import os
import subprocess
import time
import re
import threading
import ast
from collections import defaultdict

def extract_json(text):
    start = text.find("{")
    end = text.find("}")
    text = text[start:end+1].replace("true",'True').replace("false",'False').replace('"True"',"True").replace('"False"',"False")
    print(text)
    return ast.literal_eval(text)

def get_cpu_power():
    """Récupère la puissance consommée par le CPU en watts"""
    #result = subprocess.run(
    #    ["cat", "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"],
    #    stdout=subprocess.PIPE, text=True
    #)
    #return int(result.stdout.strip())
    return 0
def monitor_cpu_power(energy_log):
    while energy_log["running"]:
        power = get_cpu_power()
        time_ = time.time()
        energy_log["Time"].append(time_)
        energy_log["Power"].append(power)
        time.sleep(0.1)  # Sleep for 0.1 seconds to avoid excessive CPU usage


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
        performance = pd.DataFrame(columns=["Accuracy", "Kappa","Energy(kWh)","Time"], index=index)
        return performance

def convert_seconds(time):
    minutes, seconds = divmod(time, 60)
    return f"{int(minutes)}m{int(seconds)}s"

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
    # Handler for miss logs
    miss_handler = logging.FileHandler(f'log/miss_{out}.log', mode="w")
    miss_handler.setLevel(logging.WARNING)
    miss_handler.addFilter(lambda record: record.levelno == logging.WARNING)

    # Handler for info logs
    info_handler = logging.FileHandler(f'log/success_{out}.log', mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)

    # Add handlers to the logger
    logger.addHandler(error_handler)
    logger.addHandler(miss_handler)
    logger.addHandler(info_handler)
    # Define the model and version
    version = input("Enter the version: ") 
    model = input("Enter the model name: ")
    llm = input("Enter the llm name: ")

    # Initialize counters
    true_prediction = 0
    miss = 0
    
    matrix = pd.DataFrame(data=0,columns=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"],
                           index=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"])
    # --- Lancement de la mesure ---
    # Initialize the energy log
    energy_log = {"Time": [], "Power": [], "running": True}
    #Initialize the dic for the baseline analysis
    tab_baseline = {"comment_to_clf": [], "comment_in_db" : [], "classe_in_db" : [], "rag_clf":[], "true_clf": []}
    #Init output
    output = defaultdict(list)
    # Start the CPU power monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_cpu_power, args=(energy_log,))
    monitor_thread.start()
    print("Début du programme...")
    # Iterate through the test dataset and evaluate predictions
    start = time.time()
    for comment, tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment, "llm" : llm})
        predicted_class = extract_json(response["predictedClass"])["classe"]
        #Data for including the baseline in the excel tab
        if response["tag_db"] != predicted_class:
            tab_baseline["comment_to_clf"].append(comment)
            tab_baseline["comment_in_db"].append(response["comment_db"])
            tab_baseline["classe_in_db"].append(response["tag_db"])
            tab_baseline["rag_clf"].append(predicted_class)
            tab_baseline["true_clf"].append(tag)
        
        #Output file
        output[predicted_class].append(comment)
        if predicted_class == tag:
            true_prediction += 1
        else:
            logger.error(f"[{comment}] ; human({tag}) ; IA({predicted_class})")
        if predicted_class in matrix.columns:
            matrix.loc[tag,predicted_class] += 1
        else:
            logger.warning(f"Error: {predicted_class} is not in the matrix")
            miss += 1
    end = time.time()

    energy_log["running"] = False  
    monitor_thread.join()  # Wait for the monitoring thread to finish
    time_execution = end - start
    print("Programme terminé.")
    # --- Arrêt de la mesure ---
    # Calculate accuracy
    accuracy = true_prediction / len(comments_test)
    kappa = kappa_cohen(matrix, len(comments_test))
    # --- Calcul de l'énergie consommée ---
    energy_kwh = energy_log["Power"][-1] - energy_log["Power"][0] / 3.6e6  # Convert to kWh
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
    performance.loc[(model, version), "Energy(kWh)"] = round(energy_kwh, 5)
    performance.loc[(model, version), "Time"] = convert_seconds(time_execution)
    performance.to_excel("excel/performance.xlsx")
    #Baseline vs RAG (difference in classif)
    df_baseline = pd.DataFrame(data=tab_baseline)
    df_baseline.to_excel("excel/baseline_vs_rag_diff.xlsx")
    #Output
    df_output = pd.DataFrame( dict( [ (k, pd.Series(v)) for k,v in output.items()] ) )
    df_output.to_excel("excel/output.xlsx")
    # Log the results
    print(f"Miss: {miss}")
    print("Accuracy = ", accuracy)
    print("Kappa = ", kappa)
    print("Energy(kWh) = ", energy_kwh)
    print("Time(min) = ", time_execution/60)
    logger.info("Accuracy = %f", accuracy)
    logger.info("Kappa = %f", kappa)
    logger.info("Energy(kWh) = %f", energy_kwh)
    logger.info("Time(min) = %f", time_execution/60)
    logger.info(f"Miss: {miss}")

if __name__ == "__main__":
    main()
