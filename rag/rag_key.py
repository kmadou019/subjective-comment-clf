#!/usr/bin/env python3
from graph_builder_key import graph
import pandas as pd
import numpy as np
import logging
import itertools
import os
import subprocess
import time
import threading
import re
def get_gpu_power():
    """Récupère la puissance consommée par le GPU en watts"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True
    )

    return sum(list(np.array(result.stdout.strip().split("\n"), dtype=float))) 

def monitor_power(energy_log):
    """Surveille la consommation énergétique du GPU en arrière-plan"""
    while energy_log["running"]:
        power = get_gpu_power()  # Récupère la consommation actuelle (W)
        timestamp = time.time()  # Enregistre le temps actuel
        energy_log["Time"].append(timestamp)
        energy_log["Power"].append(power)
        time.sleep(0.1)  # Mesure toutes les 100 ms

def compute_energy(energy_log):
    """Calcule l'énergie consommée en kWh"""
    total_energy = np.trapz(energy_log["Power"], energy_log["Time"]) # En J
    return total_energy / 3600000  # Convertit Joules → kWh

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
    miss = 0
    
    matrix = pd.DataFrame(data=0,columns=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"],
                           index=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"])
    # --- Lancement de la mesure ---
    energy_log = {"running": True, "Time": [], "Power": []}
    monitor_thread = threading.Thread(target=monitor_power, args=(energy_log,))
    monitor_thread.start()
    print("Début du programme...")
    # Iterate through the test dataset and evaluate predictions
    start = time.time()
    for comment, tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment})
        predicted_class = re.search(r"^\S+", response["predictedClass"]).group(0)
        if predicted_class == tag:
            true_prediction += 1
        else:
            logger.error(f"[{comment}] ; human({tag}) ; IA({response['predictedClass']})")
        if predicted_class in matrix.columns:
            matrix.loc[tag,predicted_class] += 1
        else:
            print(f"Error: {predicted_class} is not in the matrix")
            miss += 1
    end = time.time()
    time_execution = end - start
    print(f"Miss: {miss}")
    print("Programme terminé.")
    # --- Arrêt de la mesure ---
    energy_log["running"] = False
    monitor_thread.join()
    # Calculate accuracy
    accuracy = true_prediction / len(comments_test)
    kappa = kappa_cohen(matrix, len(comments_test))
    # --- Calcul de l'énergie consommée ---
    energy_kwh = compute_energy(energy_log)
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
    # Log the results
    print("Accuracy = ", accuracy)
    print("Kappa = ", kappa)
    print("Energy(kWh) = ", energy_kwh)
    print("Time(min) = ", time_execution/60)
    logger.info("Accuracy = %f", accuracy)
    logger.info("Kappa = %f", kappa)
    logger.info("Energy(kWh) = %f", energy_kwh)
    logger.info("Time(min) = %f", time_execution/60)

if __name__ == "__main__":
    main()
