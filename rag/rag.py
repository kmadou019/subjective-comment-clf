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
    """
    Extracts and sanitizes a JSON-like structure from a raw text string,
    making it safe to evaluate using `ast.literal_eval`.

    This function assumes the JSON-like content is embedded within some larger text.
    It attempts to:
      1. Extract the portion between the first '{' and the last '}'.
      2. Replace JSON booleans `true`/`false` with Python equivalents `True`/`False`.
      3. Quote unquoted string values inside lists (e.g., [foo, bar] → ["foo", "bar"]).
    
    Args:
        text (str): The raw input string containing a JSON-like dictionary.
    
    Returns:
        dict: A Python dictionary parsed from the cleaned string.

    Raises:
        ValueError: If the extracted string cannot be safely evaluated.
    """

    # Find the first '{' and the last '}' to isolate the dictionary-like content
    start = text.find("{")
    end = text.rfind("}")
    text = text[start:end+1]

    # Replace JSON booleans with Python equivalents
    text = text.replace("true", "True").replace("false", "False")

    # Function to wrap unquoted tokens in lists with double quotes
    def quote_unquoted_tokens(match):
        content = match.group(1)
        tokens = content.split(",")
        new_tokens = []
        for token in tokens:
            token = token.strip()
            # If token is not a number and not already quoted, add double quotes
            if not re.match(r'^-?\d+(\.\d+)?$', token) and not (token.startswith('"') and token.endswith('"')):
                token = f'"{token}"'
            new_tokens.append(token)
        return "[" + ", ".join(new_tokens) + "]"

    # Apply the quoting fix to all list expressions in the text
    text = re.sub(r'\[\s*([^\[\]]+?)\s*\]', quote_unquoted_tokens, text)

    # Optional: print the cleaned text for debugging
    print(text)

    # Safely evaluate the cleaned string as a Python literal (e.g., dict)
    return ast.literal_eval(text)

def get_gpu_power():
    """
    Retrieves the total power consumption of all available NVIDIA GPUs in watts.

    This function uses `nvidia-smi` to query the current power draw of each GPU,
    then parses and sums the results to return the total power usage.

    Returns:
        float: Total power consumption in watts across all GPUs.
    """
    # Run nvidia-smi to get the power draw (in watts) for each GPU, without header or units
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True
    )

    # Extract each line, convert to float, and sum all GPU power draws
    power_values = np.array(result.stdout.strip().split("\n"), dtype=float)
    return power_values.sum()


def monitor_power(energy_log):
    """
    Continuously monitors GPU power consumption in the background.

    This function is meant to be run in a separate thread or process.
    It samples the current GPU power draw every 100 milliseconds and
    stores (timestamp, power) tuples in the provided shared dictionary.

    Args:
        energy_log (dict): A shared dictionary with:
            - "running" (bool): Set to True to start monitoring; set to False to stop.
            - "samples" (list): A list to which (timestamp, power) tuples are appended.
    """
    while energy_log["running"]:
        power = get_gpu_power()      # Get current power draw in watts
        timestamp = time.time()      # Record current timestamp
        energy_log["samples"].append((timestamp, power))
        time.sleep(0.1)              # Sleep for 100 milliseconds between samples


def compute_energy(energy_log):
    """
    Computes the total energy consumed by the GPU based on logged power samples.

    The energy is calculated by integrating the power over time using the trapezoidal rule.
    The final result is returned in kilowatt-hours (kWh).

    Args:
        energy_log (dict): A dictionary containing a "samples" list, where each element is a tuple:
            (timestamp: float, power: float in watts)

    Returns:
        float: Total energy consumed in kilowatt-hours (kWh)
    """
    samples = energy_log["samples"]
    total_energy = 0  # Total energy in Joules

    # Iterate over each consecutive pair of samples
    for i in range(1, len(samples)):
        t1, p1 = samples[i - 1]
        t2, p2 = samples[i]
        dt = t2 - t1                      # Time interval in seconds
        avg_power = (p1 + p2) / 2         # Average power in watts
        total_energy += avg_power * dt    # Energy = power × time (in Joules)

    # Convert Joules to kilowatt-hours (1 kWh = 3.6 million Joules)
    return total_energy / 3_600_000


def kappa_cohen(matrix, n):
    """
    Computes Cohen's Kappa coefficient from a confusion matrix.

    This function augments the given confusion matrix by adding:
    - A row sum ("SumRow") for each actual class
    - A column sum ("SumCol") for each predicted class

    It then calculates the expected agreement by chance and computes
    Cohen's Kappa, which measures inter-rater agreement corrected for chance.

    Args:
        matrix (pd.DataFrame): Square confusion matrix (classes × classes).
                               This matrix is modified in-place.
        n (int): Total number of samples (i.e., sum of all cells in the matrix).
    
    Returns:
        float: Cohen's Kappa score.
    """
    # Add a new row: sum of each column (total predicted per class)
    matrix.loc["SumCol", :] = matrix.sum(axis=0)

    # Add a new column: sum of each row (total actual per class)
    matrix.loc[matrix.index[:-1], "SumRow"] = matrix.sum(axis=1)

    # Compute the number of samples with agreement (trace of the main matrix)
    sum_diag = np.trace(matrix.iloc[:-1, :-1])

    # Initialize numerator and denominator for Cohen's Kappa formula
    num = sum_diag
    denom = n

    # Subtract expected agreement (by chance) from both numerator and denominator
    for i in range(len(matrix) - 1):  # Exclude "SumCol" row
        expected_freq = (matrix.iloc[i, -1] * matrix.iloc[-1, i]) / n
        num -= expected_freq
        denom -= expected_freq

    # Compute and return Kappa
    kappa = num / denom
    return kappa


def read_excel_or_create(filename):
    """
    Reads an Excel file into a pandas DataFrame with a MultiIndex.
    If the file does not exist, it creates and returns a new empty DataFrame
    with a MultiIndex of predefined (Model, Version) combinations.

    Args:
        filename (str): Path to the Excel file to read.

    Returns:
        pd.DataFrame: A DataFrame either loaded from file or newly created,
                      with columns: ["Accuracy", "Kappa", "Energy(kWh)", "Time"]
                      and MultiIndex on ["Model", "Version"].
    """
    try:
        # Try to read an existing Excel file with a MultiIndex (2 levels)
        return pd.read_excel(filename, index_col=[0, 1])
    
    except FileNotFoundError:
        # Define index combinations if the file doesn't exist
        model_version_pairs = list(itertools.product(
            ["mistral", "phi4", "llama3.3"],
            ["No_rag", "Rag", "Rag_keyword"]
        ))
        index = pd.MultiIndex.from_tuples(model_version_pairs, names=["Model", "Version"])

        # Create an empty DataFrame with the desired structure
        performance = pd.DataFrame(columns=["Accuracy", "Kappa", "Energy(kWh)", "Time"], index=index)
        return performance

def convert_seconds(time):
    """
    Converts a duration from seconds to a formatted string in minutes and seconds.

    Args:
        time (float or int): Duration in seconds.

    Returns:
        str: Formatted string in the form 'XmYs' (e.g., '2m15s').
    """
    minutes, seconds = divmod(time, 60)  # Split total seconds into minutes and remainder seconds
    return f"{int(minutes)}m{int(seconds)}s"


def main():
   # Load the test dataset
    comments_test = pd.read_csv("data/test.csv")
    performance = read_excel_or_create("../excel/performance.xlsx")

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Log file name input
    out = input("Enter the log name : ")

    # Error log handler
    error_handler = logging.FileHandler(f'log/error_{out}.log', mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(lambda record: record.levelno == logging.ERROR)

    # Warning log handler (misses)
    miss_handler = logging.FileHandler(f'log/miss_{out}.log', mode="w")
    miss_handler.setLevel(logging.WARNING)
    miss_handler.addFilter(lambda record: record.levelno == logging.WARNING)

    # Info log handler (success summary)
    info_handler = logging.FileHandler(f'log/success_{out}.log', mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)

    # Register handlers
    logger.addHandler(error_handler)
    logger.addHandler(miss_handler)
    logger.addHandler(info_handler)

    # Model and version input
    version = input("Enter the version: ") 
    llm = model = input("Enter the model name: ")

    # Initialize metrics and structures
    true_prediction = 0
    miss = 0
    energy_log = {"running": True, "samples": []}
    tab_baseline = {"comment_to_clf": [], "comment_in_db" : [], "classe_in_db" : [], "rag_clf":[], "true_clf": []}
    output = defaultdict(list)
    cls_cert = defaultdict(list)

    # Initialize confusion matrix
    matrix = pd.DataFrame(data=0,
                        columns=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", 
                                "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"],
                        index=["MonCal.Interpret", "MonCal.Facts.RF", "MonCal.Facts.DC", "BDS.Emotions", 
                                "MotOrient.Value", "MotOrient.SelfEff", "DomKldg", "StratKldg", "CTRL"])

    # Start GPU power monitoring
    monitor_thread = threading.Thread(target=monitor_power, args=(energy_log,))
    monitor_thread.start()

    print("Début du programme...")

    # Start timer
    start = time.time()

    # Run inference loop (only first 10 examples)
    for comment, tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment, "llm" : llm})
        response_json = extract_json(response["predictedClass"])
        predicted_class = response_json["classe"]
        certitude = response_json["certitude"]

        # Collect baseline vs RAG differences
        if response["tag_db"] != predicted_class:
            tab_baseline["comment_to_clf"].append(comment)
            tab_baseline["comment_in_db"].append(response["comment_db"])
            tab_baseline["classe_in_db"].append(response["tag_db"])
            tab_baseline["rag_clf"].append(predicted_class)
            tab_baseline["true_clf"].append(tag)

        # Collect outputs per class
        for class_ in predicted_class:
            output[class_].append(comment)

        # Store full classification + certitude
        cls_cert["comment"].append(comment)
        cls_cert["cls"].append(predicted_class)
        cls_cert["certitude"].append(certitude)
        cls_cert["justification"].append(response_json["justification"])

        # Update accuracy
        if tag in predicted_class:
            true_prediction += 1
        else:
            logger.error(f"[{comment}] ; human({tag}) ; IA({predicted_class})")

        # Update confusion matrix or log miss
        if tag in (predicted_class[0],predicted_class[-1]):
            matrix.loc[tag, tag] += 1
        elif predicted_class[0] in matrix.columns:
            matrix.loc[tag, predicted_class[0]] += 1
        elif predicted_class[-1] in matrix.columns:
            matrix.loc[tag, predicted_class[-1]] += 1
        else:
            logger.warning(f"Error: {predicted_class} is not in the matrix")
            miss += 1

    # Stop monitoring
    end = time.time()
    energy_log["running"] = False  
    monitor_thread.join()

    time_execution = end - start
    print("Programme terminé.")

    # Compute metrics
    accuracy = true_prediction / len(comments_test)
    kappa = kappa_cohen(matrix, len(comments_test))
    energy_kwh = compute_energy(energy_log)

    # Save confusion matrix with Kappa
    matrix["kappa"] = kappa
    filename = "../excel/matrix_cohen.xlsx"
    sheet_name = f"{model}_{version}"

    if not os.path.exists(filename):
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer)

    with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        matrix.to_excel(writer, sheet_name=sheet_name)

    # Update performance file
    performance.loc[(model, version), "Accuracy"] = round(accuracy, 2)
    performance.loc[(model, version), "Kappa"] = round(kappa, 2)
    performance.loc[(model, version), "Energy(kWh)"] = round(energy_kwh, 5)
    performance.loc[(model, version), "Time"] = convert_seconds(time_execution)
    performance.to_excel("../excel/performance.xlsx")

    # Save baseline vs RAG differences
    df_baseline = pd.DataFrame(data=tab_baseline)
    df_baseline.to_excel("../excel/baseline_vs_rag_diff.xlsx")

    # Save class-wise output
    df_output = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in output.items()]))
    df_output.to_excel("../excel/output.xlsx")

    # Save certitude output
    df_cls_cert = pd.DataFrame.from_dict(cls_cert)
    df_cls_cert.to_excel("../excel/cls_cert.xlsx", index=False)

    # Log final metrics
    print(f"Miss: {miss}")
    print("Accuracy = ", accuracy)
    print("Kappa = ", kappa)
    print("Energy(kWh) = ", energy_kwh)
    print("Time(min) = ", time_execution / 60)

    logger.info("Accuracy = %f", accuracy)
    logger.info("Kappa = %f", kappa)
    logger.info("Energy(kWh) = %f", energy_kwh)
    logger.info("Time(min) = %f", time_execution / 60)
    logger.info(f"Miss: {miss}")


if __name__ == "__main__":
    main()
