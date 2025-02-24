#!/usr/bin/env python3
from graph_builder import graph
import pandas as pd
import logging

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
    # Iterate through the test dataset and evaluate predictions
    for comment, tag in zip(comments_test["content"], comments_test["tag"]):
        response = graph.invoke({"comment": comment})
        if response["predictedClass"] == tag:
            true_prediction += 1
        else:
            logger.error(f"[{comment}] ; human({tag}) ; IA({response['predictedClass']})")
    # Calculate accuracy
    accuracy = true_prediction / len(comments_test)
    print("Accuracy = ", accuracy)
    logger.info("Accuracy = %f", accuracy)

if __name__ == "__main__":
    main()
