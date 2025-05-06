#!/usr/bin/env python3

import json

def save_checkpoint(iteration, true_prediction):
    data = {
        "iteration": iteration,
        "true_prediction": true_prediction
    }
    with open('./checkpoint.txt', "w") as checkpoint:
        json.dump(data, checkpoint)

def load_checkpoint():
    try:
        with open('./checkpoint.txt', "r") as checkpoint:
            checkpoint = json.load(checkpoint)
        return (int(checkpoint["iteration"]), int(checkpoint["true_prediction"]) )
    except:
        return (0,0)


def main():
    #save_checkpoint(8,15)
    print(load_checkpoint())
if __name__ == "__main__":
    main()
