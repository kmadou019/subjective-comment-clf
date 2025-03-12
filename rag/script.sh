#!/usr/bin/env bash
#OAR -n rag
#OAR --stdout log/output.txt
#OAR --stderr log/error.txt
#OAR -l /host=1/gpu=4,walltime=7:0:0
source mon_env/bin/activate
/home/daisy/konema/Documents/ollama/bin/ollama ps > running.oll
/home/daisy/konema/Documents/ollama/bin/ollama serve &
./rag.py
