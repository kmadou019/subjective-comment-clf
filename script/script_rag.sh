#!/usr/bin/env bash
#OAR -n rag
#OAR --stdout log/output.txt
#OAR --stderr log/error.txt
#OAR -l /host=1/gpu=4,walltime=7:0:0
cd ..
source mon_env/bin/activate
/home/daisy/konema/Documents/ollama/bin/ollama serve &

echo -e "mistral\nmistral_rag\nRag\nmistral" |./rag.py
echo -e "phi4\nphi4_rag\nRag\nphi4" |./rag.py
echo -e "llama3.3\nllama3_3_rag\nRag\nllama3.3" |./rag.py
