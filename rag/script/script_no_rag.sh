#!/usr/bin/env bash
#OAR -n rag
#OAR --stdout log/output.txt
#OAR --stderr log/error.txt
#OAR -l /host=1/gpu=4,walltime=7:0:0
cd ..
source mon_env/bin/activate
/home/daisy/konema/Documents/ollama/bin/ollama serve &
echo -e "mistral\nmistral_no_rag\nNo_rag\nmistral" |./rag_no.py
echo -e "phi4\nphi4_no_rag\nNo_rag\nphi4" |./rag_no.py
echo -e "llama3.3\nllama3_3_no_rag\nNo_rag\nllama3.3" |./rag_no.py
