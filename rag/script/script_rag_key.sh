#!/usr/bin/env bash
#OAR -n rag
#OAR --stdout log/output.txt
#OAR --stderr log/error.txt
#OAR -l /host=1/gpu=4,walltime=7:0:0
cd ..
source ../mon_env/bin/activate
/home/daisy/konema/Documents/ollama/bin/ollama serve >/dev/null 2>&1
#model/name/version/model
echo -e "mistral\nmistral_rag_keyword\nRag_keyword\nmistral" |./rag_key.py
echo -e "phi4\nphi4_rag_keyword\nRag_keyword\nphi4" |./rag_key.py
echo -e "llama3.3\nllama3_3_rag_keyword\nRag_keyword\nllama3.3" |./rag_key.py
