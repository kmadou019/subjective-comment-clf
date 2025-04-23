#!/usr/bin/env bash

source ../../mon_env/bin/activate
/home/daisy/konema/Documents/ollama/bin/ollama serve &

./rag.py