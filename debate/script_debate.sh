#!/usr/bin/env bash

source ../mon_env/bin/activate
ollama serve & >/dev/null 2>&1

echo -e "mistral" |./debate.py
