#!/usr/bin/env bash

source mon_env/bin/activate
ollama serve &>/dev/null &


python3 -m chunker.rag.rag