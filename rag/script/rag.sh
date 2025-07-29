#!/usr/bin/env bash

cd ../
source ../mon_env/bin/activate
ollama serve & >/dev/null 2>&1

rm -f checkpoint.txt

# Nombre max de tentatives (optionnel, sinon boucle infinie)
MAX_RETRIES=2
RETRY_COUNT=0

# Durée max autorisée pour chaque run (en secondes)
TIME_LIMIT=600  # 3 minutes

while true; do
    echo "Tentative $((RETRY_COUNT+1))..."
    
    # Lancement avec timeout, redirection des erreurs vers un log
    #echo -e "mistral_rag\nRag\nmistral" | timeout $TIME_LIMIT ./rag.py
    echo -e "phi4_rag\nRag\nphi4" | timeout $TIME_LIMIT ./rag.py
    #echo -e "llama3_3_rag\nRag\nllama3.3" | timeout $TIME_LIMIT ./rag.py
    
    status=$?

    # Gestion des différents cas de retour
    if [ $status -eq 0 ]; then
        echo "✅ Exécution réussie. Fin du script."
        break
    elif [ $status -eq 124 ]; then
        echo "⏱️ Timeout de $TIME_LIMIT secondes atteint. Relance..."
    else
        echo "❌ Erreur (code $status). Relance..."
    fi

    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "🚫 Nombre maximum de tentatives atteint ($MAX_RETRIES). Abandon."
        exit 1
    fi

    sleep 5  # Pause avant relance
done