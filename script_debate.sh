#!/usr/bin/env bash

source mon_env/bin/activate
ollama serve &>/dev/null &

rm -f debate_log.txt checkpoint.txt debate/log/*

# Nombre max de tentatives (optionnel, sinon boucle infinie)
MAX_RETRIES=200
RETRY_COUNT=0

# Durée max autorisée pour chaque run (en secondes)
TIME_LIMIT=300  # 10 minutes

while true; do
    echo "Tentative $((RETRY_COUNT+1))..."
    
    # Lancement avec timeout, redirection des erreurs vers un log
    echo -e "mistral" | timeout $TIME_LIMIT python3 -m debate.debate
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
