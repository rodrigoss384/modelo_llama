#/bin/bash

cd ~/

echo "Excluindo pasta do venv..."
rm -Rf meu_ambiente

echo "Excluindo cache dos arquivos py..."
cd modelo_llama/streamlit-base

rm -Rf __pycache__

cd ~/

echo "Finalizado com sucesso..."
