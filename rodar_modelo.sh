#/bin/bash

echo "Criando ambiente virtual..."
python3 -m venv ~/meu_ambiente

echo "Ativando ambiente virtual..."
source ~/meu_ambiente/bin/activate

cd ~/modelo_llama

echo "instalando Bibliotecas..."
pip install -r requeriments.txt

cd streamlit-base

echo "Executando a aplicação:"
streamlit run app.py
