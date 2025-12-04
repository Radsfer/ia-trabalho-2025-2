# Makefile para IA Trabalho 2025/2
# Ambiente: Fedora (WSL)

# --- Variáveis ---
VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Pastas (Ajustado aqui!)
SRC := src
DATA_PROC := data/processed
REPORTS := reports/part2_ml

# --- Comandos (Targets) ---
.PHONY: help setup install part1 part2 preprocess train test clean

# Ajuda
help:
	@echo "🤖 Automação do Projeto de IA"
	@echo "Uso:"
	@echo "  make setup      - Cria o ambiente virtual e instala dependências"
	@echo "  make part1      - Executa a Árvore Manual"
	@echo "  make part2      - Roda o pipeline de ML (Preprocess + Treino)"
	@echo "  make test       - Executa os testes unitários"
	@echo "  make clean      - Limpa arquivos gerados em $(REPORTS) e $(DATA_PROC)"

# 1. Configuração
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: setup

# 2. Execução
part1:
	@echo "🚀 Executando Parte 1..."
	$(PYTHON) $(SRC)/part1_tree_manual/tree_manual.py

preprocess:
	@echo "🧹 Pré-processando dados..."
	$(PYTHON) $(SRC)/part2_ml/preprocess.py

train:
	@echo "🧠 Treinando modelos..."
	$(PYTHON) $(SRC)/part2_ml/train_knn.py
	$(PYTHON) $(SRC)/part2_ml/train_svm.py
	$(PYTHON) $(SRC)/part2_ml/train_tree.py

part2: preprocess train

# 3. Algoritmo Genético
part3:
	@echo "🧬 Rodando Otimização de Hiperparâmetros (AG)..."
	$(PYTHON) $(SRC)/part3_ga/run_tuning.py

# 4. Testes
test:
	@echo "🧪 Rodando testes..."
	PYTHONPATH=. $(PYTEST) tests/ -v

# 5. Limpeza (Corrigido para a tua pasta)
clean:
	@echo "🗑️ Limpando arquivos..."
	rm -rf $(DATA_PROC)/*.npy
	rm -rf $(REPORTS)/*.csv $(REPORTS)/*.txt
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✨ Limpeza concluída!"