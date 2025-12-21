# Makefile para IA Trabalho 2025/2
# Ambiente: Fedora (WSL)

# --- Variáveis ---
VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Pastas
SRC := src
DATA_PROC := data/processed
# Define pastas de relatório separadas para limpeza precisa
REPORTS_ML := reports/part2_ml
REPORTS_GA := reports/part3_ga
REPORTS_SWARM := reports/part4_swarm_immune

# --- Comandos (Targets) ---
.PHONY: help setup install part1 part2 part3 part4 preprocess train test \
        clean clean-data clean-reports clean-py

# Ajuda
help:
	@echo "Automacao do Projeto de IA"
	@echo "Uso:"
	@echo "  make setup         - Cria venv e instala dependencias"
	@echo "  make part1         - Roda Arvore de Decisao Manual"
	@echo "  make part2         - Roda ML (Preprocess + Treino)"
	@echo "  make part3         - Roda Otimizacao AG"
	@echo "  make part4         - Roda PSO e Sistema Imune"
	@echo "  make test          - Roda os testes"
	@echo "  --- Limpeza ---"
	@echo "  make clean         - Limpa TUDO"
	@echo "  make clean-data    - Apaga apenas os .npy (dados processados)"
	@echo "  make clean-reports - Apaga apenas os CSVs e TXTs de resultados"
	@echo "  make clean-py      - Apaga apenas caches (__pycache__)"

# 1. Configuracao
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: setup

# 2. Execucoes Principais
part1:
	@echo "Executando Parte 1..."
	$(PYTHON) $(SRC)/part1_tree_manual/tree_manual.py

preprocess:
	@echo "Pre-processando dados..."
	$(PYTHON) $(SRC)/part2_ml/preprocess.py

train:
	@echo "Treinando modelos..."
	$(PYTHON) $(SRC)/part2_ml/train_knn.py
	$(PYTHON) $(SRC)/part2_ml/train_svm.py
	$(PYTHON) $(SRC)/part2_ml/train_tree.py

part2: preprocess train

part3:
	@echo "Rodando Otimizacao (AG)..."
	$(PYTHON) $(SRC)/part3_ga/run_tuning.py

part4:
	@echo "Rodando PSO..."
	$(PYTHON) $(SRC)/part4_swarm_immune/pso_tunning.py
	@echo "Rodando Sistema Imune..."
	$(PYTHON) $(SRC)/part4_swarm_immune/immune_tuning.py

# 3. Testes
test:
	@echo "Rodando testes..."
	PYTHONPATH=. $(PYTEST) tests/ -v

# 4. Limpeza Modular

clean-data:
	@echo "Limpando dados processados..."
	rm -rf $(DATA_PROC)/*.npy

clean-reports:
	@echo "Limpando relatorios (ML, AG e Swarm)..."
	rm -rf $(REPORTS_ML)/*.csv $(REPORTS_ML)/*.txt
	rm -rf $(REPORTS_GA)/*.csv $(REPORTS_GA)/*.txt
	rm -rf $(REPORTS_SWARM)/*.csv $(REPORTS_SWARM)/*.txt $(REPORTS_SWARM)/*.png

clean-py:
	@echo "Limpando cache do Python..."
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

# O 'clean' geral chama todos os sub-cleans
clean: clean-data clean-reports clean-py
	@echo "Limpeza COMPLETA concluida!"