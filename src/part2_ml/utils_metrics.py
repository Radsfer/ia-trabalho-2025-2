import os
import sys
import csv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Tenta importar o ensure_folder
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.common.utils_io import ensure_folder
except ImportError:
    def ensure_folder(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def evaluate_model(y_true, y_pred, model_name="Modelo"):
    """
    1. Mostra resultados no terminal.
    2. Salva resumo em CSV (para a tabela do relatório).
    3. Salva detalhes em TXT (para consulta).
    """
    # --- 1. Calcular Métricas ---
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    
    # --- 2. Mostrar no Terminal ---
    print(f"\n{'='*60}")
    print(f"📊 AVALIAÇÃO: {model_name}")
    print(f"{'='*60}")
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print("-" * 30)
    print("Relatório Detalhado:")
    print(report)
    print("-" * 30)
    print("Matriz de Confusão:")
    print(matrix)
    print(f"{'='*60}\n")
    
    # --- 3. Salvar Resumo em CSV (Ótimo para a tabela do relatório) ---
    csv_path = "reports/part2_ml/metrics.csv"
    ensure_folder(csv_path)
    
    # Verifica se o arquivo é novo para escrever o cabeçalho
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Escreve cabeçalho apenas se o arquivo não existia
        if not file_exists:
            writer.writerow(['Model', 'Accuracy', 'F1_Macro'])
        
        writer.writerow([model_name, f"{acc:.4f}", f"{f1:.4f}"])
        
    print(f"✅ Resumo salvo em: {csv_path}")

    # --- 4. Salvar Detalhes em TXT (Backup completo) ---
    txt_path = "reports/part2_ml/metrics_details.txt"
    ensure_folder(txt_path)
    
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\nMODELO: {model_name}\n{'='*60}\n")
        f.write(f"Acurácia: {acc:.4f}\nF1-Macro: {f1:.4f}\n\n")
        f.write(f"Relatório:\n{report}\n")
        f.write(f"Matriz Confusão:\n{matrix}\n\n")
        
    print(f"✅ Detalhes salvos em: {txt_path}")