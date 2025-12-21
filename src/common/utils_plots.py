"""
Utilitário para geração de gráficos do relatório.
Gera visualizações para as Partes 2 e 3 do trabalho.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração de Estilo
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Caminhos
BASE_PATH = "reports/"
FIGS_PATH = "reports/figs/"
os.makedirs(FIGS_PATH, exist_ok=True)


def plot_part2_comparison():
    """Gera gráfico de barras comparando os modelos da Parte 2"""
    try:
        df = pd.read_csv(os.path.join(BASE_PATH, "part2_ml/metrics.csv"))
        
        # Transformar dados para formato 'longo' (melhor para o Seaborn)
        df_melted = df.melt(id_vars="Model", var_name="Metrica", value_name="Valor")
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Model", y="Valor", hue="Metrica", data=df_melted, palette="viridis")
        
        # Ajustes visuais
        plt.title("Comparativo de Modelos (Parte 2)", fontsize=14, fontweight='bold')
        plt.xlabel("Modelo")
        plt.ylabel("Pontuacao (0-1)")
        plt.ylim(0.5, 0.65)  # Zoom para ver melhor as diferenças
        plt.legend(title="Metrica", loc='lower right')
        
        # Adicionar valores nas barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)

        output_path = os.path.join(FIGS_PATH, "part2_comparison.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Grafico Parte 2 salvo em: {output_path}")
        
    except FileNotFoundError:
        print("Arquivo metrics.csv nao encontrado. Pulando Parte 2.")


def plot_part3_convergence():
    """Gera curva de convergência do AG da Parte 3"""
    try:
        df = pd.read_csv(os.path.join(BASE_PATH, "part3_ga/generation_history.csv"))
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="Generation", y="Best_Fitness", data=df, marker="o", linewidth=2.5, color="#d62728")
        
        plt.title("Convergencia do Algoritmo Genetico (Parte 3)", fontsize=14, fontweight='bold')
        plt.xlabel("Geracao")
        plt.ylabel("Melhor Acuracia (Fitness)")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Destacar o ponto máximo
        max_val = df["Best_Fitness"].max()
        plt.axhline(y=max_val, color='green', linestyle=':', label=f'Maximo: {max_val:.4f}')
        plt.legend()

        output_path = os.path.join(FIGS_PATH, "part3_convergence.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Grafico Parte 3 salvo em: {output_path}")

    except FileNotFoundError:
        print("Arquivo generation_history.csv nao encontrado. Pulando Parte 3.")


if __name__ == "__main__":
    print("Gerando graficos para o relatorio...")
    plot_part2_comparison()
    plot_part3_convergence()
    print("Concluido!")
