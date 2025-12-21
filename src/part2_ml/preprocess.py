import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configuração de Paths e Seeds
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.common.seeds import set_seeds, DEFAULT_SEED
from src.common.utils_io import ensure_folder

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

# MUDANÇA: fixar o equilíbrio
def load_data():
    print(">>> Carregando datasets brutos...")
    try:
        orders = pd.read_csv(os.path.join(RAW_PATH, "olist_orders_dataset.csv"))
        items = pd.read_csv(os.path.join(RAW_PATH, "olist_order_items_dataset.csv"))
        products = pd.read_csv(os.path.join(RAW_PATH, "olist_products_dataset.csv"))
        customers = pd.read_csv(os.path.join(RAW_PATH, "olist_customers_dataset.csv"))
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos não encontrados em {RAW_PATH}")
        sys.exit(1)

    df = orders.merge(items, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="inner")
    df = df.merge(customers, on="customer_id", how="inner")
    return df

def create_features(df):
    print(">>> Criando Features...")
    cols_date = ['order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in cols_date:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = df.dropna(subset=cols_date).copy()
    
    # Target: 1 se atrasou, 0 se chegou antes ou no dia
    df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)
    
    # Features
    features_cols = [
        'price', 'freight_value', 'product_weight_g', 
        'product_description_lenght', 'product_photos_qty',
        'customer_state'
    ]
    return df[features_cols], df['is_late']

def balance_data(X, y):
    """
    Técnica de Undersampling:
    Pega todos os positivos (atrasos) e sorteia a mesma quantidade de negativos.
    Isso cria um dataset 50/50.
    """
    print(">>> ⚖️ Balanceando dataset (50% Atrasos / 50% No Prazo)...")
    
    # Junta X e y temporariamente
    df_temp = X.copy()
    df_temp['target'] = y
    
    # Separa as classes
    df_late = df_temp[df_temp['target'] == 1]
    df_ok = df_temp[df_temp['target'] == 0]
    
    n_late = len(df_late)
    print(f"    -> Encontrados {n_late} pedidos atrasados.")
    
    # Amostra os normais para ter o mesmo tamanho dos atrasados
    limit = n_late
    # (Limitamos a 15k total se houver muitos atrasos, para performance)
    #limit = min(n_late, 7500) 
    
    df_late_sample = df_late.sample(n=limit, random_state=DEFAULT_SEED)
    df_ok_sample = df_ok.sample(n=limit, random_state=DEFAULT_SEED)
    
    # Junta tudo e embaralha
    df_balanced = pd.concat([df_late_sample, df_ok_sample])
    df_balanced = df_balanced.sample(frac=1, random_state=DEFAULT_SEED).reset_index(drop=True)
    
    print(f"    -> Dataset balanceado final: {len(df_balanced)} linhas.")
    
    return df_balanced.drop(columns=['target']), df_balanced['target']

def main():
    set_seeds()
    
    # 1. Carregar
    df_full = load_data()
    
    # 2. Criar Features (no dataset todo primeiro)
    X_full, y_full = create_features(df_full)
    
    # 3. Balanceamento (AQUI ESTÁ O SEGREDO DO TREINAMENTO REALISTA)
    X, y = balance_data(X_full, y_full)
    
    # 4. Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=DEFAULT_SEED
    )

    # 5. Pipeline (Igual ao anterior)
    numeric_features = ['price', 'freight_value', 'product_weight_g', 'product_description_lenght', 'product_photos_qty']
    categorical_features = ['customer_state']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])

    print(">>> Processando...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Salvar
    save_files = {
        'X_train.npy': X_train_processed,
        'X_test.npy': X_test_processed,
        'y_train.npy': y_train.to_numpy(),
        'y_test.npy': y_test.to_numpy()
    }
    
    print(f">>> Salvando em {PROCESSED_PATH}...")
    for filename, data in save_files.items():
        path = os.path.join(PROCESSED_PATH, filename)
        ensure_folder(path)
        np.save(path, data)
        
    print("✅ Pré-processamento BALANCEADO concluído!")

if __name__ == "__main__":
    main()