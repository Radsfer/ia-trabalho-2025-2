import os

def ensure_folder(path):
    """Garante que a pasta do caminho especificado existe."""
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        print(f">>> Pasta criada: {dirname}")