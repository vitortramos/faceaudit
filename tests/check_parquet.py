import pandas as pd
import os
import numpy as np

# Caminho exato que apareceu no seu log de erro anterior
FILE_PATH = r"C:\Users\perito\OneDrive\Ferramentas\Reconhecimento Facial\faceaudit\data\datasets\BFW_Base_filtered_frontais.parquet"

def inspect_parquet(path):
    print(f"🔍 Inspecionando: {path}")
    
    if not os.path.exists(path):
        print("❌ Arquivo não encontrado!")
        return

    # 1. Checagem de Tamanho
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"📦 Tamanho do arquivo: {size_mb:.2f} MB")
    
    # Se o arquivo tiver menos de 1MB e tiver muitas linhas, é quase certeza que não tem embeddings
    if size_mb < 1.0:
        print("⚠️ ALERTA: O arquivo é muito pequeno. Provavelmente não contém vetores.")

    try:
        # 2. Carregamento do DataFrame
        df = pd.read_parquet(path)
        print(f"📊 Total de Registros (Linhas): {len(df)}")
        print("-" * 40)
        
        # 3. Listagem de Colunas
        print("📋 Colunas encontradas:")
        for col in df.columns:
            print(f"   - {col}")
        print("-" * 40)

        # 4. Caça aos Embeddings
        # Lista de suspeitos comuns
        suspects = ['embedding', 'embeddings', 'feature', 'vector', 'embedding_sepaelv2']
        found = False
        
        for col in suspects:
            if col in df.columns:
                found = True
                print(f"🎯 Coluna candidata encontrada: '{col}'")
                
                # Pega o primeiro valor não nulo para ver o que tem dentro
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                
                if sample is None:
                    print("   ❌ A coluna existe, mas está VAZIA (todos os valores são null/None).")
                else:
                    print(f"   ✅ Tipo do dado: {type(sample)}")
                    if isinstance(sample, (list, np.ndarray)):
                        print(f"   📏 Dimensão do vetor: {len(sample)}")
                        print(f"   👀 Amostra: {sample[:5]}...")
                    else:
                        print(f"   ⚠️ Conteúdo: {sample} (Não parece um vetor)")
        
        if not found:
            print("❌ Nenhuma coluna de embedding foi encontrada com os nomes padrão.")
            
    except Exception as e:
        print(f"💥 Erro ao ler o arquivo: {e}")

if __name__ == "__main__":
    inspect_parquet(FILE_PATH)