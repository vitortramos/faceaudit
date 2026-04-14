import shutil
import time
import numpy as np
from pathlib import Path

# Imports do Sistema
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.transformator_config import TransformatorConfig
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.enums.whitening_style import WhiteningStyle # <--- IMPORTANTE
from faceaudit.pipelines.transformation_pipeline import TransformationPipeline

# --- 1. Gerador de Dados Fake ---
def mock_dataset(name, n_samples=100, n_classes=5, dim=512):
    matrix = np.random.rand(n_samples, dim).astype(np.float32)
    items = []
    for i in range(n_samples):
        identity = f"id_{i % n_classes}"
        items.append(EmbeddingInfo(
            embedding=matrix[i],
            face_hash=f"hash_{i}",
            image_path=f"/tmp/img_{i}.jpg",
            identity=identity,
            file_name=f"file_{i}.jpg",
            source="TestScript",
            origin="synthetic",
            quality_score=0.95
        ))
    return EmbeddingDataset(name=name, items=items)

def run_pipeline_test():
    # Limpeza de testes anteriores (ROBUSTA PARA WINDOWS)
    test_dir = Path("./temp_test_pipeline")
    if test_dir.exists():
        try:
            # ignore_errors=True evita o crash se o Windows bloquear o arquivo
            shutil.rmtree(test_dir, ignore_errors=True) 
            time.sleep(0.5) # Dá um respiro para o SO liberar o handle
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível limpar a pasta temporária: {e}")

    # Cria a pasta novamente se ela foi apagada ou se não existia
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)

    print("=== 🧪 INICIANDO TESTE DA CLASSE TransformationPipeline ===\n")

    # 1. Preparar Dados
    print("1. Gerando Datasets Sintéticos...")
    ds_train = mock_dataset("Treino_Source", n_samples=200, n_classes=10)
    ds_target = mock_dataset("Calib_Target", n_samples=50, n_classes=5)
    ds_test = mock_dataset("Teste_Final", n_samples=5)

    # 2. Criar Configuração
    config = TransformationPipelineConfig(
        name="Pipeline_Completo_V1",
        description="Teste de integração de todas as etapas",
        transformators=[
            TransformatorConfig(
                type=TransformationType.CORAL_TRANSFORMATION, 
                name="Step1_Coral", 
                params={"regularization": 1.0}
            ),
            TransformatorConfig(
                type=TransformationType.LDA_TRANSFORMATION, 
                name="Step2_LDA", 
                params={"n_components": 9}
            ),
            TransformatorConfig(
                type=TransformationType.WHITENING_TRANSFORMATION, 
                name="Step3_White", 
                # CORREÇÃO CRÍTICA: Usar o Enum, não string
                params={"style": WhiteningStyle.PCA} 
            ),
            TransformatorConfig(
                type=TransformationType.L2_NORMALIZATION, 
                name="Step4_L2"
            )
        ]
    )

    # 3. Instanciar Pipeline
    print("2. Instanciando Pipeline...")
    try:
        math_service = EmbeddingMathService()
        pipeline = TransformationPipeline(config, math_service)
    except Exception as e:
        print(f"❌ ERRO NA INSTANCIAÇÃO: {e}")
        # Imprime detalhes para debug se falhar aqui
        import traceback
        traceback.print_exc()
        return

    # 4. Executar FIT
    print("\n3. Executando .fit() (Observe os logs de cascata)...")
    try:
        pipeline.fit(
            source_dataset=ds_train, 
            target_dataset=ds_target, 
            save_path=test_dir
        )
        print("   ✅ Fit concluído sem erros.")
    except Exception as e:
        print(f"   ❌ ERRO NO FIT: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Executar TRANSFORM
    print("\n4. Executando .transform() no dataset de teste...")
    try:
        ds_result = pipeline.transform(ds_test)
        
        shape_ok = ds_result.matrix.shape == (5, 9)
        norm_val = np.linalg.norm(ds_result.matrix[0])
        norm_ok = abs(norm_val - 1.0) < 1e-4

        print(f"   Output Shape: {ds_result.matrix.shape} -> {'OK' if shape_ok else 'ERRO (Esperado 9)'}")
        print(f"   Norma Vetor 0: {norm_val:.4f} -> {'OK' if norm_ok else 'ERRO (Esperado 1.0)'}")

    except Exception as e:
        print(f"   ❌ ERRO NO TRANSFORM: {e}")
        return

    # 6. Teste de CARREGAMENTO
    print("\n5. Testando Persistência (Load de disco)...")
    try:
        pipeline_reloaded = TransformationPipeline(config, math_service)
        pipeline_reloaded.load(test_dir)
        
        ds_reloaded_result = pipeline_reloaded.transform(ds_test)
        are_equal = np.allclose(ds_result.matrix, ds_reloaded_result.matrix, atol=1e-6)
        
        if are_equal:
            print("   ✅ Pipeline recarregado produziu EXATAMENTE o mesmo resultado.")
        else:
            print("   ❌ ERRO: O pipeline recarregado gerou resultados diferentes.")
            
    except Exception as e:
        print(f"   ❌ ERRO NO LOAD: {e}")

    print("\n=== FIM DO TESTE ===")

if __name__ == "__main__":
    run_pipeline_test()