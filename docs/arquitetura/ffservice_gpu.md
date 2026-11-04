# FFService: GPU, CUDA no Windows e CoreML/ANE no Apple Silicon

> Conteúdo migrado do `CLAUDE.md` da raiz na reestruturação de 2026-11. As regras permanentes continuam resumidas lá; aqui fica o detalhe.

## CUDA no Windows

- `faceaudit/services/cuda_runtime_bootstrap.py` — `ensure_cuda_dlls_loaded()` — torna as DLLs do CUDA/cuDNN instaladas por **pip** visíveis ao ONNX Runtime no Windows. Sem isto, com `onnxruntime-gpu` instalado e GPU presente, TODA sessão caía para CPU em silêncio (medido no caminho real do FFService: 32,7 ms/imagem em CPU contra 8,4 ms em GPU, **3,9×**). Causa: o Python 3.8+ chama `SetDefaultDllDirectories` no Windows, que **retira o `PATH` da busca de dependências**, e os pacotes pip ficam em `site-packages/nvidia/*/bin`. `os.add_dll_directory` sozinho NÃO resolve (medido: erro 126 persiste); o que funciona é pré-carregar com `ctypes.WinDLL` na ordem de dependência — runtime do CUDA, álgebra, **cuDNN por último**. Idempotente, nunca levanta, no-op fora do Windows. `cuda_provider_disponivel()` CRIA uma sessão para testar, porque `get_available_providers()` lista o que foi compilado e não o que carrega — foi essa diferença que escondeu o problema. Busca em **duas fontes**, verificadas isoladamente: os pacotes `nvidia-*-cu12` do ambiente (preferidos — conjunto coerente entre si) e, na ausência deles, a instalação do sistema (`CUDA\v12*\bin` e `CUDNN\v*\bin\12*` sob `C:\Program Files`). Uma fonte só, nunca as duas: misturar cublas de uma com cuDNN de outra deixaria a combinação de versões à mercê de quem chega primeiro na memória. Chamado do bootstrap de `interface/run.py` e de `FFService.__init__` quando `use_gpu=True`. Usa `sysconfig` e não `os.__file__`: num venv o segundo aponta para a `Lib` do Python base.

O `FFService` (faceaudit/services/ff_service.py) carrega N instâncias de `ForensicFace` (uma por modelo de recognition declarado) e expõe APIs per-face e batch. Em macOS, quando `use_gpu=True`, o ORT é configurado pra usar o `CoreMLExecutionProvider`. Esse caminho tem várias arestas que estão consolidadas em um único monkey-patch (`_patch_onnxruntime_for_coreml_static_shapes`), descritas abaixo.

### Patch de configuração CoreML em 3 camadas

Aplicado em `ort.InferenceSession.__init__` no momento em que `FFService(use_gpu=True)` é construído. Idempotente; só faz sentido pra GPU. Camadas:

1. **`RequireStaticInputShapes=1`** em toda sessão. Conserto histórico do crash recorrente do detector `det_10g`:
   ```
   CoreML static output shape ({1,1,1,800,1}) and inferred shape ({3200,1}) have different ranks
   ```
   O CoreML pré-computa shapes em compile time, mas operadores pós-NMS do detector emitem tensores cujo número de elementos varia com a quantidade de candidatas que sobreviveram ao threshold. Esta flag instrui o CoreML a só capturar subgrafos cujas shapes são totalmente estáticas — nodes dinâmicos caem no CPU subgraph automaticamente dentro da mesma sessão.

2. **`ModelFormat=MLProgram`** em toda sessão CoreML. Backend mais novo (Core ML 5+, macOS 12+). Necessário pra encoders ViT (`sepaelv4` adaface_vit_webface4m, `sepaelv6` KPRPE):
   - **No `NeuralNetwork` backend** (default), encoders ViT com a override da camada 3 compilam e particionam bem (~87% cobertura) mas falham em **execução** com `Unable to compute the prediction using a neural network model ... (error code -1)`. Algum op de attention do ViT crasha em runtime no NN backend, mesmo passando o partitioning.
   - **`MLProgram`** tem implementação separada e mais recente, e roda sem esse erro. Bônus: preserva FP32 (ref: https://ym2132.github.io/ONNX_MLProgram_NN_exploration); `NeuralNetwork` faz cast implícito FP32→FP16 em hardware. Importante pra defensibilidade forense — embeddings biométricos não devem variar entre máquinas por causa de precisão silenciosa.

3. **`SessionOptions.add_free_dimension_override_by_name("batch_size", N)`** (e `"batch"`) **só em sessões com path contendo `/recognition/`**. Fixa a batch dim simbólica do ONNX em `N = self._batch_size` (vem do `config.ff.batch_size`) no compile time da sessão. Sem isso, ONNX exports recentes com `dynamic_axes={'input': {0: 'batch_size'}}` (v4/v6) tinham cobertura CoreML pífia: o partitioner não consegue resolver shapes intermediárias com batch dim simbólica e despeja quase tudo na CPU (3/1694 nodes no v6 sem override → 0.2% de CoreML). Com a override, a sessão é compilada como se batch fosse literal `N` e a cobertura sobe pra ~93%.

**Restrição da camada 3:** sessões com override só aceitam exatamente `N` amostras por chamada. Detector/attributes (`1k3d68`/`genderage`)/quality (`cr_fiqa_l`) **NÃO** recebem a override (path não bate o filtro) — continuam aceitando qualquer batch, incluindo batch=1 no path per-face legado. Recognition encoders no caminho per-face (`process_image` da forensicface) **quebram** com `use_gpu=True`, porque a forensicface alimenta batch=1 no encoder com override locked a batch=N. Mitigação atual: o caller deve usar o path align+batch (ver abaixo). Pra primeira extração crua em GPU, rodar com `use_gpu=False` temporariamente.

### Padding do batch parcial em `_run_engine_batch`

Como a sessão de recognition só aceita batch=N, o último flush de `_process_dataset_align_then_batch` (que tipicamente tem `n_real < batch_size`) precisaria falhar. O helper `_run_engine_batch` detecta `self._use_gpu=True and n_real < batch_size`, **preenche com zeros até `batch_size`**, roda a inferência e descarta as `pad_count` linhas excedentes nos arrays de retorno via helper local `_truncate_padding`. Em CPU (`use_gpu=False`) não há override, então não há padding.

### Por que o ANE pode ficar em 0% mesmo com 93% de cobertura CoreML

Limitação arquitetural conhecida do Apple Neural Engine para transformers, **não bug**. Algumas ops do ViT (variantes de `LayerNorm`, `GELU`, reshapes da multi-head attention) não estão implementadas no ANE. Quando o scheduler CoreML monta partições e uma partição contém qualquer op não suportada pelo ANE, a **partição inteira** migra pro GPU/Metal (que é mais permissivo). ViT tem blocos de attention espalhados pelo grafo, então quase tudo cai no GPU. GPU/Metal continua sendo aceleração real (muito acima de CPU); o ideal teórico de ANE-only puro é inalcançável pra essa arquitetura de modelo no Apple Silicon atual.

Validação via `asitop` (`pip install asitop`): se a barra GPU sobe e ANE fica em 0% durante os flushes de batch, está no padrão esperado. Throughput melhor que CPU-only confirma o ganho.

### Os 4 caminhos de `process_dataset`

Quatro modos selecionados em ordem de prioridade (decisão dentro de `process_dataset`):

| Caminho | Quando | Helper | tqdm desc |
|---|---|---|---|
| `is_aligned=True` (faces 112×112 em disco) | Dataset COPY_ALIGNED | `_process_dataset_aligned_batched` | `... (batch)` |
| Faces com `ff_info.aligned_face` populado em memória | Cache hit (raro pós-reload, `FFInfo.to_dict(exclude_heavy=True)` dropa `aligned_face`) | `_process_dataset_cached_aligned_batched` | `... (cache+batch)` |
| **Todas as faces têm `ff_info` populado, sem cache de aligned_face** | **Enrichment típico — usuário adicionando v4 a dataset que já tem v2** | `_process_dataset_align_then_batch` (alinha per-face via `oriented_alignment_service`, extrai embedding em batch) | `... (align+batch)` |
| Nada do acima | Primeira extração crua | Loop `process_face` per-face via `engine.process_image` | `Processando {name}` |

**Orientação**: os dois caminhos que detectam na foto original aceitam `rotation_attempts` e percorrem as quatro orientações — o per-face (`process_face`, que converte a caixa de volta ao sistema do arquivo) e o align+batch (via `oriented_alignment_service`, que tenta primeiro a orientação já registrada na face). Os dois caminhos de faces ALINHADAS não recebem rotações: o recorte 112×112 está de pé por construção e não há orientação a corrigir.

O fallback per-face usa `process_image` (que faz detect+align+extract em uma chamada) e por isso preserva contrato de metadata completo (gender/age/yaw/pitch/roll/det_score). Os 3 caminhos em batch só preservam `fiqa_score` + slots de embedding via MERGE — os demais campos vêm preservados de execução anterior. **Limitação documentada acima:** com `use_gpu=True`, o fallback per-face quebra nas sessões de recognition (override locked). Pra primeira extração crua em GPU, rodar com `use_gpu=False`.

---
