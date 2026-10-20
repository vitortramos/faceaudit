import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Callable, Sequence, Union, Tuple
from tqdm.auto import tqdm

# Import absoluto baseada na estrutura do pacote
from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models import FaceInfo, FFInfo, FaceDataset
from faceaudit.services.oriented_alignment_service import (
    alinhar_tentando_rotacoes,
    rotacao_registrada,
)
from faceaudit.services.image_rotation_service import (
    caixa_girada_para_original,
    descrever_rotacao,
    girar_imagem,
    normalizar_graus,
    pontos_girados_para_originais,
)

try:
    from forensicface.app import ForensicFace
except ImportError:
    ForensicFace = None


_PATCHED_MARKER = "_faceaudit_coreml_static_patched"
_RECOGNITION_BATCH_SIZE_ATTR = "_faceaudit_recognition_batch_size"
_SYMBOLIC_BATCH_DIM_NAMES: Tuple[str, ...] = ("batch_size", "batch")


def _patch_onnxruntime_for_coreml_static_shapes(batch_size: int = 1) -> None:
    """Configura o CoreMLExecutionProvider em três camadas:

    1. `RequireStaticInputShapes=1` em **toda** sessão criada — resolve
       crash recorrente do detector `det_10g`:

           CoreML static output shape ({1,1,1,800,1}) and inferred shape
           ({3200,1}) have different ranks

       Causa: o CoreML pré-computa shapes estaticamente no load time,
       mas operadores pós-NMS do detector emitem tensores cujo número
       de elementos varia com a quantidade de candidatas que
       sobreviveram ao threshold.

    2. `ModelFormat=MLProgram` em **toda** sessão CoreML — usa o backend
       MLProgram (Core ML 5+) em vez do NeuralNetwork (default).
       Motivações:
       - **Runtime error -1 do NeuralNetwork**: com a FreeDimensionOverride
         (camada 3) aplicada aos encoders ViT (sepaelv4/v6), o
         NeuralNetwork backend compila e particiona bem (~87% cobertura)
         mas falha **em execução** com "Unable to compute the prediction
         using a neural network model ... (error code -1)". Algum op do
         ViT triggera erro genérico de execução no ANE pelo NN backend.
         MLProgram tem implementação separada e roda sem esse problema.
       - **Precisão**: NN backend faz cast implícito FP32→FP16 em
         hardware; MLProgram preserva FP32 (ref:
         https://ym2132.github.io/ONNX_MLProgram_NN_exploration).
         Relevante pra defensibilidade forense — embeddings biométricos
         não devem variar entre máquinas por causa de precisão silenciosa.
       Custos: requer macOS 12+ (Core ML 5+), o que é amplamente atendido.

    3. `SessionOptions.add_free_dimension_override_by_name("batch_size", N)`
       (e `"batch"`) **só para sessões que carregam modelos em
       `recognition/`** — fixa a batch dim simbólica do ONNX em
       `batch_size = N` no compile time da sessão. Sem isso, ONNX
       exports recentes (sepaelv4 ViT, sepaelv6 KPRPE) com
       `dynamic_axes={'input': {0: 'batch_size'}}` produzem cobertura
       CoreML pífia (3/1694 nodes no v6 sem override): o partitioner
       não consegue resolver shapes intermediárias com batch dim
       simbólica e despeja quase tudo na CPU. Com a override, a sessão
       é compilada como se batch fosse literal `N`, CoreML consegue
       fixar todas as shapes (esperado ~87% de cobertura).

       **Restrição:** sessões com override só aceitam exatamente `N`
       amostras por chamada. O caller deve fazer padding do batch
       parcial final pra `N` (ver `_run_engine_batch`). Sessões de
       detection/attributes/quality NÃO recebem a override — continuam
       aceitando qualquer batch (incluindo batch=1 no path per-face
       legado).

    Aplicação: monkey-patch idempotente em `ort.InferenceSession.__init__`.
    O batch_size pode ser atualizado entre chamadas; sessões criadas
    DEPOIS de cada chamada veem o novo valor.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return

    # Atualiza o batch_size para sessões criadas a partir de agora.
    # Sessões já criadas mantêm o valor que vigorava no momento delas.
    setattr(ort.InferenceSession, _RECOGNITION_BATCH_SIZE_ATTR, max(1, int(batch_size)))

    if getattr(ort.InferenceSession, _PATCHED_MARKER, False):
        return  # __init__ já wrappado — não re-wrappa

    _original_init = ort.InferenceSession.__init__

    def _inject_coreml_opts(providers):
        if not providers:
            return providers
        out = []
        for p in providers:
            name = p if isinstance(p, str) else (p[0] if isinstance(p, tuple) else None)
            if name == "CoreMLExecutionProvider":
                base_opts = p[1] if isinstance(p, tuple) and len(p) > 1 else {}
                merged = {
                    **base_opts,
                    "RequireStaticInputShapes": "1",
                    # ModelFormat=MLProgram: backend mais novo (Core ML 5+).
                    # No NeuralNetwork backend, encoders ViT (sepaelv4/v6)
                    # com a FreeDimensionOverride aplicada falham em
                    # runtime com "Unable to compute the prediction
                    # using a neural network model ... (error code -1)"
                    # — algum op do ViT crasha em execução no ANE no
                    # NN backend, mesmo passando o partitioning.
                    # MLProgram tem implementação separada (mais
                    # recente) e roda. Bônus: MLProgram preserva FP32;
                    # NN backend faz cast implícito pra FP16 (ref:
                    # https://ym2132.github.io/ONNX_MLProgram_NN_exploration)
                    # — importante pra reprodutibilidade forense.
                    "ModelFormat": "MLProgram",
                }
                out.append(("CoreMLExecutionProvider", merged))
            else:
                out.append(p)
        return out

    def _is_recognition_model_path(path) -> bool:
        """True quando o path aponta pra um encoder dentro de
        `recognition/`. Filtra detector/attributes/quality, que continuam
        sem override (per-face compat)."""
        if path is None:
            return False
        try:
            if isinstance(path, (bytes, bytearray)):
                return False  # path como bytes = modelo inline, não rotula
            path_str = str(path)
        except Exception:
            return False
        return "/recognition/" in path_str or "\\recognition\\" in path_str

    def _override_session_options_for_recognition(path, sess_options):
        """Injeta `add_free_dimension_override_by_name` quando o path é
        recognition. Tenta nomes simbólicos comuns (`batch_size`,
        `batch`) — se o modelo não usa algum, a API retorna sem erro."""
        if not _is_recognition_model_path(path):
            return sess_options
        bs = int(getattr(ort.InferenceSession, _RECOGNITION_BATCH_SIZE_ATTR, 1))
        if bs < 1:
            return sess_options
        if sess_options is None:
            sess_options = ort.SessionOptions()
        for name in _SYMBOLIC_BATCH_DIM_NAMES:
            try:
                sess_options.add_free_dimension_override_by_name(name, bs)
            except Exception:
                # API ausente (ORT muito antigo) ou nome não presente
                # no modelo — segue silenciosamente.
                pass
        return sess_options

    def _patched_init(self, *args, **kwargs):
        # Camada 1: provider options (RequireStaticInputShapes=1).
        # `providers` pode vir como kwarg ou como 2º arg posicional.
        if "providers" in kwargs:
            kwargs["providers"] = _inject_coreml_opts(kwargs["providers"])
        elif len(args) >= 2:
            args = (args[0], _inject_coreml_opts(args[1]), *args[2:])

        # Camada 2: sess_options com FreeDimensionOverride pra recognition.
        # `path_or_bytes` é args[0] ou kwargs['path_or_bytes'];
        # `sess_options` é tipicamente kwarg em forensicface (verificado).
        path = args[0] if args else kwargs.get("path_or_bytes")
        sess_options = kwargs.get("sess_options")
        new_sess_options = _override_session_options_for_recognition(
            path, sess_options,
        )
        if new_sess_options is not sess_options:
            kwargs["sess_options"] = new_sess_options

        return _original_init(self, *args, **kwargs)

    ort.InferenceSession.__init__ = _patched_init
    setattr(ort.InferenceSession, _PATCHED_MARKER, True)


# Conjunto canônico de chaves do FFInfo a serem mergeadas quando um run
# do encoder retorna novos valores. Sem incluir TODOS os slots de embedding
# do enum, modelos novos (v3/v5/v6) entregavam embeddings que sumiam no
# merge — só v2 e v4 eram propagados.
_FFINFO_BASE_KEYS: Tuple[str, ...] = (
    "ipd", "fiqa_score", "det_score", "gender", "age",
    "yaw", "pitch", "roll", "bbox", "keypoints",
    "aligned_face",
)
_FFINFO_MERGE_KEYS: Tuple[str, ...] = (
    _FFINFO_BASE_KEYS + tuple(m.ffinfo_slot for m in SepaelModel)
)
_FFINFO_EMBEDDING_SLOTS: Tuple[str, ...] = tuple(
    m.ffinfo_slot for m in SepaelModel
)

# Chave em FaceInfo.extra_data onde o fluxo de vídeo persiste os 5 keypoints
# alinhados (JSON, sistema 112×112) — exigidos pelos modelos KPRPE (sepaelv6).
ALIGNED_KEYPOINTS_EXTRA_KEY = "aligned_keypoints_112"


def _aligned_keypoints_from_face(face) -> Optional[np.ndarray]:
    """5 keypoints alinhados (sistema 112×112) de uma FaceInfo `(5, 2)`, ou None.
    Fonte: `ff_info.keypoints` (em memória, fluxo ponta-a-ponta) ou
    `extra_data['aligned_keypoints_112']` (JSON persistido no parquet, para
    re-enriquecimento). Necessário para os modelos KPRPE (sepaelv6)."""
    # Lê SÓ a casa própria dos pontos alinhados. `ff_info.keypoints` é, por
    # invariante, o sistema da imagem que o dataset guarda — num dataset de
    # recortes vindo de fotos são as coordenadas da foto original, e
    # alimentar o KPRPE com elas produzia embedding errado sem erro
    # aparente. Seguir sem keypoints, que o encoder sabe recusar, é melhor
    # que seguir com os errados.
    raw = (getattr(face, "extra_data", None) or {}).get(ALIGNED_KEYPOINTS_EXTRA_KEY)
    if raw:
        try:
            arr = np.asarray(json.loads(raw), dtype=np.float32)
            return arr if arr.shape == (5, 2) else None
        except Exception:
            return None
    return None


def _tem_alinhada_em_cache(face) -> bool:
    """A face já traz o recorte alinhado 112×112 em memória (RGB)."""
    ff = getattr(face, "ff_info", None)
    alinhada = getattr(ff, "aligned_face", None) if ff is not None else None
    return alinhada is not None and getattr(alinhada, "shape", None) == (112, 112, 3)


def _sub_dataset(dataset: FaceDataset, faces: List[FaceInfo]) -> FaceDataset:
    """Recorte do dataset com as MESMAS FaceInfo (mutações propagam)."""
    sub = FaceDataset(
        name=dataset.name, faces=faces,
        path_mode=dataset.path_mode, path_root=dataset.path_root,
    )
    sub._parquet_dir = getattr(dataset, "_parquet_dir", None)
    sub.is_aligned = bool(getattr(dataset, "is_aligned", False))
    return sub


class FFService:
    """
    Serviço de interface para a biblioteca ForensicFace.

    Responsável por detecção facial, alinhamento, extração de landmarks e
    cálculo de embeddings (vetores biométricos), além de utilitários como
    construção de mosaicos e extração de faces de vídeo.

    Suporta múltiplos modelos carregados em paralelo (uma instância
    independente do ForensicFace por modelo). Cada chamada que precisa de
    embedding aceita um parâmetro `model` opcional que escolhe entre os
    modelos carregados — sem `model`, usa o primeiro declarado.

    Notas de design:
    - Carregar 2 modelos no mesmo processo dobra a memória de GPU mas
      mantém o custo de inferência por chamada igual ao de um modelo só
      (o modelo não escolhido fica ocioso). Isso evita o custo da
      configuração `models=[...]` nativa do ForensicFace, que sempre roda
      todos os encoders carregados a cada `process_image`.
    - Operações que não dependem de embedding (mosaico, extração de
      vídeo) usam o engine default — qualquer um serve.
    """

    def __init__(
        self,
        models: Union[str, List[str], None] = None,
        model: Optional[str] = None,  # legado, alias para `models`
        det_size: int = 320,
        use_gpu: bool = False,
        extended: bool = True,
        batch_size: int = 32,
        det_thresh: float = 0.5,
    ):
        if ForensicFace is None:
            raise ImportError("A biblioteca 'forensicface' não foi encontrada.")

        # Resolve a lista de modelos a carregar:
        #   - `models` tem prioridade quando fornecido (forma nova)
        #   - `model` é mantido como kwarg de retrocompat (containers.py)
        #   - default = SepaelModel.default().ff_name
        resolved: List[str]
        if models is not None:
            resolved = [models] if isinstance(models, str) else list(models)
        elif model is not None:
            resolved = [model]
        else:
            resolved = [SepaelModel.default().ff_name]

        if not resolved:
            raise ValueError("FFService: ao menos um modelo deve ser declarado.")

        # Normaliza pra minúsculas e descarta duplicatas preservando ordem.
        seen: set = set()
        normalized: List[str] = []
        for m in resolved:
            key = m.lower()
            if key not in seen:
                seen.add(key)
                normalized.append(key)

        # Guarda config para o helper `matches_config` decidir se um
        # FFService preexistente atende a um pedido sem precisar rebuild.
        self._det_size = det_size
        self._use_gpu = use_gpu
        self._extended = extended
        # Limiar do detector (SCRFD). ↓ pega faces de baixa confiança
        # (pequenas/distantes em vigilância), ao custo de mais falsos positivos.
        self._det_thresh = float(det_thresh)
        # Batch size pra inferência paralela de embedding/FIQA. Sanitizado
        # pra inteiro >= 1; 1 desliga batching (loop legado).
        try:
            self._batch_size: int = max(1, int(batch_size))
        except (TypeError, ValueError):
            self._batch_size = 32

        # Pré-check de disponibilidade em disco: falha cedo com mensagem
        # amigável quando o usuário pede um modelo cuja pasta não tem os
        # ONNX. Sem essa checagem, o erro estoura lá dentro do
        # `ForensicFace.__init__` como `FileNotFoundError` num arquivo
        # interno do encoder (ex: `adaface_vit_webface4m.onnx`), com
        # mensagem que não menciona o `ff_name` que o usuário pediu.
        missing: List[str] = []
        recognized: List[SepaelModel] = []
        for name in normalized:
            member = SepaelModel.from_ff_name(name)
            if member is None:
                # Não reconhecido pelo enum — assume que o usuário sabe
                # o que está fazendo (ex: modelo experimental). Não bloqueia.
                continue
            recognized.append(member)
            if not member.installed_on_disk():
                missing.append(member.label_pt)
        if missing:
            available_now = [m.label_pt for m in SepaelModel.available_on_disk()]
            available_str = ", ".join(available_now) if available_now else "(nenhum)"
            raise FileNotFoundError(
                f"Modelo(s) não instalado(s) em ~/.forensicface/models/: "
                f"{', '.join(missing)}. Disponíveis na máquina: {available_str}. "
                f"Instale baixando do repositório do ForensicFace ou "
                f"escolha outro modelo nas Configurações."
            )

        # CoreML setup em duas camadas (ver docstring de
        # `_patch_onnxruntime_for_coreml_static_shapes`):
        # - `RequireStaticInputShapes=1` em toda sessão: conserto do
        #   crash do det_10g (shape rank mismatch pós-NMS).
        # - `FreeDimensionOverride('batch_size', N)` só em sessões
        #   `recognition/`: re-estatiza a batch dim simbólica do ONNX
        #   (sepaelv4/v6 exportados com `dynamic_axes`) pra CoreML
        #   conseguir capturar matmul/attention do ViT e rotear pro
        #   Neural Engine. Sessões com override só aceitam batch
        #   exato = N; `_run_engine_batch` faz padding do batch parcial
        #   final pra cobrir esse requisito.
        # Patch é monkey-patch no `InferenceSession.__init__`,
        # idempotente, mas o `batch_size` é atualizado a cada chamada
        # (sessões criadas DEPOIS veem o novo valor).
        if use_gpu:
            # No Windows, as DLLs do CUDA/cuDNN instaladas por pip não são
            # encontradas pelo carregador (o Python 3.8+ tirou o PATH da
            # busca de dependências) e TODA sessão caía para CPU em
            # silêncio. Tem de rodar ANTES da primeira sessão.
            from faceaudit.services.cuda_runtime_bootstrap import (
                ensure_cuda_dlls_loaded,
            )
            ensure_cuda_dlls_loaded(verbose=True)
            _patch_onnxruntime_for_coreml_static_shapes(
                batch_size=self._batch_size,
            )

        # Carrega uma ForensicFace independente por modelo.
        # Cada engine fica com `models=[<um>]`, então só esse encoder roda
        # quando aquele engine é chamado. `extended=False` pula FIQA,
        # gender, age, pose — útil para datasets enxutos.
        # `concat_embeddings=True` garante que o ForensicFace devolve a
        # chave `embedding` no dict (com 1 modelo, é equivalente a um
        # único embedding 512D — concat de uma lista de 1 elemento).
        # Se ficasse False, devolveria `embedding_sepaelv2` etc. e o
        # `process_face` abaixo (que lê `embedding`) não acharia o vetor.
        self._engines: Dict[str, ForensicFace] = {}
        for name in normalized:
            self._engines[name] = ForensicFace(
                models=[name],
                det_size=det_size,
                use_gpu=use_gpu,
                extended=extended,
                det_thresh=det_thresh,
                concat_embeddings=True,
            )

        self._default_model: str = normalized[0]
        self._loaded_models: Tuple[str, ...] = tuple(normalized)

        # Atributo legado preservado para callers que inspecionam
        # `self.model_name`. Mapeia para o modelo default.
        self.model_name: str = self._default_model
        # Atributo legado preservado para callers que acessam `self.app`
        # (ex: chamada direta do compare nativo). Mapeia para o engine
        # default. Múltiplos modelos? Use `_resolve_engine(model)`.
        self.app: ForensicFace = self._engines[self._default_model]

        # Detecta fallback silencioso CPU quando `use_gpu=True` foi
        # solicitado mas o onnxruntime instalado / drivers não suportam
        # nenhum accelerator. Mensagem fica no stderr (caller que rodar
        # pelo terminal vê). Sem isso, usuário pede GPU, vê o app rodar
        # lento e não entende — `session_providers` no log padrão do
        # ForensicFace já mostra o efetivo, mas ninguém lê esse detalhe.
        if use_gpu:
            self._warn_if_cpu_fallback()

    _ACCELERATED_PROVIDERS = frozenset({
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "ROCMExecutionProvider",
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",
    })

    def _warn_if_cpu_fallback(self) -> None:
        """Quando `use_gpu=True` mas os providers efetivos das sessions
        são só CPU, emite warning no stderr. Não bloqueia execução —
        FFService continua funcionando em CPU."""
        effective: set = set()
        for engine in self._engines.values():
            for sess in getattr(engine, "rec_inference_sessions", []) or []:
                try:
                    effective.update(sess.get_providers())
                except Exception:
                    pass
        if effective and not (effective & self._ACCELERATED_PROVIDERS):
            print(
                f"⚠️ [FFService] use_gpu=True solicitado, mas providers "
                f"efetivos são {sorted(effective)}. Verifique instalação "
                f"de drivers (CUDA/CoreML) e do pacote onnxruntime "
                f"apropriado. Processamento continuará em CPU (mais lento)."
            )

    # ─────────────────────────────────────────────────────────────────────
    # Resolução de engine por modelo
    # ─────────────────────────────────────────────────────────────────────

    @property
    def loaded_models(self) -> Tuple[str, ...]:
        """Tupla com os nomes dos modelos carregados, na ordem de declaração."""
        return self._loaded_models

    @property
    def default_model(self) -> str:
        """Nome do modelo default (primeiro declarado no construtor)."""
        return self._default_model

    def matches_config(
        self,
        models: List[str],
        det_size: int,
        extended: bool,
    ) -> bool:
        """Indica se este FFService já está pronto pra atender a uma
        execução com esses parâmetros — usado pelo
        `FeatureExtractionPipeline` pra evitar rebuild quando o Singleton
        do container já cobre o pedido.

        `use_gpu` não entra na comparação porque é capability da máquina,
        não da execução: se a config global diz GPU e a UI diz outra
        coisa, prevalece o ambiente (mudança requer restart).
        """
        wanted = {m.lower() for m in (models or [])}
        loaded = set(self._loaded_models)
        # Modelos pedidos têm que estar todos carregados; det_size e
        # extended têm que bater exatamente.
        return (
            wanted.issubset(loaded)
            and self._det_size == int(det_size)
            and bool(self._extended) == bool(extended)
        )

    def _resolve_engine(self, model: Optional[str]) -> Tuple[str, ForensicFace]:
        """
        Resolve `model` (ou None) num par (nome_normalizado, engine).

        - None → default
        - nome carregado → esse engine
        - nome não carregado → ValueError com a lista do que está disponível
        """
        if model is None:
            name = self._default_model
        else:
            name = model.lower()
            if name not in self._engines:
                raise ValueError(
                    f"Modelo '{model}' não está carregado. "
                    f"Modelos disponíveis: {list(self._engines.keys())}. "
                    f"Adicione-o ao parâmetro `models=[...]` na construção do FFService."
                )
        return name, self._engines[name]

    @staticmethod
    def _normalize_process_image_result(res) -> Optional[dict]:
        """Adapter pra `engine.process_image(..., single_face=True)`.

        Forward-compat com a transição do retorno anunciada na v0.7.0:
        atualmente devolve `dict | []`; uma futura versão vai devolver
        `list[dict] | []`. Esse helper unifica os dois formatos num
        `Optional[dict]` (`None` quando não há face).

        Sem isso, os callsites que fazem `res.get(...)` ou
        `for k in (...): res[k]` quebrariam silenciosamente quando a lib
        for atualizada.
        """
        if not res:
            return None
        if isinstance(res, dict):
            return res
        if isinstance(res, (list, tuple)) and res:
            first = res[0]
            return first if isinstance(first, dict) else None
        return None

    @staticmethod
    def _slot_for_model(model_name: str) -> str:
        """
        Retorna o atributo de FFInfo onde o embedding do `model_name` deve
        ser gravado. Modelos desconhecidos caem no slot do default
        (`SepaelModel.default()` → sepaelv2) — mantém o comportamento
        histórico do FFService antigo.
        """
        return SepaelModel.ffinfo_slot_for(model_name)

    # ─────────────────────────────────────────────────────────────────────
    # Extração de embeddings
    # ─────────────────────────────────────────────────────────────────────

    def get_embedding_from_array(
        self,
        img_array: np.ndarray,
        model: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Extrai o embedding de uma imagem em memória (ndarray BGR).

        Args:
            img_array: imagem BGR no formato OpenCV.
            model: nome do modelo a usar (None = default declarado).

        Returns:
            ndarray 1D do embedding ou None em caso de falha.
        """
        try:
            _, engine = self._resolve_engine(model)
            res = self._normalize_process_image_result(
                engine.process_image(img_array)
            )
            if res and "embedding" in res:
                return res["embedding"]
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────────────────
    # PR L3 — Alinhamento isolado e extração a partir de face já alinhada
    # ─────────────────────────────────────────────────────────────────────

    def align_face_from_array(
        self,
        bgr_image: np.ndarray,
        *,
        model: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Variante de `align_face_only` que recebe ndarray direto
        em vez de `FaceInfo` (PR I3).

        Útil em pipelines in-memory onde a imagem já está carregada
        e não há FaceInfo associado — caso típico: editor de pipeline
        de imagens calculando scores de qualidade da imagem
        processada (que existe só em RAM).

        Internamente delega ao `engine.detect_and_align(bgr_image, single_face=True)`
        do `forensicface` (>= 0.7.1), que faz `detect_faces` →
        seleção da melhor face → `norm_crop` + `_align_keypoints` numa única
        chamada e devolve os keypoints alinhados (no sistema de
        coordenadas 112×112) prontos pra serem repassados na extração
        com modelos KPRPE (sepaelv6). Antes do PR atual, esse pipeline
        era reconstruído manualmente e os aligned_keypoints eram
        descartados — quebrando a extração com v6.

        Args:
            bgr_image: ndarray BGR (H, W, 3) uint8.
            model: nome do modelo (None = default).

        Returns:
            Tupla `(bgr_aligned, aligned_keypoints)`:
            - `bgr_aligned`: ndarray BGR (112, 112, 3) uint8.
            - `aligned_keypoints`: ndarray float (5, 2) com os 5 pontos
              (olho esq., olho dir., nariz, canto esq. da boca,
              canto dir. da boca) em coordenadas do crop 112×112.
              Necessário pra modelos KPRPE (sepaelv6); ignorado pelos
              demais.
            Ambos `None` se nenhuma face foi detectada / erro.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None, None
        try:
            _, engine = self._resolve_engine(model)
            item = engine.detect_and_align(bgr_image, single_face=True)
            if not item:
                return None, None
            # detect_and_align retorna `aligned_face` em RGB; convertemos
            # pra BGR pra preservar o contrato anterior (callers consomem BGR).
            rgb_aligned = item.get("aligned_face")
            aligned_kps = item.get("aligned_keypoints")
            if rgb_aligned is None:
                return None, None
            bgr_aligned = cv2.cvtColor(rgb_aligned, cv2.COLOR_RGB2BGR)
            return bgr_aligned, aligned_kps
        except Exception as e:
            print(f"Error in align_face_from_array: {e}")
            return None, None

    def align_face_only(
        self,
        face_info: FaceInfo,
        *,
        dataset: Optional[FaceDataset] = None,
        model: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Detecta + alinha SEM extrair embedding (~30-50% mais rápido
        que `process_face`).

        Pipeline interno: `_safe_read_image` → `engine.detect_and_align`.
        Pula o `_compute_embeddings` que é a etapa cara (ONNX recognition).

        Útil pra COPY_ALIGNED em datasets grandes onde só queremos
        guardar a face alinhada como PNG, sem rodar embedding.

        Args:
            face_info: face com `image_path` apontando pra imagem
                original (foto inteira com background).
            dataset: opcional — quando passado, resolve `image_path`
                via `dataset.resolve_image_path` (suporta paths
                relativos pós-localização).
            model: nome do modelo a usar (None = default). O modelo só
                influencia qual `det_size` é aplicado; alinhamento em
                si é determinístico.

        Returns:
            ndarray BGR (112, 112, 3) uint8 da face alinhada, ou None
            se nenhuma face foi detectada / arquivo ilegível / erro.
        """
        image_path = (
            str(dataset.resolve_image_path(face_info))
            if dataset is not None
            else face_info.image_path
        )
        img_array = self._safe_read_image(image_path)
        if img_array is None:
            return None
        try:
            _, engine = self._resolve_engine(model)
            item = engine.detect_and_align(img_array, single_face=True)
            if not item:
                return None
            # detect_and_align retorna `aligned_face` em RGB — convertemos
            # pra BGR pra preservar o contrato anterior. Keypoints alinhados
            # são descartados aqui (esse método é usado em COPY_ALIGNED onde
            # só queremos materializar o crop em disco; KPRPE precisa de
            # detect novo na re-extração).
            rgb_aligned = item.get("aligned_face")
            if rgb_aligned is None:
                return None
            return cv2.cvtColor(rgb_aligned, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error in align_face_only for {image_path}: {e}")
            return None

    def extract_embeddings_from_aligned_face(
        self,
        bgr_aligned_face: np.ndarray,
        *,
        aligned_keypoints: Optional[np.ndarray] = None,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> Optional[FFInfo]:
        """Extrai embeddings + FIQA de uma face JÁ alinhada (BGR
        (112, 112, 3) uint8). Pula detecção e alinhamento.

        Útil quando o input já é uma face alinhada — caso típico:
            - Pipeline de imagens (futuro) aplicando processamento
              sobre faces alinhadas e querendo medir o impacto no
              embedding.
            - Enrichment num dataset COPY_ALIGNED (combinado via
              `process_face_aligned` que faz read + extract).

        Suporta múltiplos modelos via `models=[...]`. Os embeddings
        vão pra slots correspondentes em FFInfo
        (`embedding_sepaelv2`, `embedding_sepaelv4`, etc.) — mesmo
        layout do `process_face`. Não preenche bbox / keypoints /
        aligned_face — input já é a face alinhada.

        Internamente: BGR → RGB → `engine.process_aligned_face_image`
        (a API original do ForensicFace recebe RGB e converte de
        volta pra BGR antes de rodar o embedding model). Custo da
        conversão dupla é desprezível (~2 × 37 KB swap).

        Args:
            bgr_aligned_face: ndarray BGR (112, 112, 3) uint8.
            model: modelo único (default = `_default_model`). Ignorado
                se `models` for passado.
            models: lista de modelos a extrair em uma só passada por
                face. Cada modelo preenche o slot correspondente em
                FFInfo. FIQA score vem do primeiro engine consultado.

        Returns:
            FFInfo com embeddings + fiqa_score, ou None se a entrada
            for inválida / todos os modelos falharem.
        """
        if bgr_aligned_face is None or bgr_aligned_face.size == 0:
            return None

        target_models: List[str]
        if models:
            target_models = list(models)
        elif model is not None:
            target_models = [model]
        else:
            target_models = [self._default_model]

        # BGR → RGB (process_aligned_face_image espera RGB)
        rgb_aligned_face = bgr_aligned_face[..., ::-1].copy()

        ff_data: Optional[FFInfo] = None
        for m in target_models:
            try:
                model_name, engine = self._resolve_engine(m)
                # `aligned_keypoints` é obrigatório pra modelos KPRPE
                # (sepaelv6) no forensicface >= 0.7.0. Pra modelos não-KPRPE
                # (v2/v3/v4/v5) é ignorado pela lib; passar None é seguro.
                res = engine.process_aligned_face_image(
                    rgb_aligned_face, keypoints=aligned_keypoints,
                )
                if not res:
                    continue
                # Cada engine carrega só 1 modelo internamente com
                # concat_embeddings=True (ver __init__), então a chave
                # do dict é sempre "embedding" — não "embedding_<model>".
                emb = res.get("embedding")
                if emb is None:
                    continue
                if ff_data is None:
                    # FIQA score só vem do primeiro engine consultado
                    # (todos retornam o mesmo valor — é função do FIQA
                    # network, não do embedding model).
                    ff_data = FFInfo(fiqa_score=res.get("fiqa_score"))
                setattr(ff_data, self._slot_for_model(model_name), emb)
            except Exception as e:
                print(
                    f"Error in extract_embeddings_from_aligned_face "
                    f"(model={m}): {e}"
                )

        return ff_data

    def extract_embeddings_from_aligned_faces_batch(
        self,
        bgr_aligned_faces: np.ndarray,
        *,
        models: Optional[List[str]] = None,
        aligned_keypoints_batch: Optional[np.ndarray] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Optional[FFInfo]]:
        """Extrai embeddings + FIQA de **N faces alinhadas em paralelo**.

        Delega à lib `forensicface` (>= 0.6.0) via
        `engine._try_compute_embeddings_batch` — herda OOM recovery por
        halving recursivo e suporte automático a sepaelv6 (KPRPE) quando
        `aligned_keypoints_batch` é fornecido. Versões anteriores da lib
        caem em fallback inline (loop manual via `rec.run`).

        Args:
            bgr_aligned_faces: array `(N, 112, 112, 3)` BGR uint8.
            models: lista de modelos a extrair. Default = `_default_model`.
            aligned_keypoints_batch: array `(N, 5, 2)` opcional. Necessário
                quando algum modelo carregado for KPRPE (`sepaelv6`); ignorado
                pelos modelos ResNet/ViT.

        Returns:
            Lista de length N. Posições com erro são `None`. FIQA score
            é populado a partir do primeiro engine (todos retornam o
            mesmo — é função do FIQA network, não do encoder).
        """
        if bgr_aligned_faces is None or bgr_aligned_faces.size == 0:
            return []
        n = int(bgr_aligned_faces.shape[0])
        if n == 0:
            return []
        if bgr_aligned_faces.shape[1:] != (112, 112, 3):
            raise ValueError(
                f"extract_embeddings_from_aligned_faces_batch espera shape "
                f"(N, 112, 112, 3); recebeu {bgr_aligned_faces.shape}."
            )

        target_models: List[str]
        if models:
            target_models = list(models)
        else:
            target_models = [self._default_model]

        results: List[Optional[FFInfo]] = [None] * n
        fiqa_run_done = False

        for m in target_models:
            try:
                model_name, engine = self._resolve_engine(m)
            except Exception as e:
                msg = (
                    f"[FFService.batch] resolve_engine falhou pra '{m}': {e}"
                )
                (error_callback or print)(msg)
                continue

            embeddings, fiqa_scores = self._run_engine_batch(
                engine=engine,
                model_name=model_name,
                bgr_aligned_faces=bgr_aligned_faces,
                aligned_keypoints_batch=aligned_keypoints_batch,
                error_callback=error_callback,
            )
            if embeddings is None:
                continue

            # FIQA é compartilhado entre engines (mesmo CR-FIQA em todos);
            # guardamos só uma vez pra não sobrescrever após primeiro engine.
            if fiqa_run_done:
                fiqa_scores = None
            elif fiqa_scores is not None:
                fiqa_run_done = True

            slot = self._slot_for_model(model_name)
            for i in range(n):
                ff = results[i]
                if ff is None:
                    fiqa_val = (
                        float(fiqa_scores[i])
                        if fiqa_scores is not None else None
                    )
                    ff = FFInfo(fiqa_score=fiqa_val)
                    results[i] = ff
                setattr(ff, slot, embeddings[i])

        return results

    def _run_engine_batch(
        self,
        *,
        engine,
        model_name: str,
        bgr_aligned_faces: np.ndarray,
        aligned_keypoints_batch: Optional[np.ndarray],
        error_callback: Optional[Callable[[str], None]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Roda o batch em um engine ForensicFace específico.

        Caminhos suportados em ordem de preferência:

        1. **v0.7.x**: `engine._recognition_runner().try_compute_batch(...)`
           — método do `RecognitionRunner` (recognition.py) com OOM auto-recovery
           (halving recursivo até batch_size=1) e suporte nativo a KPRPE
           (`aligned_keypoints_batch`).

        2. **v0.6.x (legado)**: `engine._try_compute_embeddings_batch(...)` —
           mesmo método antes da reorganização em `RecognitionRunner`.

        3. **v0.5.x (legado)**: inferência inline via `rec_inference_sessions[0]`.
           Sem OOM recovery, sem suporte a KPRPE.

        Retorna `(embeddings, fiqa_scores)`. Embeddings shape `(N, dim)`;
        fiqa_scores shape `(N,)` ou None quando `extended=False`. Ambos
        viram None na tupla quando a inferência falha (erro reportado
        via `error_callback`).

        Padding pra batch parcial: quando `use_gpu=True`, as sessões de
        recognition são carregadas com `FreeDimensionOverride('batch_size',
        self._batch_size)` (ver `_patch_onnxruntime_for_coreml_static_shapes`)
        — necessário pra ANE conseguir capturar o ViT. Sessões com override
        SÓ aceitam batch exato. Se o caller mandar N < batch_size (caso
        típico: último flush do dataset), preenchemos com zeros até
        batch_size, rodamos a inferência e descartamos as posições
        excedentes nos arrays de retorno.
        """
        n_real = int(bgr_aligned_faces.shape[0])
        pad_count = 0
        if self._use_gpu and n_real > 0 and n_real < self._batch_size:
            pad_count = self._batch_size - n_real
            pad_faces = np.zeros(
                (pad_count, *bgr_aligned_faces.shape[1:]),
                dtype=bgr_aligned_faces.dtype,
            )
            bgr_aligned_faces = np.concatenate(
                [bgr_aligned_faces, pad_faces], axis=0,
            )
            if aligned_keypoints_batch is not None:
                pad_kps = np.zeros(
                    (pad_count, *aligned_keypoints_batch.shape[1:]),
                    dtype=aligned_keypoints_batch.dtype,
                )
                aligned_keypoints_batch = np.concatenate(
                    [aligned_keypoints_batch, pad_kps], axis=0,
                )

        def _truncate_padding(embeddings, fiqa_scores):
            """Descarta linhas correspondentes ao padding nos arrays de
            retorno. No-op quando `pad_count == 0`."""
            if pad_count == 0:
                return embeddings, fiqa_scores
            if embeddings is not None and hasattr(embeddings, "__getitem__"):
                try:
                    embeddings = embeddings[:n_real]
                except Exception:
                    pass
            if fiqa_scores is not None and hasattr(fiqa_scores, "__getitem__"):
                try:
                    fiqa_scores = fiqa_scores[:n_real]
                except Exception:
                    pass
            return embeddings, fiqa_scores

        try:
            # Captura `warnings.warn(...)` emitidos pela lib durante o batch
            # — o helper de OOM avisa quando acontece auto-recovery (halving).
            import warnings as _warnings

            # Caminho moderno (v0.7+): RecognitionRunner.try_compute_batch
            recognition_runner = getattr(engine, "_recognition_runner", None)
            if callable(recognition_runner):
                runner = recognition_runner()
                if hasattr(runner, "try_compute_batch"):
                    with _warnings.catch_warnings(record=True) as caught:
                        _warnings.simplefilter("always")
                        embeddings, fiqa_scores = runner.try_compute_batch(
                            bgr_aligned_faces,
                            aligned_keypoints_batch=aligned_keypoints_batch,
                        )
                    for w in caught:
                        (error_callback or print)(f"⚠ [forensicface] {w.message}")
                    if isinstance(embeddings, list) and len(embeddings) == 1:
                        embeddings = embeddings[0]
                    return _truncate_padding(embeddings, fiqa_scores)

            # Legado v0.6.x: método privado no próprio engine
            if hasattr(engine, "_try_compute_embeddings_batch"):
                kwargs = {}
                if aligned_keypoints_batch is not None:
                    kwargs["aligned_keypoints_batch"] = aligned_keypoints_batch
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter("always")
                    embeddings, fiqa_scores = engine._try_compute_embeddings_batch(
                        bgr_aligned_faces, **kwargs,
                    )
                for w in caught:
                    (error_callback or print)(f"⚠ [forensicface] {w.message}")
                if isinstance(embeddings, list) and len(embeddings) == 1:
                    embeddings = embeddings[0]
                return _truncate_padding(embeddings, fiqa_scores)

            # Fallback lib 0.5.x: roda direto via rec_inference_sessions.
            batched = bgr_aligned_faces.astype(np.float32)
            batched = ((batched / 255.0) - 0.5) / 0.5
            batched_input = batched.transpose(0, 3, 1, 2).copy()

            rec = engine.rec_inference_sessions[0]
            rec_out = rec.run(
                None, {rec.get_inputs()[0].name: batched_input},
            )
            if len(rec_out) == 2:
                embeddings = rec_out[0] * rec_out[1]
            else:
                embeddings = rec_out[0]

            fiqa_scores = None
            if self._extended:
                ort_fiqa = getattr(engine, "ort_fiqa", None)
                if ort_fiqa is not None:
                    try:
                        _, fiqa_out = ort_fiqa.run(
                            None,
                            {ort_fiqa.get_inputs()[0].name: batched_input},
                        )
                        fiqa_scores = np.asarray(fiqa_out).reshape(-1)
                    except Exception as e:
                        (error_callback or print)(
                            f"[FFService.batch] inferência FIQA falhou: {e}"
                        )

            return _truncate_padding(embeddings, fiqa_scores)

        except Exception as e:
            msg = (
                f"[FFService.batch] inferência batch falhou pra "
                f"'{model_name}': {e}"
            )
            (error_callback or print)(msg)
            return None, None

    def process_face_aligned(
        self,
        face_info: FaceInfo,
        *,
        dataset: Optional[FaceDataset] = None,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> FaceInfo:
        """Processa uma face quando o `image_path` aponta pra imagem
        JÁ alinhada (caso típico: dataset pós-COPY_ALIGNED).

        Pula detecção e alinhamento — só lê o arquivo e delega ao
        `extract_embeddings_from_aligned_face`. ~50% mais rápido que
        `process_face` e mais robusto: não falha tentando detectar
        uma face dentro de uma imagem 112×112 que JÁ é só a face.

        Args:
            face_info: face com `image_path` apontando pra
                PNG/JPG (112, 112, 3).
            dataset: opcional — pra resolver paths em RELATIVE.
            model/models: igual `process_face`.

        Returns:
            FaceInfo com `ff_info` populado (embeddings + fiqa_score).
            Se a leitura falhar ou os modelos não retornarem nada,
            retorna o face_info inalterado.
        """
        image_path = (
            str(dataset.resolve_image_path(face_info))
            if dataset is not None
            else face_info.image_path
        )
        bgr_aligned_face = self._safe_read_image(image_path)
        if bgr_aligned_face is None:
            return face_info

        new_ff_data = self.extract_embeddings_from_aligned_face(
            bgr_aligned_face, model=model, models=models,
        )
        if new_ff_data is None:
            return face_info

        # MERGE em vez de REPLACE — preserva embedding/FIQA/yaw vindos de
        # runs anteriores. Sem isso, enrich seletivo via face alinhada
        # (ex: rodar só sepaelv4) apagava sepaelv2, FIQA, etc. que já
        # estavam populados pra cada face.
        existing = face_info.ff_info
        if existing is None:
            face_info.ff_info = new_ff_data
            return face_info

        for key in _FFINFO_MERGE_KEYS:
            val = getattr(new_ff_data, key, None)
            if val is not None:
                setattr(existing, key, val)
        return face_info

    def _detectar_tentando_rotacoes(
        self,
        engine,
        img_array: np.ndarray,
        rotacoes: Sequence[int],
        image_path: str,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Optional[dict], int]:
        """Detecta na orientação em que a imagem está; se falhar, gira e tenta.

        Foto tirada com a câmera de lado não tem face detectável na
        orientação gravada e tem depois de girar 90°. Sem isso, a imagem era
        classificada como "sem face" quando o problema era só a orientação.

        Devolve `(resultado, graus)`; `graus` é a rotação horária que fez a
        detecção funcionar (0 = a imagem como está). `(None, 0)` quando
        nenhuma orientação produziu face.

        Cada tentativa é uma passada completa do detector, então a ordem
        importa: a imagem como está vem primeiro e o custo extra só é pago
        pelas imagens que de fato falharam.
        """
        tentativas = [0]
        for graus in rotacoes or ():
            g = normalizar_graus(graus)
            if g and g not in tentativas:
                tentativas.append(g)

        for graus in tentativas:
            imagem = girar_imagem(img_array, graus) if graus else img_array
            try:
                resultado = self._normalize_process_image_result(
                    engine.process_image(imagem)
                )
            except Exception as e:
                # Falha numa orientação não condena as outras — o detector
                # pode estourar num shape específico e ir bem no girado.
                msg = (
                    f"Falha ao detectar em {image_path} "
                    f"({descrever_rotacao(graus)}): {e}"
                )
                (error_callback or print)(msg)
                continue
            if resultado:
                if graus:
                    print(
                        f"🔄 [FFService] Face encontrada só com a imagem "
                        f"{descrever_rotacao(graus)}: {image_path}"
                    )
                return resultado, graus

        return None, 0

    def process_face(
        self,
        face_info: FaceInfo,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        *,
        dataset: Optional[FaceDataset] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        rotation_attempts: Sequence[int] = (),
    ) -> FaceInfo:
        """
        Processa uma face, preenchendo `face_info.ff_info` com bbox,
        keypoints, qualidade, demografia, pose e os embeddings dos
        modelos pedidos (no slot correspondente — `embedding_sepaelv2`,
        `embedding_sepaelv4`, etc.).

        Args:
            face_info: face a ser processada (consome `face_info.image_path`).
            model: nome do modelo a usar (None = default). Ignorado se
                `models` for fornecido.
            models: lista de modelos a extrair em uma só passada. Útil
                quando o usuário pede múltiplos embeddings — todos vão
                pra mesma `FFInfo`, com metadata (bbox/keypoints/...)
                vindo da primeira chamada.
            dataset: PR L1b — quando passado, resolve o `image_path` via
                `dataset.resolve_image_path(face_info)` (suporta paths
                relativos pós-localização). Quando None, usa
                `face_info.image_path` direto (compat 100% com chamadas
                pré-L1b).
            rotation_attempts: rotações horárias (múltiplos de 90°) a tentar
                quando a detecção falha na orientação gravada — tipicamente
                `ROTACOES_PADRAO`. Vazio (default) preserva o comportamento
                de uma única tentativa.

        Coordenadas: quando a detecção só funciona com a imagem girada,
        `bbox` e `keypoints` são convertidos de volta para o sistema da
        imagem ORIGINAL antes de serem gravados, e a rotação vai em
        `ff_info.detection_rotation`. A caixa tem de casar com o arquivo
        que `image_path` aponta — guardá-la no frame girado faria o recorte
        sair da região errada sem nada no dado denunciar a troca.
        `aligned_face` fica como veio: é o recorte de pé, e é justamente
        isso que o encoder deve ver.
        """
        image_path = (
            str(dataset.resolve_image_path(face_info))
            if dataset is not None
            else face_info.image_path
        )
        img_array = self._safe_read_image(image_path)
        if img_array is None:
            return face_info

        # Resolve a lista de modelos a rodar nesta chamada.
        target_models: List[str]
        if models:
            target_models = list(models)
        elif model is not None:
            target_models = [model]
        else:
            target_models = [self._default_model]

        # Começa do FFInfo existente (se houver) — semântica de MERGE,
        # não REPLACE. Sem isso, enrich com um único modelo (ex: sepaelv4)
        # apagava tudo que veio de runs anteriores: embedding_sepaelv2,
        # FIQA, yaw/pitch/roll, etc. — porque `face_info.ff_info = ff_data`
        # (linha final) substituía o objeto inteiro.
        ff_data: Optional[FFInfo] = face_info.ff_info

        # A orientação é decidida UMA vez, na primeira passada, e vale para
        # todos os modelos da chamada: embeddings de modelos diferentes têm
        # de sair do mesmo recorte alinhado, senão não são comparáveis entre
        # si nem com o resto do dataset.
        imagem_para_modelos = img_array
        rotacao_usada = 0
        primeira_passada = True

        for m in target_models:
            try:
                model_name, engine = self._resolve_engine(m)
                if primeira_passada:
                    primeira_passada = False
                    resultado, rotacao_usada = self._detectar_tentando_rotacoes(
                        engine, img_array, rotation_attempts,
                        image_path, error_callback,
                    )
                    if rotacao_usada:
                        imagem_para_modelos = girar_imagem(img_array, rotacao_usada)
                else:
                    resultado = self._normalize_process_image_result(
                        engine.process_image(imagem_para_modelos)
                    )
                if not resultado:
                    continue

                if ff_data is None:
                    ff_data = FFInfo()

                # Atualiza só os campos que o resultado efetivamente
                # trouxe — `resultado.get(key)` None NÃO sobrescreve o
                # valor anterior. Isso preserva FIQA/yaw/etc. quando o
                # usuário roda o enrich com `ff_extended=False` (que faz
                # o ForensicFace omitir esses campos no `_assemble_result`).
                forma_usada = imagem_para_modelos.shape[:2]
                for key in (
                    "ipd", "fiqa_score", "det_score", "gender", "age",
                    "yaw", "pitch", "roll", "bbox", "keypoints",
                    "aligned_face",
                ):
                    val = resultado.get(key)
                    if val is None:
                        continue
                    if rotacao_usada:
                        # Volta ao sistema de coordenadas do arquivo.
                        if key == "bbox":
                            val = caixa_girada_para_original(
                                val, forma_usada, rotacao_usada,
                            )
                        elif key == "keypoints":
                            val = pontos_girados_para_originais(
                                val, forma_usada, rotacao_usada,
                            )
                    setattr(ff_data, key, val)

                # `or None` limpa a marca quando uma nova passada detectou
                # na orientação gravada — sem isso o valor de um run anterior
                # sobreviveria ao merge e mentiria sobre esta extração.
                ff_data.detection_rotation = rotacao_usada or None

                emb = resultado.get("embedding")
                if emb is not None:
                    setattr(ff_data, self._slot_for_model(model_name), emb)

            except Exception as e:
                msg = f"Error processing face {image_path} (model={m}): {e}"
                (error_callback or print)(msg)

        if ff_data is not None:
            face_info.ff_info = ff_data

        return face_info

    def process_dataset(
        self,
        dataset: FaceDataset,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        rotation_attempts: Sequence[int] = (),
    ) -> FaceDataset:
        """
        Processa um dataset completo, gerando biometria para todas as faces
        com os modelos escolhidos. Emite progresso em % via callback.

        Args:
            dataset: FaceDataset a processar (mutado in-place).
            progress_callback: callable (mensagem, pct) chamado periodicamente.
            model: nome do modelo a usar (None = default). Ignorado se
                `models` for fornecido.
            models: lista de modelos a extrair em uma só passada por face.
                Cada face termina com um embedding em cada slot pedido.
            rotation_attempts: rotações a tentar quando a detecção falha na
                orientação gravada. Só tem efeito no caminho per-face
                (extração crua) — os caminhos em lote partem de faces já
                alinhadas ou de um `aligned_face` em cache, onde não há
                orientação a corrigir.
        """
        total_faces = len(dataset)
        if total_faces == 0:
            return dataset

        # Resolve label informativo do log.
        if models:
            model_label = ", ".join(models)
        else:
            model_label = self._resolve_engine(model)[0]
        print(
            f"Iniciando processamento biométrico do dataset: {dataset.name} "
            f"(modelos={model_label})"
        )

        # 1. Extração biométrica (até 95% do progresso)
        # Quatro caminhos por prioridade:
        # - `is_aligned=True`: faces em 112×112 no disco (Image Pipeline
        #   COPY_ALIGNED). Lê crops do disco, batch inference. ⚡
        # - **Cache hit em memória**: faces com `ff_info.aligned_face`
        #   populado dentro da mesma sessão de processo. Pula detect+align,
        #   batch da extração. ⚡ Raro pós-reload de parquet — o
        #   `FFInfo.to_dict(exclude_heavy=True)` dropa `aligned_face`.
        # - **Enrichment (ff_info populado, sem cache de aligned_face)**:
        #   caso típico do usuário adicionar sepaelv4 num dataset persistido
        #   que já tem sepaelv2 — bbox/keypoints/gender/age/pose já existem
        #   mas `aligned_face` foi descartado na serialização. Alinha
        #   per-face (barato), extrai embedding em batch. ⚡ Speedup
        #   significativo no encoder ViT.
        # - Foto original sem nenhuma extração prévia: detect+align+
        #   extract per-face via `process_image` — preserva contrato de
        #   metadata completo (gender/age/pose/det_score) que
        #   `detect_and_align` sozinho não traz.
        is_aligned = bool(getattr(dataset, "is_aligned", False))

        # Parte das faces com a face alinhada em memória e parte sem: as
        # que têm vão pelo caminho em cache, SEM detecção, e o resto segue
        # a escolha normal. Sem a partição, uma única face sem cache jogava
        # o dataset inteiro na redetecção — que é justamente o que o cache
        # existe para evitar (o importador de vídeo carrega a face alinhada
        # da extração porque redetectar no recorte de uma face pequena falha).
        if not is_aligned:
            com_cache = [f for f in dataset if _tem_alinhada_em_cache(f)]
            if com_cache and len(com_cache) < total_faces:
                ids_cache = {id(f) for f in com_cache}
                sem_cache = [f for f in dataset if id(f) not in ids_cache]
                print(
                    f"[FFService] {len(com_cache)} face(s) com face alinhada "
                    f"em cache (sem detecção) e {len(sem_cache)} sem."
                )
                self._process_dataset_cached_aligned_batched(
                    _sub_dataset(dataset, com_cache), len(com_cache),
                    model, models, progress_callback,
                )
                self.process_dataset(
                    _sub_dataset(dataset, sem_cache),
                    progress_callback=progress_callback, model=model,
                    models=models, rotation_attempts=rotation_attempts,
                )
                # O consenso de gênero roda por identidade e tem de ver o
                # dataset inteiro, não cada metade.
                self._apply_identity_gender_consensus(dataset)
                return dataset

        all_have_cached_aligned = (
            total_faces > 0
            and all(
                f.ff_info is not None
                and getattr(f.ff_info, "aligned_face", None) is not None
                for f in dataset
            )
        )
        all_have_prior_ffinfo = (
            total_faces > 0
            and all(f.ff_info is not None for f in dataset)
        )

        if is_aligned:
            print(
                f"[FFService] Dataset is_aligned=True - batch inference "
                f"(batch_size={self._batch_size})."
            )
            self._process_dataset_aligned_batched(
                dataset, total_faces, model, models, progress_callback,
            )
        elif all_have_cached_aligned:
            print(
                f"[FFService] Todas as faces tem aligned_face cacheado - "
                f"batch via cache (batch_size={self._batch_size})."
            )
            self._process_dataset_cached_aligned_batched(
                dataset, total_faces, model, models, progress_callback,
            )
        elif all_have_prior_ffinfo:
            print(
                f"[FFService] Enrichment detectado (ff_info populado, "
                f"aligned_face não cacheado) - align per-face + batch "
                f"extract (batch_size={self._batch_size})."
            )
            self._process_dataset_align_then_batch(
                dataset, total_faces, model, models, progress_callback,
                rotation_attempts=rotation_attempts,
            )
        else:
            # Adapter: erros por modelo dentro do process_face viram
            # linhas de log no LogProgressDialog em vez de print()
            # silencioso. Sem isso, falhas (ex: V4 com input
            # incompatível) eram descobertas só olhando stdout.
            def _err_cb(msg: str) -> None:
                if progress_callback:
                    progress_callback(f"⚠ {msg}", 0)
                else:
                    print(msg)

            for idx, face in enumerate(
                tqdm(dataset, desc=f"Processando {dataset.name}", unit="face")
            ):
                self.process_face(
                    face, model=model, models=models, dataset=dataset,
                    error_callback=_err_cb,
                    rotation_attempts=rotation_attempts,
                )
                if progress_callback:
                    pct = int(((idx + 1) / total_faces) * 95)
                    progress_callback(
                        f"Extraindo biometria ({idx + 1}/{total_faces})...", pct
                    )

        # 2. Pós-processamento: consenso de gênero por identidade (5%)
        if progress_callback:
            progress_callback("Calculando consenso demográfico...", 95)

        self._apply_identity_gender_consensus(dataset)

        if progress_callback:
            progress_callback("Concluído", 100)

        return dataset

    def _process_dataset_cached_aligned_batched(
        self,
        dataset: "FaceDataset",
        total_faces: int,
        model: Optional[str],
        models: Optional[List[str]],
        progress_callback: Optional[Callable[[str, int], None]],
    ) -> None:
        """Batch usando `face.ff_info.aligned_face` já cacheado.

        Caso de uso típico: enrichment com modelo adicional. As faces já
        foram processadas antes (têm ff_info.aligned_face em RGB), o
        usuário pediu pra adicionar um novo embedding (ex: sepaelv4
        sobre dataset que já tem sepaelv2). Pulamos detect+align (já
        cacheado), só batch a extração do novo modelo.

        O aligned_face fica em RGB no FFInfo (per `_assemble_result`
        do forensicface); convertemos pra BGR antes do batch — mesma
        convenção do path is_aligned.
        """
        target_models: List[str]
        if models:
            target_models = list(models)
        elif model is not None:
            target_models = [model]
        else:
            target_models = [self._default_model]

        buffer_crops: List[np.ndarray] = []
        buffer_faces: List[FaceInfo] = []
        buffer_keypoints: List[Optional[np.ndarray]] = []

        # Adapter que rebobina erros internos do batch helper como
        # linhas de log no LogProgressDialog (via progress_callback).
        # Sem isso, falhas por modelo (ex: V4 com shape de input
        # incompatível) viravam print() silencioso pro terminal.
        def _err_cb(msg: str) -> None:
            if progress_callback:
                progress_callback(f"⚠ {msg}", 0)
            else:
                print(msg)

        def flush() -> None:
            if not buffer_crops:
                return
            arr = np.stack(buffer_crops, axis=0)
            # Pontos 112×112 quando TODAS as faces do lote os têm — o KPRPE
            # (sepaelv6) os exige; os demais modelos ignoram. Mesma regra do
            # caminho de faces alinhadas em disco.
            kps_batch = None
            if buffer_keypoints and all(k is not None for k in buffer_keypoints):
                kps_batch = np.stack(buffer_keypoints, axis=0)
            try:
                ff_infos = self.extract_embeddings_from_aligned_faces_batch(
                    arr, models=target_models,
                    aligned_keypoints_batch=kps_batch, error_callback=_err_cb,
                )
            except Exception as e:
                _err_cb(
                    f"[FFService] Falha no batch de tamanho {len(arr)}: {e}"
                )
                ff_infos = [None] * len(buffer_faces)
            for face, new_ff in zip(buffer_faces, ff_infos):
                if new_ff is None:
                    continue
                existing = face.ff_info  # já existe (gate do caller)
                # MERGE: novo modelo sobrescreve só o slot dele +
                # fiqa_score (FIQA score é mesmo entre engines).
                # Itera TODOS os slots conhecidos pra acomodar runs com
                # qualquer modelo (v2/v3/v4/v5/v6).
                for key in ("fiqa_score", *_FFINFO_EMBEDDING_SLOTS):
                    val = getattr(new_ff, key, None)
                    if val is not None:
                        setattr(existing, key, val)
            buffer_crops.clear()
            buffer_faces.clear()
            buffer_keypoints.clear()

        processed = 0
        for face in tqdm(
            dataset, desc=f"Processando {dataset.name} (cache+batch)",
            unit="face",
        ):
            processed += 1
            ff = face.ff_info
            aligned_rgb = (
                getattr(ff, "aligned_face", None) if ff is not None else None
            )
            if aligned_rgb is None or aligned_rgb.shape != (112, 112, 3):
                # Cache inválido — pula. (Não deveria acontecer pois o
                # caller já garantiu, mas defensivo contra shapes
                # estranhos vindos de payloads legados.)
                continue
            # RGB → BGR pro batch helper (espera BGR, consistente com
            # `extract_embeddings_from_aligned_face`).
            aligned_bgr = aligned_rgb[..., ::-1].copy()
            buffer_crops.append(aligned_bgr)
            buffer_faces.append(face)
            buffer_keypoints.append(_aligned_keypoints_from_face(face))

            if len(buffer_crops) >= self._batch_size:
                flush()

            if progress_callback:
                pct = int((processed / total_faces) * 95)
                progress_callback(
                    f"Extraindo embedding ({processed}/{total_faces}) "
                    f"[cache+batch={self._batch_size}]...", pct,
                )

        flush()

    def _process_dataset_align_then_batch(
        self,
        dataset: "FaceDataset",
        total_faces: int,
        model: Optional[str],
        models: Optional[List[str]],
        progress_callback: Optional[Callable[[str, int], None]],
        rotation_attempts: Sequence[int] = (),
    ) -> None:
        """Caminho enrichment: alinha per-face, extrai embedding em batch.

        Caso de uso: usuário roda enrich com modelo adicional num dataset
        que já tem `ff_info` populado (bbox/keypoints/gender/age/pose
        vindos de run anterior), MAS `aligned_face` não está cacheado em
        memória — `FFInfo.to_dict(exclude_heavy=True)` dropa `aligned_face`
        na serialização parquet, então abrir um dataset salvo zera o cache
        e `_process_dataset_cached_aligned_batched` não dispara.

        Pipeline: lê imagem do disco → `engine.detect_and_align` per-face
        (parte barata — só detector + crop afim) → acumula crop alinhado
        em buffer → a cada `batch_size`, chama
        `extract_embeddings_from_aligned_faces_batch` (parte cara — encoder
        ViT em batch). Speedup esperado pro v4 (ViT) vs per-face: ~5–10×.

        Limitação: `detect_and_align` retorna só `aligned_face` +
        `aligned_keypoints` — NÃO retorna gender/age/yaw/pitch/roll/
        det_score (esses só vêm do `process_image` completo). Por isso o
        gate no caller exige `all_have_prior_ffinfo=True` (caso
        enrichment): assume que esses campos já estão populados do run
        anterior. Pra primeira extração crua, o caller cai no per-face
        fallback que usa `process_image` (preserva contrato de metadata
        completo).

        MERGE: novo embedding + fiqa_score (se vier) sobrescrevem só os
        slots correspondentes; demais campos do `FFInfo` existente são
        preservados (idêntico ao `_process_dataset_cached_aligned_batched`).
        """
        target_models: List[str]
        if models:
            target_models = list(models)
        elif model is not None:
            target_models = [model]
        else:
            target_models = [self._default_model]

        # Engine pro detect+align: primeira da lista alvo. det_10g é o
        # mesmo entre engines (detector compartilhado quando layout shared,
        # ou idêntico nas pastas legadas) — escolha não afeta resultado.
        _, align_engine = self._resolve_engine(target_models[0])

        # Modelos KPRPE (sepaelv6) consomem aligned_keypoints como segundo
        # input ONNX. Detecta isso antes do loop pra decidir se buffer
        # de keypoints é populado.
        needs_keypoints = any(
            (SepaelModel.from_ff_name(m) or SepaelModel.default()).requires_keypoints
            for m in target_models
        )

        buffer_crops: List[np.ndarray] = []
        buffer_faces: List[FaceInfo] = []
        buffer_kps: List[np.ndarray] = []  # só populado se needs_keypoints

        # Adapter de erro: mesma motivação dos outros helpers em batch —
        # falhas internas viram linhas no LogProgressDialog em vez de
        # print() silencioso pro terminal.
        def _err_cb(msg: str) -> None:
            if progress_callback:
                progress_callback(f"⚠ {msg}", 0)
            else:
                print(msg)

        def flush() -> None:
            if not buffer_crops:
                return
            arr = np.stack(buffer_crops, axis=0)
            kps_arr = (
                np.stack(buffer_kps, axis=0)
                if needs_keypoints and buffer_kps else None
            )
            try:
                ff_infos = self.extract_embeddings_from_aligned_faces_batch(
                    arr,
                    models=target_models,
                    aligned_keypoints_batch=kps_arr,
                    error_callback=_err_cb,
                )
            except Exception as e:
                _err_cb(
                    f"[FFService] Falha no batch de tamanho {len(arr)}: {e}"
                )
                ff_infos = [None] * len(buffer_faces)
            for face, new_ff in zip(buffer_faces, ff_infos):
                if new_ff is None:
                    continue
                existing = face.ff_info  # gate do caller garante non-None
                # MERGE só de fiqa_score + slots de embedding. NÃO mexe em
                # bbox/keypoints/gender/age/pose — esses estão preservados
                # do run anterior; detect_and_align nem os retornaria.
                for key in ("fiqa_score", *_FFINFO_EMBEDDING_SLOTS):
                    val = getattr(new_ff, key, None)
                    if val is not None:
                        setattr(existing, key, val)
            buffer_crops.clear()
            buffer_faces.clear()
            buffer_kps.clear()

        def enfileirar(face, image_path: str) -> Optional[str]:
            """Alinha e põe no buffer. Devolve o motivo quando não deu."""
            img_array = self._safe_read_image(image_path)
            if img_array is None:
                return "imagem ilegível"
            item, graus = alinhar_tentando_rotacoes(
                align_engine, img_array, rotation_attempts, image_path,
                rotacao_conhecida=rotacao_registrada(face),
                error_callback=_err_cb,
            )
            if not item:
                return "face não detectada na imagem"
            if graus and face.ff_info is not None:
                face.ff_info.detection_rotation = graus
            rgb_aligned = item.get("aligned_face")
            if rgb_aligned is None or rgb_aligned.shape != (112, 112, 3):
                return "alinhamento não produziu face 112×112"
            # RGB → BGR pro batch helper (consistente com
            # `extract_embeddings_from_aligned_face`).
            bgr_aligned = rgb_aligned[..., ::-1].copy()
            if needs_keypoints:
                aligned_kps = item.get("aligned_keypoints")
                if aligned_kps is None:
                    # KPRPE precisa de keypoints; sem eles o encoder
                    # rejeita o input. Fica fora para manter os buffers em sync.
                    return "alinhamento sem pontos 112×112 (exigidos pelo modelo)"
                buffer_kps.append(aligned_kps)
            buffer_crops.append(bgr_aligned)
            buffer_faces.append(face)
            return None

        processed = 0
        n_sem_alinhamento = 0
        for face in tqdm(
            dataset, desc=f"Processando {dataset.name} (align+batch)",
            unit="face",
        ):
            processed += 1
            image_path = str(dataset.resolve_image_path(face))
            motivo = enfileirar(face, image_path)
            if motivo is not None:
                # A face fica sem embedding; quem marca o erro nela é o
                # pipeline. Aqui o registro é para o log de progresso, que
                # é onde o operador vê QUAL imagem caiu e por quê.
                n_sem_alinhamento += 1
                _err_cb(f"[FFService] {motivo}: {Path(image_path).name}")

            if len(buffer_crops) >= self._batch_size:
                flush()

            if progress_callback:
                pct = int((processed / total_faces) * 95)
                progress_callback(
                    f"Extraindo embedding ({processed}/{total_faces}) "
                    f"[align+batch={self._batch_size}]...", pct,
                )

        flush()
        if n_sem_alinhamento:
            _err_cb(
                f"[FFService] {n_sem_alinhamento} de {total_faces} face(s) "
                "ficaram sem embedding por falha de detecção/alinhamento."
            )

    def _process_dataset_aligned_batched(
        self,
        dataset: "FaceDataset",
        total_faces: int,
        model: Optional[str],
        models: Optional[List[str]],
        progress_callback: Optional[Callable[[str, int], None]],
    ) -> None:
        """Processa um dataset is_aligned=True em batches.

        Loop: lê crops 112×112 do disco sequencialmente, acumula em
        buffer; quando o buffer enche (ou no fim), chama
        `extract_embeddings_from_aligned_faces_batch` UMA vez por modelo.

        Falhas individuais (arquivo ausente, crop com shape errado)
        viram None no resultado — face mantém `ff_info=None`, igual ao
        caminho legado por-face. Logs explicam o motivo.
        """
        target_models: List[str]
        if models:
            target_models = list(models)
        elif model is not None:
            target_models = [model]
        else:
            target_models = [self._default_model]

        # Buffers paralelos: crops carregados + ponteiros pras FaceInfo
        # correspondentes + keypoints alinhados (p/ KPRPE). Flush quando
        # atinge `_batch_size`.
        buffer_crops: List[np.ndarray] = []
        buffer_faces: List[FaceInfo] = []
        buffer_keypoints: List[Optional[np.ndarray]] = []

        # Mesma motivação do helper cached: erros internos do batch
        # precisam virar linhas de log no LogProgressDialog em vez de
        # print() silencioso pro terminal.
        def _err_cb(msg: str) -> None:
            if progress_callback:
                progress_callback(f"⚠ {msg}", 0)
            else:
                print(msg)

        def flush() -> None:
            if not buffer_crops:
                return
            arr = np.stack(buffer_crops, axis=0)  # (N, 112, 112, 3)
            # Passa os keypoints alinhados quando TODAS as faces do batch os têm
            # (KPRPE/sepaelv6 exige; os demais ignoram). Se faltar em alguma, vai
            # None — modelos que exigem keypoints falham com mensagem clara.
            kps_batch = None
            if buffer_keypoints and all(k is not None for k in buffer_keypoints):
                kps_batch = np.stack(buffer_keypoints, axis=0)  # (N, 5, 2)
            try:
                ff_infos = self.extract_embeddings_from_aligned_faces_batch(
                    arr, models=target_models,
                    aligned_keypoints_batch=kps_batch, error_callback=_err_cb,
                )
            except Exception as e:
                _err_cb(
                    f"[FFService] Falha no batch de tamanho {len(arr)}: {e}"
                )
                ff_infos = [None] * len(buffer_faces)
            for face, new_ff in zip(buffer_faces, ff_infos):
                if new_ff is None:
                    continue
                # MERGE em vez de REPLACE — preserva campos vindos de runs
                # anteriores (mesma semântica do process_face_aligned).
                existing = face.ff_info
                if existing is None:
                    face.ff_info = new_ff
                    continue
                for key in _FFINFO_MERGE_KEYS:
                    val = getattr(new_ff, key, None)
                    if val is not None:
                        setattr(existing, key, val)
            buffer_crops.clear()
            buffer_faces.clear()
            buffer_keypoints.clear()

        processed = 0
        for face in tqdm(
            dataset, desc=f"Processando {dataset.name} (batch)", unit="face"
        ):
            image_path = str(dataset.resolve_image_path(face))
            crop = self._safe_read_image(image_path)
            processed += 1

            if crop is None:
                # Face inválida — pula do batch (mantém ff_info=None).
                pass
            elif crop.shape != (112, 112, 3):
                # Shape inesperado: dataset marcado is_aligned mas o
                # arquivo não é um crop 112×112. Avisa e pula.
                print(
                    f"[FFService] Crop com shape {crop.shape} ignorado "
                    f"(esperado (112,112,3)): {image_path}"
                )
            else:
                buffer_crops.append(crop)
                buffer_faces.append(face)
                buffer_keypoints.append(_aligned_keypoints_from_face(face))

            if len(buffer_crops) >= self._batch_size:
                flush()

            if progress_callback:
                pct = int((processed / total_faces) * 95)
                progress_callback(
                    f"Extraindo biometria ({processed}/{total_faces}) "
                    f"[batch={self._batch_size}]...", pct,
                )

        # Flush final.
        flush()

    def process_batch(
        self,
        faces: List[FaceInfo],
        model: Optional[str] = None,
        *,
        dataset: Optional[FaceDataset] = None,
    ) -> List[FaceInfo]:
        """Processa uma lista de faces em sequência (wrapper para tqdm).

        Args:
            dataset: PR L1b — quando passado, propaga pra `process_face`
                pra resolver paths em modo RELATIVE.
        """
        return [
            self.process_face(f, model=model, dataset=dataset)
            for f in tqdm(faces, desc="Extracting Features", unit="img")
        ]

    def compare(
        self,
        image_path_1: str,
        image_path_2: str,
        model: Optional[str] = None,
    ) -> Optional[float]:
        """
        Compara duas imagens faciais usando o `compare` nativo do
        ForensicFace (cosine similarity entre embeddings).

        Args:
            image_path_1: caminho da primeira imagem.
            image_path_2: caminho da segunda imagem.
            model: nome do modelo a usar (None = default declarado).

        Returns:
            Score cosine como float, ou None se a comparação falhar.
        """
        try:
            _, engine = self._resolve_engine(model)
            score = engine.compare(image_path_1, image_path_2)
            if score is None:
                return None
            return float(score)
        except Exception as e:
            print(f"Error comparing faces: {image_path_1} vs {image_path_2} -> {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Mosaico e extração de vídeo (não dependem de embedding)
    # ─────────────────────────────────────────────────────────────────────

    def build_mosaic(
        self,
        image_paths: List[str],
        mosaic_shape: Tuple[int, int],
        border: float = 0.03,
        save_to: Optional[str] = None,
        draw_keypoints: bool = False,
        model: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Monta um mosaico de faces a partir de uma lista de caminhos. As
        faces são detectadas e alinhadas pelo ForensicFace antes de irem
        pro mosaico.

        Args:
            image_paths: lista de paths de imagem.
            mosaic_shape: (linhas, colunas) — produto deve casar com len(image_paths).
            border: borda relativa entre tiles (0.03 = ~3% do tile).
            save_to: se informado, grava o mosaico nesse path (PNG/JPG).
            draw_keypoints: sobrepõe os 5 landmarks em cada face.
            model: engine a usar (None = default). Não afeta o resultado
                visual — só seleciona qual instância faz a detecção.

        Returns:
            ndarray BGR do mosaico, ou None em caso de falha.
        """
        try:
            _, engine = self._resolve_engine(model)
            return engine.build_mosaic(
                img_path_list=list(image_paths),
                mosaic_shape=mosaic_shape,
                border=border,
                save_to=save_to,
                draw_keypoints=draw_keypoints,
            )
        except Exception as e:
            print(f"Error building mosaic ({len(image_paths)} images): {e}")
            return None

    def extract_faces_from_video(
        self,
        video_path: str,
        dest_folder: Optional[str] = None,
        every_n_frames: int = 1,
        margin: float = 2.0,
        start_from: float = 0.0,
        export_metadata: bool = False,
        model: Optional[str] = None,
    ) -> int:
        """
        Extrai faces de um vídeo, gravando cada uma como PNG individual.

        Args:
            video_path: caminho do vídeo de entrada.
            dest_folder: pasta de saída (None = pasta com nome do vídeo, criada
                ao lado dele).
            every_n_frames: amostra 1 a cada N frames (1 = todos).
            margin: fator de expansão do bbox antes do crop (2.0 = 2x).
            start_from: tempo (em segundos) a partir do qual processar.
            export_metadata: se True, grava também um JSONL com bbox e
                landmarks de cada face extraída.
            model: engine a usar (None = default). Não afeta o crop —
                só seleciona qual instância faz a detecção.

        Returns:
            Número total de faces extraídas (0 se falha).
        """
        try:
            _, engine = self._resolve_engine(model)
            count = engine.extract_faces(
                video_path=video_path,
                dest_folder=dest_folder,
                every_n_frames=every_n_frames,
                margin=margin,
                start_from=start_from,
                export_metadata=export_metadata,
            )
            return int(count) if count is not None else 0
        except Exception as e:
            print(f"Error extracting faces from video {video_path}: {e}")
            return 0

    # ─────────────────────────────────────────────────────────────────────
    # Pós-processamento e helpers internos
    # ─────────────────────────────────────────────────────────────────────

    def _apply_identity_gender_consensus(self, dataset: FaceDataset):
        """
        Agrupa faces por identidade e elege um único gênero por pessoa,
        usando voto ponderado pelo FIQA Score (qualidade). Predições
        explícitas pelo humano (`face.gender` já preenchido) prevalecem.
        """
        print(
            "📊 Calculando consenso demográfico por identidade "
            "(ponderado por qualidade)..."
        )

        faces_by_id: Dict[str, List[FaceInfo]] = {}
        for face in dataset:
            if face.identity:
                faces_by_id.setdefault(face.identity, []).append(face)

        count_updated = 0

        for identity, faces in faces_by_id.items():
            primeira_face = faces[0]
            gender_atual = (
                str(primeira_face.gender).strip().lower()
                if primeira_face.gender is not None
                else ""
            )
            valores_vazios = ["", "unknown", "none", "nan", "null"]

            # Anotação humana prevalece — IA pula.
            if gender_atual not in valores_vazios:
                continue

            votos: Dict[str, float] = {"Male": 0.0, "Female": 0.0}

            for face in faces:
                if (
                    face.ff_info
                    and face.ff_info.gender
                    and face.ff_info.fiqa_score is not None
                ):
                    pred = str(face.ff_info.gender).strip().title()
                    peso = float(face.ff_info.fiqa_score)
                    votos[pred] = votos.get(pred, 0.0) + peso

            if sum(votos.values()) == 0:
                continue

            vencedor = max(votos, key=votos.get)
            for face in faces:
                face.gender = vencedor

            count_updated += 1

        if count_updated > 0:
            print(f"✅ Gênero estimado unificado para {count_updated} identidades.")

    def _safe_read_image(self, path: str) -> Optional[np.ndarray]:
        """Lê imagens do disco lidando com caracteres especiais no caminho.

        Failure NÃO é silenciosa: imprime path + razão. Retorna None pra
        manter contrato com `process_face`, mas o operador vê no terminal
        que aquela face não pôde ser lida (path quebrado, arquivo
        corrompido, permissão, etc.).
        """
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️ [FFService] cv2.imdecode falhou: {path}")
            return img
        except Exception as e:
            print(f"⚠️ [FFService] _safe_read_image falhou ({type(e).__name__}): {path} — {e}")
            return None
