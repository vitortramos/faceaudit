import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Tuple
from tqdm.auto import tqdm

# Import absoluto baseada na estrutura do pacote
from faceaudit.models import FaceInfo, FFInfo, FaceDataset

try:
    from forensicface.app import ForensicFace
except ImportError:
    ForensicFace = None


# Mapeamento entre nome do modelo e slot correspondente em FFInfo.
# Adicionar entradas novas aqui é o único ponto necessário para suportar
# um novo backbone do ForensicFace.
_MODEL_TO_FFINFO_SLOT: Dict[str, str] = {
    "sepaelv2": "embedding_sepaelv2",
    "sepaelv4": "embedding_sepaelv4",
}


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
    ):
        if ForensicFace is None:
            raise ImportError("A biblioteca 'forensicface' não foi encontrada.")

        # Resolve a lista de modelos a carregar:
        #   - `models` tem prioridade quando fornecido (forma nova)
        #   - `model` é mantido como kwarg de retrocompat (containers.py)
        #   - default = ["sepaelv2"]
        resolved: List[str]
        if models is not None:
            resolved = [models] if isinstance(models, str) else list(models)
        elif model is not None:
            resolved = [model]
        else:
            resolved = ["sepaelv2"]

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
    def _slot_for_model(model_name: str) -> str:
        """
        Retorna o atributo de FFInfo onde o embedding do `model_name` deve
        ser gravado. Modelos desconhecidos caem no slot v2 (mantém o
        comportamento histórico do FFService antigo).
        """
        return _MODEL_TO_FFINFO_SLOT.get(model_name.lower(), "embedding_sepaelv2")

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
            res = engine.process_image(img_array)
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
    ) -> Optional[np.ndarray]:
        """Variante de `align_face_only` que recebe ndarray direto
        em vez de `FaceInfo` (PR I3).

        Útil em pipelines in-memory onde a imagem já está carregada
        e não há FaceInfo associado — caso típico: editor de pipeline
        de imagens calculando scores de qualidade da imagem
        processada (que existe só em RAM).

        Pipeline interno idêntico: `backend.detect_faces` →
        `_get_best_face` → `backend.norm_crop`. Pula o
        `_safe_read_image` (input já é ndarray).

        Args:
            bgr_image: ndarray BGR (H, W, 3) uint8.
            model: nome do modelo (None = default).

        Returns:
            ndarray BGR (112, 112, 3) uint8 ou None se não detectou.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None
        try:
            _, engine = self._resolve_engine(model)
            faces = engine.backend.detect_faces(bgr_image)
            if not faces:
                return None
            face = engine._get_best_face(bgr_image, faces, criterion="size")
            return engine.backend.norm_crop(bgr_image, face.kps)
        except Exception as e:
            print(f"Error in align_face_from_array: {e}")
            return None

    def align_face_only(
        self,
        face_info: FaceInfo,
        *,
        dataset: Optional[FaceDataset] = None,
        model: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Detecta + alinha SEM extrair embedding (~30-50% mais rápido
        que `process_face`).

        Pipeline interno: `_safe_read_image` → `backend.detect_faces` →
        `_get_best_face` (criterion=size) → `backend.norm_crop`. Pula
        o `_compute_embeddings` que é a etapa cara (AdaFace ONNX).

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
            faces = engine.backend.detect_faces(img_array)
            if not faces:
                return None
            face = engine._get_best_face(img_array, faces, criterion="size")
            return engine.backend.norm_crop(img_array, face.kps)
        except Exception as e:
            print(f"Error in align_face_only for {image_path}: {e}")
            return None

    def extract_embeddings_from_aligned_face(
        self,
        bgr_aligned_face: np.ndarray,
        *,
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
                res = engine.process_aligned_face_image(rgb_aligned_face)
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

        ff_data = self.extract_embeddings_from_aligned_face(
            bgr_aligned_face, model=model, models=models,
        )
        if ff_data is not None:
            face_info.ff_info = ff_data
        return face_info

    def process_face(
        self,
        face_info: FaceInfo,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        *,
        dataset: Optional[FaceDataset] = None,
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

        ff_data: Optional[FFInfo] = None

        for m in target_models:
            try:
                model_name, engine = self._resolve_engine(m)
                resultado = engine.process_image(img_array)
                if not resultado:
                    continue

                # Primeira iteração cria o FFInfo com toda a metadata
                # (bbox/keypoints/quality/etc.); iterações seguintes só
                # preenchem o slot do embedding correspondente.
                if ff_data is None:
                    ff_data = FFInfo(
                        ipd=resultado.get("ipd"),
                        fiqa_score=resultado.get("fiqa_score"),
                        det_score=resultado.get("det_score"),
                        gender=resultado.get("gender"),
                        age=resultado.get("age"),
                        yaw=resultado.get("yaw"),
                        pitch=resultado.get("pitch"),
                        roll=resultado.get("roll"),
                        bbox=resultado.get("bbox"),
                        keypoints=resultado.get("keypoints"),
                        aligned_face=resultado.get("aligned_face"),
                    )

                emb = resultado.get("embedding")
                setattr(ff_data, self._slot_for_model(model_name), emb)

            except Exception as e:
                print(f"Error processing face {image_path} (model={m}): {e}")

        if ff_data is not None:
            face_info.ff_info = ff_data

        return face_info

    def process_dataset(
        self,
        dataset: FaceDataset,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
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
        # PR L1b — propaga `dataset` pra `process_face` resolver paths
        # corretamente quando o dataset estiver em modo RELATIVE.
        for idx, face in enumerate(
            tqdm(dataset, desc=f"Processando {dataset.name}", unit="face")
        ):
            self.process_face(face, model=model, models=models, dataset=dataset)

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
        """Lê imagens do disco lidando com caracteres especiais no caminho."""
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
