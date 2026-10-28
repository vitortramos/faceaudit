"""ImagePipelineBatchService — aplica pipeline de imagens em todo o
dataset, produzindo um dataset derivado AUTOCONTIDO.

Saída, numa pasta qualquer escolhida pelo usuário:

    <pasta escolhida>/<nome>/
    ├── <nome>.parquet               ← dataset derivado (RELATIVE, path_root=images)
    ├── <nome>.audit.json            ← cadeia de custódia, nasce com CREATE
    ├── <nome>__errors.parquet       ← falhas (se houver)
    ├── manifest.json                ← pipeline aplicado + contagens
    └── images/
        └── <file>.png               ← imagens processadas

Mesma convenção dos outros fluxos que produzem dataset autocontido
(`self_contained_dataset_dir` + `SELF_CONTAINED_IMAGES_SUBDIR`): a pasta
é uma unidade e pode ser movida, compactada ou levada para outra máquina
sem quebrar os caminhos. Antes o parquet ia para a biblioteca global e
as imagens para outra pasta, com caminhos absolutos — mover qualquer um
dos dois quebrava o dataset.

O que o derivado leva de cada face:
  - identidade, origem, etnia, gênero e as colunas que descrevem a
    PESSOA ou o REGISTRO (role CUSTOM — colunas de template como idade —
    e colunas manuais), com os specs de coluna.
  - NÃO leva o que foi medido na imagem de origem (embedding, qualidade,
    pose, geometria, metadados de captura): a imagem mudou e essas
    medidas deixaram de valer. O perito roda Enriquecer depois.

Medidas de degradação: cada imagem processada é medida na região da
face (ver `image_degradation_profile_service`), e o manifesto e a cadeia
de custódia registram o resumo do conjunto e — quando o pipeline traz a
questionada que reproduz — a comparação da mediana com ela.

Determinismo: `seed_jitter = face.image_path` propaga pro pipeline.
Idempotência via `skip_existing=True`: PNG já existe → não regrava (a
imagem é processada em memória mesmo assim, para ser medida).

Service stateless. Construa uma instância por chamada ou compartilhe.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.column_catalog import ColumnRole, ColumnSource
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)
from faceaudit.models.image_processing.batch_processing_result import (
    BatchProcessingResult,
)
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.pipelines.image_processing_pipeline import (
    ImageProcessingPipeline,
)
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.dataset_localization_service import (
    SELF_CONTAINED_IMAGES_SUBDIR,
    sanitize_dir_name,
    self_contained_dataset_dir,
)
from faceaudit.services.hash_service import HashService
from faceaudit.services.image_degradation_profile_service import (
    DegradationMeasures,
    agreement,
    compare_summary,
    measure,
    summarize,
)


_MANIFEST_NAME = "manifest.json"

# Colunas que descrevem a pessoa ou o registro, e por isso continuam
# verdadeiras depois que a imagem é degradada.
_ROLES_QUE_ACOMPANHAM = frozenset({ColumnRole.CUSTOM})
_FONTES_QUE_ACOMPANHAM = frozenset({ColumnSource.MANUAL, ColumnSource.CUSTOM})
# Mesmo de fonte manual, estas descrevem a IMAGEM.
_ROLES_DA_IMAGEM = frozenset({
    ColumnRole.EMBEDDING, ColumnRole.QUALITY, ColumnRole.POSE,
    ColumnRole.GEOMETRY, ColumnRole.CAPTURE,
})


def colunas_que_acompanham(dataset: FaceDataset) -> Tuple[List[str], List[str]]:
    """Separa as colunas extras do dataset em (levadas, descartadas).

    Levada: descreve a pessoa ou o registro (coluna de template, anotação
    manual). Descartada: foi medida na imagem de origem. Coluna sem spec
    é levada — no parquet ela já nasceria como CUSTOM.
    """
    specs = dataset.column_specs or {}
    nomes: List[str] = []
    vistos = set()
    for face in dataset.faces:
        for k in (face.extra_data or {}):
            if k not in vistos:
                vistos.add(k)
                nomes.append(k)
    levadas, descartadas = [], []
    for nome in nomes:
        spec = specs.get(nome)
        if spec is None:
            levadas.append(nome)
            continue
        role = getattr(spec, "role", None)
        fonte = getattr(spec, "source", None)
        if role in _ROLES_DA_IMAGEM:
            descartadas.append(nome)
        elif role in _ROLES_QUE_ACOMPANHAM or fonte in _FONTES_QUE_ACOMPANHAM:
            levadas.append(nome)
        else:
            descartadas.append(nome)
    return levadas, descartadas


def geometria_do_registro(
    face: FaceInfo, dataset: FaceDataset,
) -> Optional[ImageFaceGeometry]:
    """Geometria da face no sistema da imagem que o dataset guarda.

    A do `ff_info` (invariante do dataset). Em dataset de faces alinhadas
    sem ela, os pontos 112×112 da casa própria — a imagem guardada É a
    face alinhada, então estão no sistema dela.
    """
    geom = ImageFaceGeometry.from_face_info(face)
    if geom is not None or not getattr(dataset, "is_aligned", False):
        return geom
    from faceaudit.services.ff_service import ALIGNED_KEYPOINTS_EXTRA_KEY

    bruto = (face.extra_data or {}).get(ALIGNED_KEYPOINTS_EXTRA_KEY)
    if not bruto:
        return None
    try:
        return ImageFaceGeometry.from_values(None, json.loads(bruto))
    except (TypeError, ValueError):
        return None


def derivado_e_alinhado(
    dataset: FaceDataset, pipeline_config: ImageProcessingPipelineConfig,
) -> bool:
    """O derivado é face alinhada 112×112?

    Sim quando o pipeline alinha antes dos steps, ou quando a origem já
    era alinhada — desde que nenhum step recorte, porque o recorte tira a
    imagem do gabarito. Sem a marca, o Enriquecer seguinte redetectaria a
    face dentro da face alinhada.
    """
    recorta = any(
        s.step_type == ImageProcessingStepType.CROP for s in pipeline_config.steps
    )
    if recorta:
        return False
    return (
        pipeline_config.input_source == ImageInputSource.ALIGNED_FACE
        or bool(getattr(dataset, "is_aligned", False))
    )


class ImagePipelineBatchService:
    """Aplica pipeline em todo dataset → dataset derivado autocontido.
    Stateless."""

    def run(
        self,
        dataset: FaceDataset,
        pipeline_config: ImageProcessingPipelineConfig,
        base_dir: Path,
        *,
        dataset_name: Optional[str] = None,
        ff_service=None,
        skip_existing: bool = True,
        source_parquet_path: Optional[Path] = None,
        source_has_unsaved_changes: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchProcessingResult:
        """Aplica pipeline em todo dataset.

        Args:
            dataset: dataset de entrada (não é mutado).
            pipeline_config: configuração do pipeline a aplicar.
            base_dir: pasta escolhida pelo usuário. O dataset ganha pasta
                própria dentro dela (`<base_dir>/<nome>/`).
            dataset_name: nome do dataset derivado (também nome da pasta e
                do parquet). Default = `<dataset>__<pipeline>`.
            ff_service: necessário se `input_source=ALIGNED_FACE` ou
                algum step usa FFService (ex: CropMode.FACE_DETECTED).
            skip_existing: idempotência — PNGs já existentes são pulados.
            source_parquet_path: parquet do dataset de origem, para a
                cadeia de custódia do derivado registrar de onde veio.
            source_has_unsaved_changes: a origem em memória difere do
                arquivo (o hash registrado não descreve o que foi usado).
            progress_callback: chamado por face com `(idx, total, msg)`.

        Returns:
            `BatchProcessingResult` agregando totais e caminhos.
        """
        # O nome vira pasta e arquivo: os dois precisam do mesmo nome limpo.
        nome = sanitize_dir_name(
            (dataset_name or "").strip() or self._build_derived_name(
                dataset.name, pipeline_config.name,
            )
        )
        dataset_dir = self_contained_dataset_dir(base_dir, nome).resolve()
        images_dir = dataset_dir / SELF_CONTAINED_IMAGES_SUBDIR
        images_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        pipeline = ImageProcessingPipeline(
            pipeline_config, ff_service=ff_service,
        )
        levadas, descartadas = colunas_que_acompanham(dataset)

        new_faces: List[FaceInfo] = []
        medidas: List[DegradationMeasures] = []
        failures: List[Tuple[str, str]] = []
        n_processed = 0
        n_succeeded = 0
        n_skipped = 0
        used_names: set[str] = set()
        hash_service = HashService()

        for idx, face in enumerate(dataset.faces):
            n_processed += 1
            if progress_callback:
                progress_callback(idx, len(dataset), face.image_path or "")

            try:
                result = self._process_one(
                    face, dataset, pipeline,
                    pipeline_config.input_source,
                    images_dir, used_names, skip_existing,
                    ff_service, hash_service, levadas,
                )
            except Exception as e:
                failures.append((face.image_path or "<empty>", str(e)))
                continue

            if result["status"] == "skipped":
                n_skipped += 1
            else:
                n_succeeded += 1
            new_faces.append(result["face"])
            used_names.add(result["rel_name"])
            medidas.append(result["measures"])

        specs_origem = dataset.column_specs or {}
        new_dataset = FaceDataset(
            name=nome,
            description=(
                f"Derivado de '{dataset.name}' via pipeline "
                f"'{pipeline_config.name}'."
            ),
            faces=new_faces,
            column_specs={k: specs_origem[k] for k in levadas if k in specs_origem},
            path_mode=PathMode.RELATIVE,
            path_root=SELF_CONTAINED_IMAGES_SUBDIR,
            is_aligned=derivado_e_alinhado(dataset, pipeline_config),
        )
        new_dataset._parquet_dir = dataset_dir

        parquet_path = dataset_dir / f"{nome}.parquet"
        new_dataset.to_parquet(parquet_path)

        error_parquet_path: Optional[Path] = None
        if failures:
            error_parquet_path = dataset_dir / f"{nome}__errors.parquet"
            self._save_error_parquet(
                dataset, failures, error_parquet_path, nome,
            )

        duration = time.time() - start_time

        source_sha = None
        if source_parquet_path is not None and Path(source_parquet_path).exists():
            source_sha = hash_service.calculate_file_hash(Path(source_parquet_path))

        questionada = pipeline_config.questioned
        resumo = summarize(medidas) if medidas else {}
        alvo = None if questionada is None else questionada.measures
        degradacao = {
            "questioned": None if questionada is None else questionada.to_dict(),
            "n_measured": len(medidas),
            "summary": resumo,
            "comparison": compare_summary(alvo, resumo, medidas) if medidas else [],
            # Uma a uma: a mediana sozinha esconde um conjunto que mistura
            # imagens claras e escuras demais.
            "agreement": agreement(alvo, medidas),
        }

        manifest = {
            "pipeline_name": pipeline_config.name,
            "pipeline_seed": pipeline_config.seed,
            "input_source": pipeline_config.input_source.value,
            "step_types": [s.step_type.value for s in pipeline_config.steps],
            "n_steps": len(pipeline_config.steps),
            "n_processed": n_processed,
            "n_succeeded": n_succeeded,
            "n_skipped": n_skipped,
            "n_failed": len(failures),
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "duration_seconds": round(duration, 3),
            "source_dataset": dataset.name,
            "source_parquet_path": str(source_parquet_path) if source_parquet_path else None,
            "source_parquet_sha256": source_sha,
            "columns_carried": levadas,
            "columns_dropped": descartadas,
            "degradation_measures": degradacao,
            "operator": AuditLogService.current_operator(),
            "pipeline_config": pipeline_config.to_dict(),
        }
        manifest_path = dataset_dir / _MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._audit_criacao(
            parquet_path, new_dataset, dataset, pipeline_config, manifest,
            manifest_path, source_sha, source_parquet_path,
            source_has_unsaved_changes, hash_service,
        )

        return BatchProcessingResult(
            images_dir=images_dir,
            parquet_path=parquet_path,
            pipeline_name=pipeline_config.name,
            n_processed=n_processed,
            n_succeeded=n_succeeded,
            n_skipped=n_skipped,
            n_failed=len(failures),
            failures=tuple(failures),
            duration_seconds=duration,
            error_parquet_path=error_parquet_path,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            degradation=degradacao,
        )

    # ── Cadeia de custódia do derivado ─────────────────────────────────────

    @staticmethod
    def _audit_criacao(
        parquet_path: Path,
        new_dataset: FaceDataset,
        source: FaceDataset,
        pipeline_config: ImageProcessingPipelineConfig,
        manifest: Dict[str, Any],
        manifest_path: Path,
        source_sha: Optional[str],
        source_parquet_path: Optional[Path],
        source_has_unsaved_changes: bool,
        hash_service: HashService,
    ) -> None:
        """Primeira entrada da cadeia do derivado.

        Quem abrir o derivado precisa saber, pela própria cadeia, que as
        imagens foram degradadas de propósito, por qual pipeline e a partir
        de qual arquivo. A entrada do dataset de ORIGEM (COPY_IMAGES) não
        basta: ela não viaja com o derivado.
        """
        passos = " → ".join(
            s.step_type.label_pt for s in pipeline_config.steps
        ) or "nenhum step"
        descricao = (
            f"Dataset derivado de '{source.name}' pelo pipeline de imagens "
            f"'{pipeline_config.name}' ({passos}). As imagens foram "
            f"transformadas de propósito para reproduzir as condições de "
            f"uma imagem questionada."
        )
        metadata = {
            "operation": "image_pipeline_batch",
            "rows": len(new_dataset),
            "identities": new_dataset.number_of_identities,
            "source_dataset": source.name,
            "source_parquet_path": str(source_parquet_path) if source_parquet_path else None,
            "source_parquet_sha256": source_sha,
            "source_has_unsaved_changes": bool(source_has_unsaved_changes),
            "pipeline_config": pipeline_config.to_dict(),
            "manifest_sha256": hash_service.calculate_file_hash(manifest_path),
            "n_processed": manifest["n_processed"],
            "n_succeeded": manifest["n_succeeded"],
            "n_skipped": manifest["n_skipped"],
            "n_failed": manifest["n_failed"],
            "columns_carried": manifest["columns_carried"],
            "columns_dropped": manifest["columns_dropped"],
            "is_aligned": bool(new_dataset.is_aligned),
        }
        degradacao = manifest.get("degradation_measures") or {}
        questionada = degradacao.get("questioned")
        if questionada:
            metadata["questioned_file_name"] = questionada.get("file_name")
            metadata["questioned_sha256"] = questionada.get("sha256")
        if degradacao.get("comparison"):
            metadata["degradation_comparison"] = degradacao["comparison"]
        if degradacao.get("agreement"):
            metadata["degradation_agreement"] = degradacao["agreement"]
        AuditLogService.append(
            parquet_path,
            AuditOperation.CREATE,
            descricao,
            metadata=metadata,
            dataset_hash=hash_service.calculate_file_hash(parquet_path),
        )

    # ── Per-face ───────────────────────────────────────────────────────────

    def _process_one(
        self,
        face: FaceInfo,
        dataset: FaceDataset,
        pipeline: ImageProcessingPipeline,
        input_source: ImageInputSource,
        images_dir: Path,
        used_names: set[str],
        skip_existing: bool,
        ff_service,
        hash_service: HashService,
        colunas_levadas: List[str],
    ) -> dict:
        """Processa uma face. Retorna dict com `status` (success/skipped)
        + `face` (FaceInfo novo) + `rel_name`."""
        src_path = dataset.resolve_image_path(face)
        if not src_path.exists():
            raise RuntimeError(f"arquivo fonte não encontrado: {src_path}")

        rel_name = self._build_dest_name(face, src_path, used_names)
        dst = images_dir / rel_name

        # Carrega imagem via open+imdecode — `cv2.imread` falha
        # SILENCIOSAMENTE em paths com chars não-ASCII no Windows.
        try:
            with open(src_path, "rb") as fh:
                file_bytes = fh.read()
            bgr = cv2.imdecode(
                np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR,
            )
        except OSError as exc:
            raise RuntimeError(f"falha ao ler {src_path.name}: {exc}") from exc
        if bgr is None:
            raise RuntimeError(f"falha ao decodificar: {src_path.name}")

        if input_source == ImageInputSource.ALIGNED_FACE:
            if ff_service is None:
                raise RuntimeError(
                    "input_source=ALIGNED_FACE requer FFService"
                )
            aligned, aligned_kps = ff_service.align_face_from_array(bgr)
            if aligned is None:
                raise RuntimeError("nenhuma face detectada")
            bgr = aligned
            geom = ImageFaceGeometry.from_values(None, aligned_kps)
        else:
            geom = geometria_do_registro(face, dataset)

        jitter = str(face.image_path or "")
        saida = pipeline.process(bgr, seed_jitter=jitter, face=geom)
        processed = saida.image
        medidas = measure(processed, saida.face)

        # Idempotência: PNG já existe → não regrava. A imagem foi
        # processada assim mesmo, para ser medida.
        if skip_existing and dst.exists():
            existing_hash = hash_service.calculate_file_hash(dst)
            new_face = self._clone_face_minimal(
                face, rel_name, existing_hash, colunas_levadas,
            )
            return {"status": "skipped", "face": new_face,
                    "rel_name": rel_name, "measures": medidas}

        # Grava PNG (lossless) via imencode+write_bytes — `cv2.imwrite`
        # tem o mesmo bug de não-ASCII no Windows.
        dst.parent.mkdir(parents=True, exist_ok=True)
        success, buf = cv2.imencode(".png", processed)
        if not success:
            raise RuntimeError("cv2.imencode retornou False")
        try:
            dst.write_bytes(buf.tobytes())
        except OSError as exc:
            raise RuntimeError(f"falha ao gravar {dst.name}: {exc}") from exc

        file_hash = hash_service.calculate_file_hash(dst)
        new_face = self._clone_face_minimal(
            face, rel_name, file_hash, colunas_levadas,
        )
        return {"status": "success", "face": new_face,
                "rel_name": rel_name, "measures": medidas}

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_dest_name(
        face: FaceInfo, src_path: Path, used_names: set,
    ) -> str:
        """PRESERVE com `.png` forçado. Em colisão (a.png já existe na
        rodada atual), sufixa `_1`, `_2`, ..."""
        base = face.file_name or src_path.name
        stem = Path(base).stem
        candidate = f"{stem}.png"
        if candidate not in used_names:
            return candidate
        i = 1
        while True:
            candidate = f"{stem}_{i}.png"
            if candidate not in used_names:
                return candidate
            i += 1

    @staticmethod
    def _clone_face_minimal(
        face: FaceInfo,
        rel_image_path: str,
        new_hash: Optional[str],
        colunas_levadas: List[str],
    ) -> FaceInfo:
        """FaceInfo do derivado: caminho relativo à pasta de imagens,
        hash do PNG, dados da pessoa e as colunas que a acompanham.

        Descarta ff_info, ofiq_info, serfiq_info e o que foi medido na
        imagem de origem — a imagem mudou (perito enriquece depois).
        """
        extra = face.extra_data or {}
        return FaceInfo(
            image_path=rel_image_path.replace("\\", "/"),
            file_name=Path(rel_image_path).name,
            hash=new_hash,
            identity=face.identity,
            origin=face.origin,
            ethnicity=face.ethnicity,
            gender=face.gender,
            extra_data={k: extra[k] for k in colunas_levadas if k in extra},
        )

    @staticmethod
    def _build_derived_name(dataset_name: str, pipeline_name: str) -> str:
        """`<dataset>__<pipeline>` — sanitizado pra safe filename."""
        bad = '\\/:*?"<>| '
        ds = dataset_name or "dataset"
        pn = pipeline_name or "pipeline"
        for ch in bad:
            ds = ds.replace(ch, "_")
            pn = pn.replace(ch, "_")
        return f"{ds}__{pn}"

    @staticmethod
    def _save_error_parquet(
        source_dataset: FaceDataset,
        failures: List[Tuple[str, str]],
        error_parquet_path: Path,
        output_stem: str,
    ) -> None:
        """Salva parquet com 1 linha por face que falhou + coluna
        `error_message`. Schema mínimo (image_path/identity/file_name/
        hash/error_message) — perito abre como dataset normal pra
        ordenar/filtrar/inspecionar e gerar lista pra reanálise.

        Os caminhos apontam para a IMAGEM-FONTE (o PNG não chegou a ser
        gerado), absolutos: o arquivo mora na pasta do derivado, e a
        âncora relativa da origem não vale a partir dali.
        """
        by_path = {f.image_path: f for f in source_dataset.faces}
        error_faces: List[FaceInfo] = []
        messages: List[str] = []
        for img_path, msg in failures:
            src = by_path.get(img_path)
            if src is None:
                error_faces.append(FaceInfo(image_path=img_path))
            else:
                try:
                    absoluto = source_dataset.resolve_image_path(src).as_posix()
                except Exception:
                    absoluto = src.image_path
                error_faces.append(FaceInfo(
                    image_path=absoluto,
                    file_name=src.file_name,
                    identity=src.identity,
                    hash=src.hash,
                    origin=src.origin,
                    ethnicity=src.ethnicity,
                    gender=src.gender,
                ))
            messages.append(msg)

        error_ds = FaceDataset(
            name=f"{output_stem}__errors",
            description=(
                f"Falhas do pipeline em '{source_dataset.name}'. "
                f"Coluna `error_message` traz a razão de cada falha."
            ),
            faces=error_faces,
            path_mode=PathMode.ABSOLUTE,
            path_root=None,
        )
        error_ds.to_parquet(error_parquet_path)

        # Injeta coluna `error_message` no parquet recém-escrito.
        # Não dá pra passar via FaceInfo (modelo não tem o campo),
        # então re-escreve com a coluna extra preservando a metadata.
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(error_parquet_path)
            if table.num_rows == len(messages):
                meta = table.schema.metadata or {}
                table = table.append_column(
                    "error_message", pa.array(messages, type=pa.string()),
                )
                table = table.replace_schema_metadata(meta)
                pq.write_table(table, error_parquet_path)
        except Exception:
            # Best-effort — sem a coluna, o parquet de erros ainda
            # lista os paths das falhas; o motivo pode ser inferido
            # do manifest.json (que registra os totals).
            pass
