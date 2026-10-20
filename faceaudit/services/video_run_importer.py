# faceaudit/services/video_run_importer.py
"""Importador de um run de extração de vídeo → `FaceDataset`.

Fonte ÚNICA de verdade para promover a proveniência por face ao parquet:
lê o manifesto + sidecars de um run produzido, valida os hashes dos
recortes, e monta um `FaceDataset` RELATIVE (apontando para os crops do
próprio run) com a projeção escalar em `extra_data`. É usado tanto pelo
modo produtor→importação manual quanto pelo ponta-a-ponta (Fase 6).

Invariante (Fase 6): o dataset final só inclui faces com
`human_review.status` em `accepted_statuses` (default `("accepted",)`);
a revisão (Fase 3) flipa pending→accepted/rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.video.face_provenance import FaceProvenanceScalarProjection
from faceaudit.services.face_alignment_window_service import ALIGNMENT_MISSING_PIXELS_COLUMN
from faceaudit.services.face_geometry_frame_service import transladar_geometria
from faceaudit.services.hash_service import HashService
from faceaudit.models.column_catalog import (
    ColumnSpec, ColumnRole, ColumnSource, ColumnDType,
    QualityDirection, QualitySubtype,
)

SELECTION_WARNING = (
    "Dataset criado por seleção condicionada a similaridade biométrica com "
    "uma âncora. Não representa amostra populacional neutra."
)

_CATEGORY_VIDEO = "Vídeo (extração)"



def build_video_column_specs() -> Dict[str, ColumnSpec]:
    """`ColumnSpec` para as colunas `video_*`/proveniência promovidas ao
    parquet — sem isso o Schema Report da UI fica sem label/categoria."""
    def meta_str(name, label):
        return ColumnSpec(name=name, label=label, role=ColumnRole.METADATA,
                          source=ColumnSource.DERIVED, dtype=ColumnDType.STR,
                          editable=False, visible_default=False, category=_CATEGORY_VIDEO)

    def meta_num(name, label, dtype):
        return ColumnSpec(name=name, label=label, role=ColumnRole.METADATA,
                          source=ColumnSource.DERIVED, dtype=dtype,
                          editable=False, visible_default=False, category=_CATEGORY_VIDEO)

    def qual(name, label, scale=None):
        return ColumnSpec(name=name, label=label, role=ColumnRole.QUALITY,
                          source=ColumnSource.DERIVED, dtype=ColumnDType.FLOAT,
                          editable=False, visible_default=False, category=_CATEGORY_VIDEO,
                          metric_name=name, direction=QualityDirection.HIGHER_IS_BETTER,
                          scale=scale, quality_subtype=QualitySubtype.AUX)

    specs = {
        "target_label": meta_str("target_label", "Alvo (rótulo)"),
        "similarity_to_anchor": qual("similarity_to_anchor", "Similaridade c/ âncora", (-1.0, 1.0)),
        "source_video_sha256": meta_str("source_video_sha256", "SHA-256 do vídeo"),
        "frame_index": meta_num("frame_index", "Índice do quadro", ColumnDType.INT),
        "timestamp_s_from_fps": meta_num("timestamp_s_from_fps", "Timestamp (s)", ColumnDType.FLOAT),
        "video_extraction_det_score": qual("video_extraction_det_score", "Det. (triagem vídeo)", (0.0, 1.0)),
        "video_extraction_fiqa_score": qual("video_extraction_fiqa_score", "FIQA (triagem vídeo)"),
        "video_det_score_decision": meta_str("video_det_score_decision", "Decisão det (vídeo)"),
        "video_fiqa_score_decision": meta_str("video_fiqa_score_decision", "Decisão FIQA (vídeo)"),
        "video_det_score_policy": meta_str("video_det_score_policy", "Política det (vídeo)"),
        "video_fiqa_score_policy": meta_str("video_fiqa_score_policy", "Política FIQA (vídeo)"),
        "video_min_det_score": meta_num("video_min_det_score", "Min det (vídeo)", ColumnDType.FLOAT),
        "video_min_fiqa_score": meta_num("video_min_fiqa_score", "Min FIQA (vídeo)", ColumnDType.FLOAT),
        "anchor_embedding_fingerprint": meta_str("anchor_embedding_fingerprint", "Fingerprint da âncora"),
        ALIGNMENT_MISSING_PIXELS_COLUMN: ColumnSpec(
            name=ALIGNMENT_MISSING_PIXELS_COLUMN, label="Alinhamento sem pixel (fração)",
            role=ColumnRole.METADATA, source=ColumnSource.DERIVED, dtype=ColumnDType.FLOAT,
            editable=False, visible_default=False, category=_CATEGORY_VIDEO,
            description=(
                "Fração da face alinhada 112×112 que, refeita a partir do recorte, "
                "sairia sem o pixel que a extração teve no quadro (o recorte não "
                "alcançou aquela parte). 0 = realinhar a partir do recorte reproduz "
                "o alinhamento da extração. Acima de 0, um embedding de modelo "
                "acrescentado depois, realinhado do recorte, difere do que seria "
                "extraído do quadro. Runs novos cobrem a janela inteira e dão 0."
            ),
        ),
    }
    # Colunas de captura do vídeo de origem: as mesmas do caminho de fotos,
    # vindas do catálogo em vez de redefinidas aqui. Sem isto elas nasceriam
    # como CUSTOM no dataset de vídeo e teriam rótulo e categoria diferentes
    # das mesmas colunas num dataset de fotos — e as duas se encontram num
    # merge.
    from faceaudit.models.column_catalog import (
        ColumnRole as _Role,
        DatasetType as _Tipo,
        get_default_specs as _padrao,
    )

    for nome, spec in _padrao(_Tipo.FACE).items():
        if spec.role is _Role.CAPTURE and nome not in specs:
            specs[nome] = spec
    return specs


class VideoRunImporter:
    def __init__(self, hash_service: Optional[HashService] = None):
        self._hash = hash_service or HashService()

    def import_run(
        self,
        run_dir,
        *,
        dataset_name: Optional[str] = None,
        accepted_statuses: Tuple[str, ...] = ("accepted",),
        image_kind: str = "aligned",   # "aligned" (face 112×112) | "crop" (recorte c/ margem)
        validate_hashes: bool = True,
        logger: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        enrich: bool = False,
        pipeline: Any = None,
        ingestion_config: Any = None,
        require_review: bool = True,
    ) -> FaceDataset:
        run_dir = Path(run_dir)
        meta_dir = run_dir / "_meta"
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"Pasta _meta/ não encontrada em {run_dir}")
        manifest = self._load_manifest(meta_dir)

        # Invariante END_TO_END: o parquet só é finalizado após o commit da
        # revisão (review_status completed ou skipped-override).
        if enrich and require_review:
            rs = manifest.get("review_status", "not_started")
            if rs not in ("completed", "skipped"):
                raise RuntimeError(
                    f"enrich bloqueado: revisão não concluída (review_status={rs!r}). "
                    f"Conclua a revisão (ou marque skipped como override) antes do dataset final."
                )

        # Colunas de captura por vídeo, calculadas UMA vez. Todas as faces
        # do mesmo arquivo compartilham a mesma instância de cada string
        # (inclusive o dump completo), em vez de uma cópia por face.
        captura_por_video = self._capture_columns_by_video(manifest)
        quadro_por_video = self._frame_size_by_video(manifest)

        faces: List[FaceInfo] = []
        n_skipped_status = n_hash_mismatch = n_missing = 0
        any_aligned = False
        # Recortes cuja biometria vem da face ALINHADA da extração (ver
        # `_load_extraction_aligned_face`), e os que ficaram sem ela.
        n_biometria_da_extracao = n_sem_alinhada = 0
        n_com_captura = 0
        for sidecar in sorted(meta_dir.glob("*.json")):
            nm = sidecar.name
            if nm.startswith("_run_") or nm.startswith("_decisions_") or nm.startswith("_frames_"):
                continue
            try:
                d = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception as e:
                if logger:
                    logger(f"sidecar ilegível {nm}: {e}")
                continue
            status = (d.get("human_review") or {}).get("status", "pending")
            if status not in accepted_statuses:
                n_skipped_status += 1
                continue

            artifact = d.get("artifact") or {}
            label = (d.get("identity") or {}).get("label")
            image_file = artifact.get("image_file")
            if not label or not image_file:
                continue
            # Imagem do dataset conforme `image_kind`:
            #  - "aligned" (default): face ALINHADA 112×112 → enrich extrai o
            #    embedding direto, SEM re-detectar (essencial p/ faces pequenas).
            #    Cai no recorte se o run não tiver a face alinhada.
            #  - "crop": recorte COM MARGEM (evidência) → o enrich re-detecta a
            #    face (pode falhar em faces muito pequenas).
            aligned_file = artifact.get("aligned_image_file")
            if image_kind == "aligned" and aligned_file:
                rel_path = aligned_file
                abs_path = run_dir / aligned_file
                expected = artifact.get("aligned_image_sha256")
                used_aligned = True
            else:
                rel_path = f"{label}/{image_file}"
                abs_path = run_dir / label / image_file
                expected = artifact.get("image_sha256")
                used_aligned = False
            if not abs_path.exists():
                n_missing += 1
                if logger:
                    logger(f"recorte ausente {rel_path}")
                continue
            if validate_hashes and expected:
                actual = self._hash.calculate_file_hash(abs_path)
                if actual and actual != expected:
                    n_hash_mismatch += 1
                    if logger:
                        logger(f"hash divergente {rel_path}")
                    continue

            extra = FaceProvenanceScalarProjection.from_sidecar_dict(d).to_extra_data()
            # Promove os metadados biométricos capturados NA EXTRAÇÃO (o detector
            # roda com extended=True → genderage + pose) para as colunas canônicas
            # via FFInfo — espelha o que o process_image faz no caminho de imagens.
            # Sem isto, gender_est/yaw/pitch/age_est ficavam vazios no parquet,
            # porque o enrich em faces ALINHADAS não recomputa pose/genderage.
            # Como o enrich faz MERGE (só fiqa_score + embeddings sobrescrevem,
            # e só quando não-None), estes campos sobrevivem ao enriquecimento.
            det = d.get("detection") or {}
            pose = det.get("pose") or {}
            qg = d.get("quality_gates") or {}
            akps = det.get("aligned_keypoints")  # 5×2 no sistema 112×112 (KPRPE)
            ff = FFInfo(
                det_score=det.get("det_score"),
                fiqa_score=qg.get("fiqa_score"),
                gender=det.get("gender"),
                age=det.get("age"),
                yaw=pose.get("yaw"),
                pitch=pose.get("pitch"),
                roll=pose.get("roll"),
                # `keypoints` NÃO recebe os pontos alinhados: por
                # invariante o slot é o sistema da imagem que o dataset
                # guarda. Os pontos 112×112 que o KPRPE exige vão para a
                # casa própria deles, em `extra_data`, logo abaixo.
            )
            falta = None
            if not used_aligned:
                # Recorte com margem: a geometria é a da detecção que gerou
                # o recorte, levada do sistema do QUADRO para o do recorte —
                # o invariante do dataset (bbox/keypoints no sistema da
                # imagem que ele guarda). Na face alinhada não há o que
                # gravar: a transformação de similaridade não é translação.
                self._geometry_into_crop(ff, artifact, det)
                falta = self._missing_alignment_pixels(
                    artifact, det,
                    quadro_por_video.get((d.get("source_video") or {}).get("sha256")),
                )
                if enrich:
                    alinhada = self._load_extraction_aligned_face(
                        run_dir, artifact, validate_hashes)
                    if alinhada is False:
                        n_hash_mismatch += 1
                        if logger:
                            logger(f"hash divergente {artifact.get('aligned_image_file')}")
                        continue
                    if alinhada is None:
                        n_sem_alinhada += 1
                    else:
                        # Vai como cache em memória: o FFService extrai o
                        # embedding dela SEM detectar de novo. Redetectar no
                        # recorte derruba a confiança do detector em faces
                        # pequenas (medido: ~0,25–0,30) e a face cai abaixo
                        # do limiar — ou pior, ele acha outra região. Não é
                        # persistida (`to_dict` descarta `aligned_face`).
                        ff.aligned_face = alinhada
                        n_biometria_da_extracao += 1
            # Persiste os keypoints no parquet (via extra_data, JSON) — assim o
            # KPRPE/sepaelv6 funciona tanto no enrich E2E quanto num re-enrich.
            if akps:
                extra["aligned_keypoints_112"] = json.dumps(akps)
            fi = FaceInfo(image_path=rel_path, identity=label, hash=expected, ff_info=ff)
            # Origem = vídeo de origem (análogo ao campo "Origem" do caminho de
            # imagens, que registra a proveniência da face).
            src = d.get("source_video") or {}
            vid = src.get("filename")
            if vid:
                fi.origin = f"Vídeo: {vid}"

            # Casa pelo SHA-256 do vídeo, não pelo nome: um run pode ter
            # vários vídeos, e dois arquivos podem ter o mesmo nome em
            # pastas diferentes.
            base_captura = captura_por_video.get(src.get("sha256"))
            if base_captura is not None:
                from faceaudit.services.video_capture_metadata_service import (
                    columns_for_frame,
                )
                meta_video, colunas_base = base_captura
                extra.update(
                    columns_for_frame(
                        meta_video,
                        extra.get("timestamp_s_from_fps"),
                        colunas_base,
                    )
                )
                n_com_captura += 1

            if falta is not None:
                extra[ALIGNMENT_MISSING_PIXELS_COLUMN] = falta
            fi.extra_data = extra
            faces.append(fi)
            if used_aligned:
                any_aligned = True

        name = dataset_name or run_dir.name
        ds = FaceDataset(name=name, faces=faces, path_mode=PathMode.RELATIVE, path_root=".")
        ds._parquet_dir = run_dir.resolve()
        ds.is_aligned = any_aligned
        ds.column_specs = build_video_column_specs()
        ds.provenance_metadata = self._build_provenance_metadata(manifest)
        if not any_aligned:
            # Recorte com margem: a geometria gravada é a da extração.
            ds.provenance_metadata["geometry_source"] = "video_extraction_detection"
        if n_biometria_da_extracao:
            ds.provenance_metadata["embedding_source"] = (
                "video_extraction_aligned_face"
                if not n_sem_alinhada else "mixed_extraction_aligned_and_redetection"
            )
            if logger:
                logger(
                    f"biometria de {n_biometria_da_extracao} recorte(s) extraída da face "
                    "alinhada da extração (sem nova detecção)"
                    + (f"; {n_sem_alinhada} sem face alinhada no run serão "
                       "redetectados no recorte" if n_sem_alinhada else "")
                )

        if n_com_captura and logger:
            logger(
                f"metadados de captura em {n_com_captura} face(s) "
                f"(data calculada do instante de cada quadro)"
            )

        if logger:
            logger(f"importadas {len(faces)} face(s) | status-pulados={n_skipped_status} "
                   f"| hash-divergentes={n_hash_mismatch} | ausentes={n_missing}")

        if enrich:
            if pipeline is None:
                raise ValueError("enrich=True requer um pipeline (FeatureExtractionPipeline).")
            cfg = ingestion_config or self._default_ingestion_config(ds.name, run_dir)
            # Força is_aligned no config do enrich: o enrich faz
            # `dataset.is_aligned = config.is_aligned`, então sem isto ele
            # resetaria pra False e voltaria a re-detectar.
            try:
                cfg.is_aligned = bool(ds.is_aligned)
            except Exception:
                pass
            # Heads-up: o carregamento dos modelos (N pipelines com vários
            # embeddings) é silencioso e lento — avisa antes de mergulhar nele.
            if progress_callback:
                n_models = len(getattr(cfg, "ff_models", None) or [1])
                progress_callback(
                    f"Enriquecendo: carregando {n_models} modelo(s) e extraindo "
                    "embeddings… (pode levar minutos com vários modelos)", 45)
            if n_biometria_da_extracao:
                from faceaudit.enums.sepael_model import SepaelModel
                ds.record_embedding_source(
                    getattr(cfg, "ff_models", None) or [SepaelModel.default().ff_name],
                    ds.provenance_metadata["embedding_source"],
                )
            try:
                pipeline.enrich_existing_dataset(ds, cfg, progress_callback=progress_callback)
            finally:
                # A face alinhada só servia ao enriquecimento; 112×112×3 por
                # face, em run de milhares de faces, é memória que sobra.
                for face in ds.faces:
                    if face.ff_info is not None:
                        face.ff_info.aligned_face = None
            self._finalize_manifest(meta_dir, run_dir, ds, now=None)

        return ds

    # ---- helpers ----
    @staticmethod
    def _crop_origin(artifact: Dict[str, Any], det: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Origem `(x1, y1)` do recorte no quadro, ou None se não há como saber.

        Runs novos gravam `crop_box_xyxy`. Nos antigos a conta do
        `extend_bbox` é refeita a partir da caixa e do fator de margem — a
        ORIGEM não depende do tamanho do quadro (o recorte só é preso a zero
        à esquerda e acima; a largura do quadro só limita o canto oposto).
        """
        box = artifact.get("crop_box_xyxy")
        if box and len(box) >= 2:
            return int(box[0]), int(box[1])
        bbox = det.get("bbox_xyxy")
        margem = artifact.get("crop_margin_factor")
        if not bbox or margem is None:
            return None
        import numpy as np
        from forensicface.geometry import extend_bbox
        enorme = (2 ** 31, 2 ** 31)
        x1, y1, _, _ = extend_bbox(np.asarray(bbox, dtype=float), enorme, float(margem))
        return int(x1), int(y1)

    @staticmethod
    def _frame_size_by_video(manifest: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        """SHA-256 do vídeo → (largura, altura) do quadro como foi decodificado."""
        saida: Dict[str, Tuple[int, int]] = {}
        for v in manifest.get("source_videos") or []:
            if not isinstance(v, dict) or not v.get("sha256"):
                continue
            res = v.get("resolution_effective")
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except ValueError:
                    res = None
            if isinstance(res, (list, tuple)) and len(res) == 2:
                try:
                    saida[v["sha256"]] = (int(res[0]), int(res[1]))
                except (TypeError, ValueError):
                    pass
        return saida

    @classmethod
    def _missing_alignment_pixels(
        cls, artifact: Dict[str, Any], det: Dict[str, Any],
        quadro: Optional[Tuple[int, int]],
    ) -> Optional[float]:
        """Fração da face 112×112 que o recorte não alcança e o quadro tinha.

        Sem o tamanho do quadro, conta tudo o que sai do recorte (limite
        superior). None quando falta geometria para calcular.
        """
        from faceaudit.services.face_alignment_window_service import fracao_sem_pixel
        origem = cls._crop_origin(artifact, det)
        kps = det.get("keypoints_5pt")
        tamanho = list(artifact.get("image_size_px") or [])
        if origem is None or kps is None or len(tamanho) < 2 or None in tamanho[:2]:
            return None
        ox, oy = origem
        pontos = [[x - ox, y - oy] for x, y in kps]
        existente = None
        if quadro is not None:
            existente = (-ox, -oy, quadro[0] - ox, quadro[1] - oy)
        try:
            return round(fracao_sem_pixel(pontos, int(tamanho[0]), int(tamanho[1]), existente), 6)
        except (ValueError, RuntimeError):
            return None

    @classmethod
    def _geometry_into_crop(cls, ff: FFInfo, artifact: Dict[str, Any], det: Dict[str, Any]) -> None:
        """Grava em `ff` a caixa e os 5 pontos da extração, no sistema do recorte."""
        import numpy as np
        origem = cls._crop_origin(artifact, det)
        bbox = det.get("bbox_xyxy")
        kps = det.get("keypoints_5pt")
        if origem is None or (bbox is None and kps is None):
            return
        if bbox is not None:
            ff.bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
        if kps is not None:
            ff.keypoints = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        tamanho = list(artifact.get("image_size_px") or [])
        largura, altura = (tamanho + [None, None])[:2]
        transladar_geometria(ff, origem[0], origem[1], largura, altura)

    def _load_extraction_aligned_face(self, run_dir: Path, artifact: Dict[str, Any], validate_hashes: bool):
        """Face alinhada 112×112 da extração, em RGB.

        None quando o run não a tem (runs antigos) ou ela não lê; False
        quando o hash diverge do registrado — a mesma política do recorte:
        arquivo adulterado não entra.
        """
        rel = artifact.get("aligned_image_file")
        if not rel:
            return None
        caminho = run_dir / rel
        if not caminho.exists():
            return None
        esperado = artifact.get("aligned_image_sha256")
        if validate_hashes and esperado:
            atual = self._hash.calculate_file_hash(caminho)
            if atual and atual != esperado:
                return False
        import cv2
        import numpy as np
        bgr = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (112, 112, 3):
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _default_ingestion_config(name: str, run_dir: Path) -> Any:
        """DatasetIngestionConfig mínimo p/ enrich (FF apenas; input_path é
        ignorado pelo enrich, mas o DTO o exige)."""
        from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
        return DatasetIngestionConfig(
            input_path=run_dir, dataset_name=name,
            compute_ff=True, compute_ofiq=False, compute_serfiq=False,
        )

    def _finalize_manifest(self, meta_dir: Path, run_dir: Path, ds: FaceDataset, now=None) -> None:
        """Após o enrich, grava no manifesto dataset_status=enriched +
        final_dataset_path/sha256/n_faces (fecha a trilha run→parquet)."""
        manifests = sorted(meta_dir.glob("_run_*.json"))
        if not manifests:
            return
        parquet_path = Path(ds._parquet_dir or run_dir) / f"{ds.name}.parquet"
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        manifest["dataset_status"] = "enriched"
        manifest["final_dataset"] = {
            "path": parquet_path.name,
            "sha256": self._hash.calculate_file_hash(parquet_path) if parquet_path.exists() else None,
            "n_faces": len(ds),
        }
        manifests[0].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Mantém a cópia legível da raiz em sincronia com o autoritativo (senão
        # ficaria congelada em dataset_status=not_imported / sem final_dataset).
        from faceaudit.services.video_provenance_writer import mirror_manifest_to_root
        mirror_manifest_to_root(run_dir)

    def audit_metadata(self, run_dir) -> Dict[str, Any]:
        """Metadata para o audit log VIDEO_EXTRACT (emitido pela UI na Fase 5)."""
        manifest = self._load_manifest(Path(run_dir) / "_meta")
        summary = manifest.get("summary") or {}
        review = manifest.get("review") or {}
        src_videos = manifest.get("source_videos") or (
            [manifest.get("source_video")] if manifest.get("source_video") else [])
        return {
            "run_id": manifest.get("run_id"),
            # 1º vídeo (compat) + lista completa (multivídeo) para a cadeia de custódia.
            "video_sha256": (manifest.get("source_video") or {}).get("sha256"),
            "video_sha256_all": [(v or {}).get("sha256") for v in src_videos],
            "n_videos": summary.get("n_videos"),
            "n_frames_sampled": summary.get("n_frames_sampled"),
            "n_accepted_auto": summary.get("n_accepted_auto"),
            "n_accepted_after_review": review.get("n_accepted"),
            "n_rejected_human": review.get("n_rejected"),
            "review_skipped": review.get("skipped"),
        }

    @staticmethod
    def _load_manifest(meta_dir: Path) -> Dict[str, Any]:
        candidates = sorted(meta_dir.glob("_run_*.json"))
        if not candidates:
            return {}
        try:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _capture_columns_by_video(manifest: Dict[str, Any]) -> Dict[str, Any]:
        """SHA-256 do vídeo → (metadados, colunas base). Nunca levanta.

        Lê do manifesto, não do arquivo: o run é portátil e o vídeo pode já
        não estar no caminho original. O que está aqui foi capturado na
        extração, junto com o hash — é o estado do arquivo naquele momento.

        As colunas base são idênticas para todas as faces do mesmo vídeo;
        calculá-las uma vez faz todas compartilharem as mesmas strings.
        """
        bruto = manifest.get("capture_metadata") or {}
        if not isinstance(bruto, dict) or not bruto:
            return {}

        from faceaudit.services.exif_extraction_service import (
            ImageMetadata,
            build_capture_columns,
        )
        from faceaudit.services.video_capture_metadata_service import (
            VideoCaptureMetadata,
        )

        saida: Dict[str, Any] = {}
        for chave, entrada in bruto.items():
            if not isinstance(entrada, dict):
                continue
            tags = entrada.get("tags") or {}
            meta = VideoCaptureMetadata.from_manifest_entry(entrada)
            # Sem o dump, não há colunas a promover — mas a data por quadro
            # ainda pode ser calculada, porque o início vem do manifesto.
            colunas = (
                build_capture_columns(ImageMetadata(source_path="", tags=tags))
                if tags else {}
            )
            saida[chave] = (meta, colunas)
        return saida

    @staticmethod
    def _build_provenance_metadata(manifest: Dict[str, Any]) -> Dict[str, str]:
        targets = manifest.get("targets") or []
        model = manifest.get("recognition_model")
        threshold = targets[0].get("similarity_threshold") if targets else None
        run_id = manifest.get("run_id")
        meta = {
            "dataset_origin_type": "VIDEO_TARGET_EXTRACTION",
            "selection_conditioned_on_embedding": "true",
            "selection_warning": SELECTION_WARNING,
        }
        if model:
            meta["selection_anchor_model"] = str(model)
        if threshold is not None:
            meta["selection_threshold"] = str(threshold)
        if run_id:
            meta["source_run_id"] = str(run_id)

        # Rotação aplicada aos quadros. Sem isto, ninguém reconstrói depois
        # por que os recortes estão de pé e o vídeo, aberto num player,
        # aparece deitado.
        rotacoes = {
            int((e or {}).get("container_rotation") or 0)
            for e in (manifest.get("capture_metadata") or {}).values()
            if isinstance(e, dict)
        } - {0}
        if rotacoes:
            meta["container_rotation_applied"] = ",".join(
                str(r) for r in sorted(rotacoes)
            )
        return meta
