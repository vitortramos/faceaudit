# faceaudit/services/video_frame_face_extraction_service.py
"""Serviço núcleo: extração de faces de um alvo ao longo de um vídeo.

O FaceAudit é dono do laço (cv2.VideoCapture); o ForensicFace entra só
para DETECTAR + alinhar (sem embedding) e para gerar o embedding de
re-identificação. A identidade é o árbitro (gate `similarity_to_anchor`);
det_score/fiqa_score são gates de qualidade configuráveis (default
TAG_ONLY na v1, só registram).

Fase 1 (produtor mínimo): 1 alvo, gate de identidade greedy por quadro,
sem tracking pesado. O assignment N×M + tracking entram na Fase 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from forensicface.geometry import extend_bbox

from faceaudit.services.hash_service import HashService
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.video_provenance_writer import VideoProvenanceWriter
from faceaudit.services.target_assignment import (
    Candidate, TargetState, assign_frame, cosine,
)
from faceaudit.enums.quality_gate_policy import QualityGatePolicy
from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.video.video_info import VideoInfo
from faceaudit.models.video.video_target import VideoTarget
from faceaudit.models.video.video_extraction_config import VideoExtractionConfig
from faceaudit.models.video.video_extraction_result import VideoExtractionResult
from faceaudit.models.video.face_provenance import FaceProvenance


class VideoOpenError(RuntimeError):
    """O arquivo de vídeo não pôde ser aberto (codec/path não-ASCII/corrompido)."""


@dataclass
class AlignedFaceResult:
    """Resultado de alinhar uma detecção específica (clicada) para a âncora."""
    aligned_face_bgr: np.ndarray
    aligned_keypoints: Optional[np.ndarray]
    bbox_xyxy: List[float]
    det_score: float
    keypoints_5pt: List[List[float]]
    align_matrix_2x3: Optional[List[List[float]]]
    align_matrix_available: bool
    pose: Optional[Dict[str, float]]
    gender: Optional[str]
    age: Optional[float]


def _gate_decision(
    policy: QualityGatePolicy,
    score: Optional[float],
    min_score: Optional[float],
) -> str:
    """Decisão de um gate de qualidade. Nunca derruba acima do limiar."""
    if policy == QualityGatePolicy.DISABLED:
        return "disabled"
    if score is None or min_score is None:
        return "passed"  # política ativa mas sem limiar avaliável → só registra
    if score >= min_score:
        return "passed"
    if policy == QualityGatePolicy.TAG_ONLY:
        return "tag_low"
    if policy == QualityGatePolicy.ROUTE_TO_REVIEW:
        return "route_to_review"
    if policy == QualityGatePolicy.HARD_REJECT:
        return "hard_reject"
    return "passed"


def _pose_to_dict(pose: Any) -> Optional[Dict[str, float]]:
    if pose is None:
        return None
    try:
        yaw, pitch, roll = pose
        return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
    except Exception:
        return None


def _utc_now_iso() -> str:
    """Timestamp UTC ISO-8601 timezone-aware (cadeia de custódia)."""
    return datetime.now(timezone.utc).isoformat()


def _library_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"opencv": cv2.__version__, "numpy": np.__version__}
    for mod_name, key in (("forensicface", "forensicface"), ("onnxruntime", "onnxruntime"), ("faceaudit", "faceaudit")):
        try:
            mod = __import__(mod_name)
            versions[key] = getattr(mod, "__version__", None)
        except Exception:
            versions[key] = None
    return versions


class VideoFrameFaceExtractionService:
    def __init__(self, ff_service, hash_service: Optional[HashService] = None):
        self._ff = ff_service
        self._engine = ff_service.app
        self._hash = hash_service or HashService()

    # ─────────────────────────── probe / leitura ───────────────────────────

    def probe_video(self, video_path) -> VideoInfo:
        p = Path(video_path)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(
                f"Não foi possível abrir o vídeo: {video_path}. "
                f"Verifique o codec e se o caminho contém caracteres especiais."
            )
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            try:
                backend = cap.getBackendName()
            except Exception:
                backend = None
        finally:
            cap.release()
        duration = (frame_count / fps) if fps > 0 else 0.0
        sha = self._hash.calculate_file_hash(p)
        size = p.stat().st_size if p.exists() else 0
        return VideoInfo(
            path=str(p),
            filename=p.name,
            sha256=sha,
            file_size_bytes=size,
            fps_reported=fps,
            frame_count_reported=frame_count,
            duration_reported_s=duration,
            resolution=(width, height),
            opencv_version=cv2.__version__,
            decoder_backend=backend,
        )

    def read_frame(self, video_path, index: int) -> Optional[np.ndarray]:
        """Lê um quadro específico (uso do scrubber da UI; seek pontual)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ret, frame = cap.read()
        finally:
            cap.release()
        return frame if ret else None

    def detect_in_frame(self, frame_bgr: np.ndarray) -> List[dict]:
        """Detecta + alinha todas as faces do quadro (SEM embedding)."""
        try:
            res = self._engine.detect_and_align(frame_bgr, single_face=False)
        except Exception as e:
            print(f"⚠️ [VideoExtraction] detect_and_align falhou: {e}")
            return []
        return res or []

    # ─────────────────────────── âncora (clique) ───────────────────────────

    @staticmethod
    def detection_containing_point(detections: List[dict], x: float, y: float) -> Optional[int]:
        """Índice da detecção cuja bbox contém (x, y); em sobreposição,
        a de menor área (mais específica)."""
        best_idx, best_area = None, None
        for i, d in enumerate(detections):
            bb = np.asarray(d.get("bbox")).astype(float).flatten()
            if bb.size < 4:
                continue
            x1, y1, x2, y2 = bb[:4]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if best_area is None or area < best_area:
                    best_area, best_idx = area, i
        return best_idx

    def align_detected_face_from_array(self, frame_bgr: np.ndarray, detection: dict) -> AlignedFaceResult:
        """Alinha EXATAMENTE a detecção dada (não 'melhor face')."""
        kps = np.asarray(detection["keypoints"], dtype=np.float64)
        align_matrix, available = None, False
        try:
            M = self._engine.backend.estimate_norm(kps)
            align_matrix = np.asarray(M, dtype=float).tolist()
            available = True
        except Exception:
            available = False
        rgb_aligned = detection.get("aligned_face")
        bgr_aligned = cv2.cvtColor(rgb_aligned, cv2.COLOR_RGB2BGR) if rgb_aligned is not None else None
        aligned_kps = detection.get("aligned_keypoints")
        return AlignedFaceResult(
            aligned_face_bgr=bgr_aligned,
            aligned_keypoints=np.asarray(aligned_kps) if aligned_kps is not None else None,
            bbox_xyxy=np.asarray(detection["bbox"]).astype(float).flatten()[:4].tolist(),
            det_score=float(detection.get("det_score", 0.0)),
            keypoints_5pt=kps.tolist(),
            align_matrix_2x3=align_matrix,
            align_matrix_available=available,
            pose=_pose_to_dict(detection.get("pose")),
            gender=detection.get("gender"),
            age=detection.get("age"),
        )

    def compute_anchor_embedding(self, detection: dict, model: Optional[str] = None) -> Optional[np.ndarray]:
        """Embedding da face já alinhada da detecção (para a âncora)."""
        rgb = detection.get("aligned_face")
        if rgb is None:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        akps = detection.get("aligned_keypoints")
        ff = self._ff.extract_embeddings_from_aligned_face(bgr, aligned_keypoints=akps, model=model)
        return self._embedding_from_ffinfo(ff, model)

    def build_target_from_click(
        self,
        frame_bgr: np.ndarray,
        x: float,
        y: float,
        *,
        label: str,
        similarity_threshold: float,
        model: Optional[str] = None,
        frame_index: Optional[int] = None,
    ) -> Optional[Tuple[VideoTarget, int, List[dict]]]:
        """Constrói o alvo a partir de um clique. Retorna (target, det_idx,
        detections) ou None se nenhuma detecção contém o ponto. `frame_index`
        é registrado no alvo para salvar o quadro-âncora anotado."""
        detections = self.detect_in_frame(frame_bgr)
        idx = self.detection_containing_point(detections, x, y)
        if idx is None:
            return None
        emb = self.compute_anchor_embedding(detections[idx], model)
        if emb is None:
            return None
        bbox = np.asarray(detections[idx]["bbox"]).astype(float).flatten()[:4].tolist()
        target = VideoTarget(
            label=label,
            anchor_embeddings=[np.asarray(emb)],
            anchor_policy="SINGLE",
            similarity_threshold=float(similarity_threshold),
            similarity_threshold_source="MANUAL",
            recognition_model=model or self._ff.default_model,
            anchor_frame_index=frame_index,
            anchor_bbox=bbox,
        )
        return target, idx, detections

    def _save_annotated_frame(self, frame_bgr, bbox, dst_path, *, label: str = "",
                              color=(0, 200, 83)) -> None:
        """Salva o quadro inteiro com a bbox marcada (cadeia de custódia/laudo)."""
        try:
            annotated = frame_bgr.copy()
            x1, y1, x2, y2 = (int(v) for v in np.asarray(bbox).astype(float).flatten()[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(annotated, label, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            imwrite_unicode_safe(dst_path, annotated)
        except Exception:
            pass

    def score_frame_against_target(self, frame_bgr, target, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Para a prévia de limiar (UI): similaridade + det_score + fiqa de
        cada detecção do quadro contra a âncora do alvo."""
        model = model or self._ff.default_model
        out: List[Dict[str, Any]] = []
        for d in self.detect_in_frame(frame_bgr):
            rgb = d.get("aligned_face")
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ff = self._ff.extract_embeddings_from_aligned_face(
                bgr, aligned_keypoints=d.get("aligned_keypoints"), model=model)
            emb = self._embedding_from_ffinfo(ff, model)
            sim = target.similarity_to(emb) if emb is not None else float("-inf")
            out.append({
                "det_score": float(d.get("det_score", 0.0)),
                "fiqa": (float(ff.fiqa_score) if ff and ff.fiqa_score is not None else None),
                "similarity": sim,
            })
        return out

    # ─────────────────────────── helpers internos ──────────────────────────

    def _embedding_from_ffinfo(self, ff_info, model: Optional[str]) -> Optional[np.ndarray]:
        if ff_info is None:
            return None
        name = model or self._ff.default_model
        slot = SepaelModel.ffinfo_slot_for(name)
        emb = getattr(ff_info, slot, None)
        if emb is not None:
            return np.asarray(emb)
        for member in SepaelModel:
            e = getattr(ff_info, member.ffinfo_slot, None)
            if e is not None:
                return np.asarray(e)
        return None

    @staticmethod
    def _derive_run_tag(video_sha: Optional[str], config_manifest: dict, fingerprints: List[Optional[str]]) -> str:
        h = hashlib.sha256()
        h.update((video_sha or "").encode("utf-8"))
        h.update(json.dumps(config_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for fp in fingerprints:
            h.update((fp or "").encode("utf-8"))
        return h.hexdigest()[:10]

    # ─────────────────────────────── extract ───────────────────────────────

    def extract(
        self,
        config: VideoExtractionConfig,
        targets: List[VideoTarget],
        *,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        now_utc: Optional[Callable[[], str]] = None,
        spatial_weight: float = 0.05,
        gap_tolerance: Optional[int] = None,
    ) -> VideoExtractionResult:
        if not targets:
            raise ValueError("extract requer ao menos um VideoTarget.")
        model = config.recognition_model or self._ff.default_model
        clock = now_utc or _utc_now_iso

        # Estado de track por alvo (mutável ao longo do laço). A lista pode
        # ter N; a v1 da UI expõe 1, mas o pipeline já é multi-alvo (N-ready).
        target_states = [
            TargetState(label=t.label, reference_embedding=t.reference_embedding(),
                        threshold=t.similarity_threshold)
            for t in targets
        ]
        target_by_label = {t.label: t for t in targets}
        target_index_by_label = {t.label: i for i, t in enumerate(targets)}

        info = self.probe_video(config.video_path)
        fps = info.fps_reported if info.fps_reported > 0 else 30.0
        every_n = config.every_n_frames or max(1, int(round(fps / max(1e-6, config.target_fps))))
        # Tolerância de gap do track em quadros NATIVOS: por padrão ~3 quadros
        # amostrados (every_n*3) — além disso, o re-id re-adquire (recovery).
        eff_gap_tolerance = gap_tolerance if gap_tolerance is not None else max(1, every_n * 3)
        start_frame = int(round(fps * max(0.0, config.start_s)))
        end_frame = int(round(fps * config.end_s)) if config.end_s is not None else None

        run_tag = config.run_tag or self._derive_run_tag(
            info.sha256, config.to_manifest_dict(), [t.anchor_embedding_fingerprint() for t in targets]
        )
        video_stem = Path(config.video_path).stem
        run_dir = Path(config.output_root) / f"{video_stem}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_tag

        n_sampled = n_det_total = n_accepted = n_rejected = n_no_det = 0
        accepted_faces: List[FaceProvenance] = []
        extraction_status = "completed"

        cap = cv2.VideoCapture(str(config.video_path))
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(f"Não foi possível abrir o vídeo: {config.video_path}")

        with VideoProvenanceWriter(run_dir, run_id) as writer:
            # embedding da âncora na auditoria
            for t in targets:
                ref = t.reference_embedding()
                writer.add_embedding(
                    kind="anchor", target_label=t.label, frame_index=None, det_index=None,
                    recognition_model=t.recognition_model, embedding=ref,
                    fingerprint=t.anchor_embedding_fingerprint(),
                )

            # Quadro-âncora anotado (cadeia de custódia / laudo).
            if config.save_annotated_anchor:
                for t in targets:
                    if t.anchor_frame_index is not None and t.anchor_bbox is not None:
                        af = self.read_frame(config.video_path, t.anchor_frame_index)
                        if af is not None:
                            self._save_annotated_frame(
                                af, t.anchor_bbox,
                                run_dir / "annotated" / f"anchor_{t.label}_f{int(t.anchor_frame_index):07d}.png",
                                label=t.label)

            try:
                if start_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                native = start_frame
                while True:
                    if cancel_check is not None and cancel_check():
                        extraction_status = "cancelled"
                        break
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if end_frame is not None and native > end_frame:
                        break
                    if (native - start_frame) % every_n != 0:
                        native += 1
                        continue

                    n_sampled += 1
                    ts_fps = native / fps if fps > 0 else None
                    ts_cap = (pos_msec / 1000.0) if pos_msec and pos_msec > 0 else None

                    status = self._process_frame(
                        frame=frame, native=native, ts_fps=ts_fps, ts_cap=ts_cap,
                        target_states=target_states, target_by_label=target_by_label,
                        target_index_by_label=target_index_by_label, model=model,
                        config=config, info=info, run_dir=run_dir, run_id=run_id,
                        writer=writer, clock=clock, accepted_faces=accepted_faces,
                        spatial_weight=spatial_weight, gap_tolerance=eff_gap_tolerance,
                    )
                    n_det_total += status["n_detections"]
                    n_accepted += status["n_accepted"]
                    n_rejected += status["n_rejected"]
                    if status["frame_status"] == "no_detection":
                        n_no_det += 1

                    if progress_callback:
                        denom = max(1, (end_frame if end_frame is not None
                                        else info.frame_count_reported) - start_frame)
                        pct = int(min(99, (native - start_frame) / denom * 100))
                        progress_callback(
                            f"Quadro {native} · amostrados {n_sampled} · "
                            f"aceitos {n_accepted} · sem face {n_no_det}", pct)

                    native += 1
            except Exception as e:
                extraction_status = "failed"
                print(f"❌ [VideoExtraction] erro no laço: {e}")
                raise
            finally:
                cap.release()

            manifest = {
                "created_at_utc": clock(),
                "source_video": info.to_manifest_dict(),
                "versions": _library_versions(),
                "parameters": config.to_manifest_dict(),
                "derived": {"every_n_frames": every_n, "start_frame": start_frame, "end_frame": end_frame, "fps_used": fps},
                "targets": [
                    {
                        "label": t.label,
                        "anchor_policy": t.anchor_policy,
                        "anchor_embedding_count": t.anchor_embedding_count,
                        "anchor_embedding_fingerprint": t.anchor_embedding_fingerprint(),
                        "similarity_threshold": t.similarity_threshold,
                        "similarity_threshold_source": t.similarity_threshold_source,
                        "recognition_model": t.recognition_model,
                    }
                    for t in targets
                ],
                "recognition_model": model,
                "extraction_status": extraction_status,
                "review_status": "not_started",
                "dataset_status": "not_imported",
                "summary": {
                    "n_frames_sampled": n_sampled,
                    "n_detections": n_det_total,
                    "n_accepted_auto": n_accepted,
                    "n_rejected_auto": n_rejected,
                    "n_no_detection_frames": n_no_det,
                },
            }
            writer.write_manifest(manifest)

            return VideoExtractionResult(
                run_dir=str(run_dir), run_id=run_id, extraction_status=extraction_status,
                manifest_path=str(writer.manifest_path), decisions_path=str(writer.decisions_path),
                frames_path=str(writer.frames_path), n_frames_sampled=n_sampled,
                n_detections=n_det_total, n_accepted_auto=n_accepted, n_rejected_auto=n_rejected,
                n_no_detection_frames=n_no_det, target_labels=[t.label for t in targets],
                faces=accepted_faces,
            )

    def _process_frame(
        self, *, frame, native, ts_fps, ts_cap, target_states, target_by_label,
        target_index_by_label, model, config, info, run_dir, run_id, writer, clock,
        accepted_faces, spatial_weight, gap_tolerance,
    ) -> Dict[str, Any]:
        detections = self.detect_in_frame(frame)
        n_det = len(detections)
        if n_det == 0:
            writer.write_frame({"frame_index": native, "timestamp_s_from_fps": ts_fps,
                                "n_detections": 0, "n_above_threshold": 0, "status": "no_detection"})
            return {"n_detections": 0, "n_accepted": 0, "n_rejected": 0, "frame_status": "no_detection"}

        # pré-filtro det HARD_REJECT (antes do embedding)
        survivors: List[Tuple[int, dict]] = []
        for i, d in enumerate(detections):
            ds = float(d.get("det_score", 0.0))
            if config.det_score_policy == QualityGatePolicy.HARD_REJECT and \
               config.min_det_score is not None and ds < config.min_det_score:
                writer.write_decision({"frame_index": native, "det_index": i, "det_score": ds,
                                       "decision": "hard_reject_det_pre_embedding", "reason": "low_det_score"})
                continue
            survivors.append((i, d))
        if not survivors:
            writer.write_frame({"frame_index": native, "timestamp_s_from_fps": ts_fps,
                                "n_detections": n_det, "n_above_threshold": 0, "status": "below_threshold"})
            return {"n_detections": n_det, "n_accepted": 0, "n_rejected": 1, "frame_status": "below_threshold"}

        # embeddings em batch
        bgr_aligned = np.stack([cv2.cvtColor(d["aligned_face"], cv2.COLOR_RGB2BGR) for _, d in survivors], axis=0)
        kps_list = [d.get("aligned_keypoints") for _, d in survivors]
        kps_batch = np.stack(kps_list, axis=0) if all(k is not None for k in kps_list) else None
        ff_infos = self._ff.extract_embeddings_from_aligned_faces_batch(
            bgr_aligned, models=[model], aligned_keypoints_batch=kps_batch
        )

        # candidatos + mapa det_index → (detecção, ff_info, embedding)
        candidates: List[Candidate] = []
        cand_map: Dict[int, Tuple[dict, Any, Any]] = {}
        for j, (i, d) in enumerate(survivors):
            ff_info = ff_infos[j] if j < len(ff_infos) else None
            emb = self._embedding_from_ffinfo(ff_info, model)
            bb = np.asarray(d["bbox"]).astype(float).flatten()[:4]
            candidates.append(Candidate(det_index=i, bbox=(bb[0], bb[1], bb[2], bb[3]),
                                        embedding=emb, det_score=float(d.get("det_score", 0.0))))
            cand_map[i] = (d, ff_info, emb)

        # log de decisão por detecção (melhor similaridade entre os alvos)
        n_above = 0
        for c in candidates:
            best_sim = float("-inf")
            best_label = None
            above = False
            for t in target_states:
                s = cosine(t.reference_embedding, c.embedding)
                if s > best_sim:
                    best_sim, best_label = s, t.label
                if s >= t.threshold:
                    above = True
            if above:
                n_above += 1
            d, ff_info, _ = cand_map[c.det_index]
            writer.write_decision({
                "frame_index": native, "det_index": c.det_index, "det_score": c.det_score,
                "fiqa_score": (float(ff_info.fiqa_score) if ff_info and ff_info.fiqa_score is not None else None),
                "similarity_to_anchor": (None if best_sim == float("-inf") else best_sim),
                "best_target": best_label, "above_threshold": above,
                "decision": ("candidate" if above else "non_match"),
            })

        # atribuição detecções×alvos (gate de identidade + gating espacial + greedy 1-1)
        assignments = assign_frame(target_states, candidates, native,
                                   spatial_weight=spatial_weight, gap_tolerance=gap_tolerance)
        if not assignments:
            writer.write_frame({"frame_index": native, "timestamp_s_from_fps": ts_fps,
                                "n_detections": n_det, "n_above_threshold": n_above, "status": "no_match"})
            return {"n_detections": n_det, "n_accepted": 0, "n_rejected": 0, "frame_status": "no_match"}

        n_acc = n_rej = 0
        any_conflict = False
        for a in assignments:
            d, ff_info, emb = cand_map[a.det_index]
            target = target_by_label[a.target_label]
            det_decision = _gate_decision(config.det_score_policy, float(d.get("det_score", 0.0)), config.min_det_score)
            fiqa_val = float(ff_info.fiqa_score) if ff_info and ff_info.fiqa_score is not None else None
            fiqa_decision = _gate_decision(config.fiqa_score_policy, fiqa_val, config.min_fiqa_score)
            if det_decision == "hard_reject" or fiqa_decision == "hard_reject":
                writer.write_decision({"frame_index": native, "det_index": a.det_index, "target": a.target_label,
                                       "decision": "hard_reject_quality",
                                       "det_score_decision": det_decision, "fiqa_score_decision": fiqa_decision})
                n_rej += 1
                continue
            auto_decision = "conflict" if a.margin_to_second_best is not None else "accepted"
            review_reasons: List[str] = []
            if auto_decision == "conflict":
                review_reasons.append("conflict")
                any_conflict = True
            if det_decision == "route_to_review":
                review_reasons.append("low_det")
            if fiqa_decision == "route_to_review":
                review_reasons.append("low_fiqa")
            prov = self._save_accepted_face(
                frame=frame, native=native, ts_fps=ts_fps, ts_cap=ts_cap, d=d, ff_info=ff_info, emb=emb,
                target=target, target_index=target_index_by_label.get(a.target_label, 0), assignment=a,
                model=model, config=config, info=info, run_dir=run_dir, run_id=run_id, writer=writer,
                clock=clock, auto_decision=auto_decision, review_reasons=review_reasons,
                det_decision=det_decision, fiqa_decision=fiqa_decision, fiqa_val=fiqa_val,
            )
            if prov is not None:
                accepted_faces.append(prov)
                n_acc += 1
            else:
                n_rej += 1

        if n_acc > 0:
            frame_status = "conflict" if any_conflict else "accepted"
        elif n_rej > 0:
            frame_status = "below_threshold"
        else:
            frame_status = "no_match"
        writer.write_frame({"frame_index": native, "timestamp_s_from_fps": ts_fps,
                            "n_detections": n_det, "n_above_threshold": n_above, "status": frame_status})
        return {"n_detections": n_det, "n_accepted": n_acc, "n_rejected": n_rej, "frame_status": frame_status}

    def _save_accepted_face(
        self, *, frame, native, ts_fps, ts_cap, d, ff_info, emb, target, target_index, assignment,
        model, config, info, run_dir, run_id, writer, clock, auto_decision, review_reasons,
        det_decision, fiqa_decision, fiqa_val,
    ) -> Optional[FaceProvenance]:
        bbox = np.asarray(d["bbox"]).astype(float).flatten()[:4]
        ext = extend_bbox(np.asarray(bbox), frame.shape, margin_factor=config.margin)
        x1, y1, x2, y2 = (int(v) for v in ext)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        label = target.label
        crop_name = f"{label}_f{native:07d}.png"
        crop_path = run_dir / label / crop_name
        if not imwrite_unicode_safe(crop_path, crop):
            print(f"⚠️ [VideoExtraction] falha ao gravar recorte {crop_path}")
            return None
        image_sha = self._hash.calculate_file_hash(crop_path)

        decoded_sha = decoded_algo = decoded_cs = None
        if config.save_decoded_frame_hash:
            decoded_sha = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
            decoded_algo, decoded_cs = "sha256", "BGR"

        aligned = self.align_detected_face_from_array(frame, d)
        h, w = crop.shape[:2]
        margin = assignment.margin_to_second_best

        # Salva a face ALINHADA 112×112 (já temos do detect_and_align). É a
        # imagem que o dataset usa para extrair embedding SEM re-detectar —
        # crucial para faces pequenas de vigilância, onde re-detectar o
        # recorte com margem falharia.
        aligned_file = aligned_sha = None
        if aligned.aligned_face_bgr is not None:
            aligned_rel = f"_aligned/{label}/{crop_name}"
            aligned_abs = run_dir / "_aligned" / label / crop_name
            if imwrite_unicode_safe(aligned_abs, aligned.aligned_face_bgr):
                aligned_file = aligned_rel
                aligned_sha = self._hash.calculate_file_hash(aligned_abs)

        prov = FaceProvenance(
            run_id=run_id, run_manifest=f"_run_{run_id}.json", identity_label=label,
            image_file=crop_name, image_sha256=image_sha, image_width_px=w, image_height_px=h,
            crop_margin_factor=config.margin, color_space="BGR", saved_at_utc=clock(), target_index=target_index,
            aligned_image_file=aligned_file, aligned_image_sha256=aligned_sha,
            source_video_filename=info.filename, source_video_sha256=info.sha256,
            frame_native_index=native, timestamp_s_from_fps=ts_fps, timestamp_s_capture_prop=ts_cap,
            decoded_frame_sha256=decoded_sha, decoded_frame_hash_algorithm=decoded_algo,
            decoded_frame_color_space=decoded_cs, decoder=info.decoder_backend, opencv_version=info.opencv_version,
            bbox_xyxy=bbox.tolist(), det_score=float(d.get("det_score", 0.0)),
            keypoints_5pt=aligned.keypoints_5pt, align_matrix_2x3=aligned.align_matrix_2x3,
            align_matrix_available=aligned.align_matrix_available, pose=aligned.pose,
            gender=aligned.gender, age=aligned.age,
            aligned_keypoints=(aligned.aligned_keypoints.tolist()
                               if aligned.aligned_keypoints is not None else None),
            reid_method="embedding+tracking", similarity_to_anchor=float(assignment.similarity),
            similarity_threshold=target.similarity_threshold, assigned_by=assignment.assigned_by,
            track_id=target_index, recognition_model=model,
            anchor_embedding_fingerprint=target.anchor_embedding_fingerprint(),
            margin_to_second_best=(float(margin) if margin is not None else None),
            min_det_score=config.min_det_score, det_score_policy=config.det_score_policy.to_str(),
            det_score_decision=det_decision, fiqa_score=fiqa_val, min_fiqa_score=config.min_fiqa_score,
            fiqa_score_policy=config.fiqa_score_policy.to_str(), fiqa_score_decision=fiqa_decision,
            auto_decision=auto_decision, review_reasons=review_reasons, human_review_status="pending",
        )
        writer.write_sidecar(prov)
        writer.add_embedding(
            kind=("conflict" if auto_decision == "conflict" else "accepted"),
            target_label=label, frame_index=native, det_index=assignment.det_index,
            recognition_model=model, embedding=emb,
        )
        # Conflito (≥2 candidatos acima do limiar) → cópia em _candidates/ p/
        # auditoria visual rápida do que ficou ambíguo.
        if auto_decision == "conflict":
            imwrite_unicode_safe(run_dir / "_candidates" / label / crop_name, crop)
        # Quadro inteiro anotado dos aceitos (opcional; default OFF — explode disco).
        if config.save_annotated_accepted:
            self._save_annotated_frame(
                frame, bbox, run_dir / "annotated" / f"{label}_f{native:07d}.png", label=label)
        return prov
