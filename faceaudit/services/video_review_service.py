# faceaudit/services/video_review_service.py
"""Revisão/cull humana de um run de extração de vídeo (backend).

A revisão é obrigatória antes do dataset final (invariante END_TO_END).
Aqui ficam as operações sobre o run produzido:
  - `list_faces`   — popula a grade de revisão (Fase 5).
  - `set_face_decision` — flipa `human_review.status` no sidecar e move o
    recorte entre `<label>/` e `_rejected/` (rejeitados NÃO são apagados —
    cadeia de custódia preserva o que foi descartado e por quê).
  - `commit_review` — grava `review_status` no manifesto; `skipped` é
    override explícito e auditado (não caminho normal).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_NON_FACE_PREFIXES = ("_run_", "_decisions_", "_frames_")
VALID_STATUSES = ("pending", "accepted", "rejected")


@dataclass
class ReviewFace:
    """Uma face materializada, do ponto de vista da revisão (para a grade)."""
    sidecar_path: str
    image_file: str
    label: str
    frame_index: Optional[int]
    similarity_to_anchor: Optional[float]
    det_score: Optional[float]
    fiqa_score: Optional[float]
    auto_decision: str
    review_reasons: List[str] = field(default_factory=list)
    human_review_status: str = "pending"
    crop_path: Optional[str] = None      # localização atual no disco (ou None se ausente)


class VideoReviewService:
    def list_faces(self, run_dir, *, statuses: Optional[List[str]] = None) -> List[ReviewFace]:
        run_dir = Path(run_dir)
        meta_dir = run_dir / "_meta"
        out: List[ReviewFace] = []
        for sidecar in sorted(meta_dir.glob("*.json")):
            if any(sidecar.name.startswith(p) for p in _NON_FACE_PREFIXES):
                continue
            try:
                d = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                continue
            rf = self._to_review_face(run_dir, sidecar, d)
            if statuses is None or rf.human_review_status in statuses:
                out.append(rf)
        return out

    def set_face_decision(
        self,
        run_dir,
        image_file: str,
        *,
        status: str,
        operator: Optional[str] = None,
        reason: Optional[str] = None,
        now_utc: Optional[Callable[[], str]] = None,
    ) -> ReviewFace:
        if status not in VALID_STATUSES:
            raise ValueError(f"status inválido: {status!r} (use {VALID_STATUSES})")
        run_dir = Path(run_dir)
        meta_dir = run_dir / "_meta"
        sidecar = meta_dir / (Path(image_file).stem + ".json")
        if not sidecar.exists():
            raise FileNotFoundError(f"sidecar não encontrado para {image_file}")
        d = json.loads(sidecar.read_text(encoding="utf-8"))

        hr = d.get("human_review") or {}
        hr["status"] = status
        hr["operator"] = operator
        hr["timestamp_utc"] = (now_utc or _utc_now_iso)()
        hr["reason"] = reason
        d["human_review"] = hr
        sidecar.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

        label = (d.get("identity") or {}).get("label")
        self._relocate_crop(run_dir, label, image_file, status)
        return self._to_review_face(run_dir, sidecar, d)

    def commit_review(
        self,
        run_dir,
        *,
        operator: Optional[str] = None,
        skipped: bool = False,
        now_utc: Optional[Callable[[], str]] = None,
    ) -> Dict[str, Any]:
        """Fecha a revisão no manifesto. `skipped=True` é override explícito
        (dispensa de revisão humana) — registrado para auditoria."""
        run_dir = Path(run_dir)
        meta_dir = run_dir / "_meta"
        manifests = sorted(meta_dir.glob("_run_*.json"))
        if not manifests:
            raise FileNotFoundError("manifesto _run_*.json não encontrado")
        mpath = manifests[0]
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

        summary = self.review_summary(run_dir)
        manifest["review_status"] = "skipped" if skipped else "completed"
        manifest["review"] = {
            "operator": operator,
            "timestamp_utc": (now_utc or _utc_now_iso)(),
            "skipped": bool(skipped),
            "n_accepted": summary["accepted"],
            "n_rejected": summary["rejected"],
            "n_pending": summary["pending"],
        }
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest["review"]

    def review_summary(self, run_dir) -> Dict[str, int]:
        counts = {"pending": 0, "accepted": 0, "rejected": 0}
        for rf in self.list_faces(run_dir):
            counts[rf.human_review_status] = counts.get(rf.human_review_status, 0) + 1
        return counts

    # ---- helpers ----
    @staticmethod
    def _to_review_face(run_dir: Path, sidecar: Path, d: Dict[str, Any]) -> ReviewFace:
        ident = d.get("identity") or {}
        reid = d.get("reidentification") or {}
        qg = d.get("quality_gates") or {}
        frame = d.get("frame") or {}
        hr = d.get("human_review") or {}
        artifact = d.get("artifact") or {}
        label = ident.get("label")
        image_file = artifact.get("image_file")
        crop = VideoReviewService._locate_crop(run_dir, label, image_file)
        return ReviewFace(
            sidecar_path=str(sidecar), image_file=image_file, label=label,
            frame_index=frame.get("native_index"),
            similarity_to_anchor=reid.get("similarity_to_anchor"),
            det_score=qg.get("det_score"), fiqa_score=qg.get("fiqa_score"),
            auto_decision=d.get("auto_decision", "accepted"),
            review_reasons=list(d.get("review_reasons") or []),
            human_review_status=hr.get("status", "pending"),
            crop_path=(str(crop) if crop else None),
        )

    @staticmethod
    def _locate_crop(run_dir: Path, label: Optional[str], image_file: Optional[str]) -> Optional[Path]:
        if not image_file:
            return None
        in_label = run_dir / (label or "") / image_file
        if in_label.exists():
            return in_label
        in_rejected = run_dir / "_rejected" / image_file
        if in_rejected.exists():
            return in_rejected
        return None

    @staticmethod
    def _relocate_crop(run_dir: Path, label: Optional[str], image_file: str, status: str) -> None:
        """Garante o recorte em `<label>/` (accepted/pending) ou em
        `_rejected/` (rejected). Idempotente."""
        in_label = run_dir / (label or "") / image_file
        in_rejected = run_dir / "_rejected" / image_file
        if status == "rejected":
            src, dst = in_label, in_rejected
        else:
            src, dst = in_rejected, in_label
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
