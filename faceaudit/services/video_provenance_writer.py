# faceaudit/services/video_provenance_writer.py
"""Escrita dos artefatos de proveniência de um run de extração de vídeo.

Camadas (todas versionadas):
  _meta/_run_<id>.json        manifesto do run (cabeçalho de proveniência)
  _meta/_decisions_<id>.jsonl uma linha por DETECÇÃO (inclui rejeitadas)
  _meta/_frames_<id>.jsonl    uma linha por QUADRO amostrado (inclui sem detecção)
  _meta/<crop>.json           sidecar por face salva (autodescritivo)
  _embeddings_<id>.parquet    embeddings de âncora + candidatos (auditoria do re-id)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from faceaudit.models.video.face_provenance import FaceProvenance


RUN_SCHEMA = "faceaudit.video_extraction.run_manifest"
# v2: campos aditivos de reprodutibilidade (reference_images, timestamps de
# início/fim/duração, versão+commit do app, proveniência completa das âncoras)
# + cópia legível na raiz. Retrocompatível: leitores usam .get().
RUN_SCHEMA_VERSION = 2

# Nome ASCII (sem diacríticos) por robustez de FS/sync/zip cross-plataforma.
ROOT_MANIFEST_NAME = "manifesto_extracao.json"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo não serializável em JSON: {type(o)!r}")


def mirror_manifest_to_root(run_dir) -> None:
    """Reflete o manifesto autoritativo (_meta/_run_*.json) na cópia legível da
    raiz (<run_dir>/manifesto_extracao.json). DEVE ser chamado após QUALQUER
    mutação do manifesto (fim da revisão, importação/enriquecimento) — senão a
    cópia da raiz congela no estado de extração e engana o perito (mostraria
    review_status=not_started / sem final_dataset). Best-effort."""
    try:
        run_dir = Path(run_dir)
        metas = sorted((run_dir / "_meta").glob("_run_*.json"))
        if not metas:
            return
        (run_dir / ROOT_MANIFEST_NAME).write_text(
            metas[0].read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ [video_provenance] falha ao sincronizar cópia do manifesto na raiz: {e}")


class VideoProvenanceWriter:
    """Escreve incrementalmente os artefatos de um run. Use como context
    manager para garantir o fechamento dos arquivos JSONL."""

    def __init__(self, run_dir, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.meta_dir = self.run_dir / "_meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.meta_dir / f"_run_{run_id}.json"
        # Cópia legível do manifesto na RAIZ da pasta (nome óbvio p/ o perito
        # achar sem entrar em _meta/). O _meta/_run_<id>.json continua sendo o
        # autoritativo lido pela importação; a cópia é inerte para ela.
        self.root_manifest_path = self.run_dir / ROOT_MANIFEST_NAME
        self.decisions_path = self.meta_dir / f"_decisions_{run_id}.jsonl"
        self.frames_path = self.meta_dir / f"_frames_{run_id}.jsonl"
        self.embeddings_path = self.run_dir / f"_embeddings_{run_id}.parquet"

        self._decisions_fh = None
        self._frames_fh = None
        self._embedding_rows: List[Dict[str, Any]] = []

    # ---- ciclo de vida ----
    def __enter__(self) -> "VideoProvenanceWriter":
        self._decisions_fh = open(self.decisions_path, "w", encoding="utf-8")
        self._frames_fh = open(self.frames_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for fh in (self._decisions_fh, self._frames_fh):
            try:
                if fh is not None:
                    fh.flush()
                    fh.close()
            except Exception:
                pass
        self._decisions_fh = None
        self._frames_fh = None
        self._flush_embeddings()

    # ---- linhas JSONL ----
    def write_decision(self, record: Dict[str, Any]) -> None:
        if self._decisions_fh is None:
            return
        self._decisions_fh.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )

    def write_frame(self, record: Dict[str, Any]) -> None:
        if self._frames_fh is None:
            return
        self._frames_fh.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )

    # ---- sidecar por face ----
    def write_sidecar(self, provenance: FaceProvenance) -> Path:
        path = self.meta_dir / provenance.sidecar_filename()
        path.write_text(
            json.dumps(
                provenance.to_sidecar_dict(),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        return path

    # ---- embeddings (auditoria do re-id) ----
    def add_embedding(
        self,
        *,
        kind: str,                 # "anchor" | "candidate" | "accepted"
        target_label: Optional[str],
        frame_index: Optional[int],
        det_index: Optional[int],
        recognition_model: Optional[str],
        embedding: Optional[np.ndarray],
        fingerprint: Optional[str] = None,
    ) -> None:
        if embedding is None:
            return
        arr = np.ascontiguousarray(embedding, dtype=np.float32)
        self._embedding_rows.append({
            "kind": kind,
            "target_label": target_label,
            "frame_index": frame_index,
            "det_index": det_index,
            "recognition_model": recognition_model,
            "fingerprint": fingerprint,
            "dim": int(arr.shape[0]) if arr.ndim == 1 else int(arr.size),
            "dtype": "float32",
            "normalization": "raw",
            "embedding": arr.tolist(),
        })

    def _flush_embeddings(self) -> None:
        if not self._embedding_rows:
            return
        try:
            import pandas as pd
            df = pd.DataFrame(self._embedding_rows)
            df.to_parquet(self.embeddings_path, index=False)
        except Exception as e:
            print(f"⚠️ [VideoProvenanceWriter] falha ao gravar embeddings parquet: {e}")

    # ---- manifesto ----
    def write_manifest(self, manifest: Dict[str, Any]) -> Path:
        payload = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            **manifest,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        self.manifest_path.write_text(text, encoding="utf-8")
        # Cópia idêntica na raiz (visibilidade/reprodutibilidade). Best-effort:
        # nunca deixa a extração falhar só por causa da cópia.
        try:
            self.root_manifest_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ [VideoProvenanceWriter] falha ao gravar cópia do manifesto na raiz: {e}")
        return self.manifest_path
