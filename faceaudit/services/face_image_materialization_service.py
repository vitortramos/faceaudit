"""Materializa imagens de face de uma pasta para outra, na forma pedida.

Um lugar só para "produzir a imagem que o dataset (ou o laudo) vai
guardar", nas três formas de `DatasetImageKind`. Antes, cada fluxo
resolvia isso por conta própria: a ingestão materializava crops alinhados
dentro do pipeline, a extração de vídeo recortava com margem usando
`extend_bbox`, e o `CropProcessor` reimplementava o recorte com outra
unidade de margem. Três caminhos, três comportamentos em caso de falha.

Regras que valem para todos os chamadores:

- **Imagem sem face detectada é PULADA**, nunca copiada como está. Uma
  pasta em que algumas imagens são recortes e outras são fotos inteiras
  parece uniforme e não é — quem abrir depois não tem como saber. As
  falhas vão para o manifesto e para o resultado, com a contagem.
- **Toda saída tem manifesto.** Uma pasta de recortes sem procedência não
  se sustenta em laudo: o `manifest.json` registra origem, forma, margem,
  e o SHA-256 de cada arquivo gerado.
- **A margem é fator sobre a bbox** (1.0 = bbox exata, 2.0 = o dobro), a
  mesma unidade da extração de vídeo, via `forensicface.geometry.extend_bbox`.

Sem Qt e sem I/O de UI — o progresso sai por callback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from faceaudit.enums.dataset_image_kind import DatasetImageKind
from faceaudit.models.image_file_extensions import EXTENSOES_DE_IMAGEM
from faceaudit.enums.path_mode import PathMode
from faceaudit.protocols.face_geometry_protocol import FaceGeometryProviderProtocol

_MANIFEST_NAME = "manifest.json"
# Lista única, compartilhada com a ingestão de pasta e com o seletor de
# arquivo — ver `faceaudit/models/image_file_extensions.py`.
_IMAGE_SUFFIXES = frozenset(EXTENSOES_DE_IMAGEM)


@dataclass
class MaterializationResult:
    """Saldo da operação, para a UI relatar sem recontar nada."""

    target_dir: Path
    kind: DatasetImageKind
    margin_factor: float
    n_processed: int = 0
    n_written: int = 0
    n_skipped: int = 0
    failures: List[Tuple[str, str]] = field(default_factory=list)
    manifest_path: Optional[Path] = None
    written_paths: List[Path] = field(default_factory=list)

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    def summary_pt(self) -> str:
        partes = [f"{self.n_written} imagem(ns) gerada(s)"]
        if self.n_skipped:
            partes.append(f"{self.n_skipped} já existia(m)")
        if self.n_failed:
            partes.append(f"{self.n_failed} sem face detectada")
        return " · ".join(partes)


def list_image_files(source_dir: Path, *, recursive: bool = True) -> List[Path]:
    """Arquivos de imagem sob `source_dir`, ordenados."""
    raiz = Path(source_dir)
    padrao = "**/*" if recursive else "*"
    return sorted(
        p for p in raiz.glob(padrao)
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


class FaceImageMaterializationService:
    """Stateless. A geometria facial entra por injeção (ver protocolo)."""

    def __init__(self, geometry: FaceGeometryProviderProtocol) -> None:
        self._geometry = geometry

    # ── API ──────────────────────────────────────────────────────────────

    def materialize_directory(
        self,
        source_dir: Path,
        target_dir: Path,
        *,
        kind: DatasetImageKind,
        margin_factor: float = 2.0,
        recursive: bool = True,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> MaterializationResult:
        """Gera em `target_dir` a forma `kind` de cada imagem de `source_dir`.

        Não cria dataset nenhum — só imagens e o manifesto. `ORIGINAL` não
        é aceito aqui: copiar arquivo é trabalho de cópia, não de
        materialização, e mascararia a ausência de detecção.
        """
        if not kind.requires_face_detection:
            raise ValueError(
                "Materializar exige recorte ou alinhamento; para a foto "
                "original não há o que produzir."
            )
        origem = Path(source_dir)
        if not origem.is_dir():
            raise FileNotFoundError(f"Pasta de origem não encontrada: {origem}")

        destino = Path(target_dir)
        destino.mkdir(parents=True, exist_ok=True)

        arquivos = list_image_files(origem, recursive=recursive)
        resultado = MaterializationResult(
            target_dir=destino, kind=kind, margin_factor=float(margin_factor),
        )
        entradas = []

        for indice, arquivo in enumerate(arquivos):
            if progress_callback:
                progress_callback(indice, len(arquivos), str(arquivo))
            resultado.n_processed += 1
            saida = self._destination_for(arquivo, origem, destino)

            if skip_existing and saida.exists():
                resultado.n_skipped += 1
                continue

            try:
                imagem = self._read_image(arquivo)
            except Exception as exc:
                resultado.failures.append((str(arquivo), f"leitura: {exc}"))
                continue
            if imagem is None:
                resultado.failures.append((str(arquivo), "arquivo ilegível"))
                continue

            try:
                produzida = self._materialize_one(imagem, kind, float(margin_factor))
            except Exception as exc:
                resultado.failures.append((str(arquivo), f"processamento: {exc}"))
                continue
            if produzida is None:
                resultado.failures.append((str(arquivo), "nenhuma face detectada"))
                continue

            try:
                saida.parent.mkdir(parents=True, exist_ok=True)
                self._write_image(saida, produzida)
            except Exception as exc:
                resultado.failures.append((str(arquivo), f"gravação: {exc}"))
                continue

            resultado.n_written += 1
            resultado.written_paths.append(saida)
            entradas.append({
                "source": str(arquivo),
                "output": saida.name,
                "sha256": self._sha256(saida),
            })

        if progress_callback:
            progress_callback(len(arquivos), len(arquivos), "")

        resultado.manifest_path = self._write_manifest(
            destino, origem, resultado, entradas,
        )
        return resultado

    # ── Internos ─────────────────────────────────────────────────────────

    def _materialize_one(
        self, image: np.ndarray, kind: DatasetImageKind, margin_factor: float,
    ) -> Optional[np.ndarray]:
        if kind is DatasetImageKind.ALIGNED:
            return self._geometry.align_face(image)
        return self._crop_with_margin(image, margin_factor)

    def _crop_with_margin(
        self, image: np.ndarray, margin_factor: float,
    ) -> Optional[np.ndarray]:
        """Recorte da maior face, expandido pelo fator. Sem rotação."""
        from forensicface.geometry import extend_bbox

        faces = self._geometry.detect_faces(image) or []
        melhor = self._largest_face(faces)
        if melhor is None:
            return None

        x1, y1, x2, y2 = extend_bbox(
            np.asarray(melhor, dtype="float32"), image.shape,
            margin_factor=float(margin_factor),
        )
        altura, largura = image.shape[:2]
        x1 = max(0, min(largura - 1, int(x1)))
        y1 = max(0, min(altura - 1, int(y1)))
        x2 = max(x1 + 1, min(largura, int(x2)))
        y2 = max(y1 + 1, min(altura, int(y2)))
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _largest_face(faces: Sequence) -> Optional[np.ndarray]:
        """Bbox da maior face — critério estável quando há mais de uma."""
        melhor = None
        melhor_area = 0.0
        for face in faces:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > melhor_area:
                melhor_area = area
                melhor = np.asarray([x1, y1, x2, y2])
        return melhor

    @staticmethod
    def _destination_for(arquivo: Path, origem: Path, destino: Path) -> Path:
        """Espelha a estrutura da origem e normaliza a extensão para PNG.

        PNG porque recorte e alinhamento são material de exame: recomprimir
        em JPEG introduz artefato sobre artefato.
        """
        try:
            relativo = arquivo.relative_to(origem)
        except ValueError:
            relativo = Path(arquivo.name)
        return destino / relativo.with_suffix(".png")

    @staticmethod
    def _read_image(caminho: Path) -> Optional[np.ndarray]:
        from faceaudit.services.image_io import imread_unicode_safe

        return imread_unicode_safe(str(caminho))

    @staticmethod
    def _write_image(caminho: Path, imagem: np.ndarray) -> None:
        from faceaudit.services.image_io import imwrite_unicode_safe

        if not imwrite_unicode_safe(str(caminho), imagem):
            raise OSError(f"falha ao gravar {caminho}")

    @staticmethod
    def _sha256(caminho: Path) -> Optional[str]:
        from faceaudit.services.hash_service import HashService

        try:
            return HashService().calculate_file_hash(caminho)
        except Exception:
            return None

    def _write_manifest(
        self, destino: Path, origem: Path,
        resultado: MaterializationResult, entradas: list,
    ) -> Path:
        manifesto = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(origem),
            "image_kind": resultado.kind.value,
            "margin_factor": (
                resultado.margin_factor
                if resultado.kind is DatasetImageKind.CROP else None
            ),
            "totals": {
                "processed": resultado.n_processed,
                "written": resultado.n_written,
                "skipped_existing": resultado.n_skipped,
                "failed": resultado.n_failed,
            },
            # Imagens sem face ficam de FORA da pasta; registrar aqui é o
            # que impede alguém de concluir, mais tarde, que a origem só
            # tinha as que sobraram.
            "failures": [
                {"source": caminho, "reason": motivo}
                for caminho, motivo in resultado.failures
            ],
            "images": entradas,
        }
        caminho = destino / _MANIFEST_NAME
        caminho.write_text(
            json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        return caminho

    # ── Segunda porta de entrada: a partir de um dataset ──────────────────

    def materialize_dataset(
        self,
        dataset,
        target_dir: Path,
        *,
        kind: DatasetImageKind,
        margin_factor: float = 2.0,
        images_subdir: str = "images",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """Produz a forma `kind` de cada face e devolve um dataset novo.

        Irmã de `materialize_directory`, para quando a entrada já é um
        `FaceDataset` — as faces podem apontar para pastas diferentes, então
        varrer um diretório não serviria. O dataset devolvido é autocontido:
        imagens em `<target_dir>/<images_subdir>/` e caminhos RELATIVOS
        ancorados ali, prontos para o parquet ser gravado em `target_dir`.

        A entrada não é mutada. Faces cuja imagem não pôde ser produzida
        ficam DE FORA do dataset devolvido e são registradas em `failures` —
        mesma regra do caminho de pasta, e o caller precisa mostrá-las.

        `ALIGNED` marca `is_aligned=True` no dataset resultante; `CROP` NÃO
        — um recorte não é face alinhada, e declarar que é faria o
        `FFService` pular detect+align depois.
        """
        import copy as _copy

        if not kind.requires_face_detection:
            raise ValueError(
                "Materializar exige recorte ou alinhamento; para a foto "
                "original use a cópia simples de imagens."
            )
        destino = Path(target_dir)
        pasta_imagens = destino / images_subdir
        pasta_imagens.mkdir(parents=True, exist_ok=True)

        resultado = MaterializationResult(
            target_dir=destino, kind=kind, margin_factor=float(margin_factor),
        )
        entradas = []
        faces_novas = []
        usados: set = set()
        faces = list(getattr(dataset, "faces", []) or [])

        for indice, face in enumerate(faces):
            if progress_callback:
                progress_callback(indice, len(faces), getattr(face, "image_path", "") or "")
            resultado.n_processed += 1

            try:
                origem = Path(dataset.resolve_image_path(face))
                imagem = self._read_image(origem)
            except Exception as exc:
                resultado.failures.append((str(getattr(face, "image_path", "")), f"leitura: {exc}"))
                continue
            if imagem is None:
                resultado.failures.append((str(origem), "arquivo ilegível"))
                continue

            try:
                produzida = self._materialize_one(imagem, kind, float(margin_factor))
            except Exception as exc:
                resultado.failures.append((str(origem), f"processamento: {exc}"))
                continue
            if produzida is None:
                resultado.failures.append((str(origem), "nenhuma face detectada"))
                continue

            nome = self._unique_png_name(origem, usados)
            saida = pasta_imagens / nome
            try:
                self._write_image(saida, produzida)
            except Exception as exc:
                resultado.failures.append((str(origem), f"gravação: {exc}"))
                continue

            nova = _copy.copy(face)
            nova.image_path = nome          # relativo à pasta de imagens
            nova.file_name = nome
            nova.hash = self._sha256(saida)
            if getattr(nova, "ff_info", None) is not None:
                nova.ff_info.aligned_face = None
            faces_novas.append(nova)

            resultado.n_written += 1
            resultado.written_paths.append(saida)
            entradas.append({
                "source": str(origem), "output": nome, "sha256": nova.hash,
            })

        if progress_callback:
            progress_callback(len(faces), len(faces), "")

        resultado.manifest_path = self._write_manifest(
            destino, destino, resultado, entradas,
        )

        novo = _copy.copy(dataset)
        novo.faces = faces_novas
        novo.path_mode = PathMode.RELATIVE
        novo.path_root = images_subdir
        novo.is_aligned = kind.implies_aligned_dataset
        novo.hash = None
        novo._parquet_dir = destino.resolve()
        return novo, resultado

    @staticmethod
    def _unique_png_name(origem: Path, usados: set) -> str:
        """Nome PNG único no destino, derivado do arquivo de origem."""
        stem = origem.stem or "face"
        nome = f"{stem}.png"
        i = 1
        while nome in usados:
            nome = f"{stem}_{i}.png"
            i += 1
        usados.add(nome)
        return nome
