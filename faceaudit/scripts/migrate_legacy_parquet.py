"""Migra parquets legados de FaceDataset para o novo formato (PR 3).

Uso:
    python -m faceaudit.scripts.migrate_legacy_parquet <path1.parquet> [path2.parquet ...]
    python -m faceaudit.scripts.migrate_legacy_parquet --dir <path>          # todos os .parquet do dir
    python -m faceaudit.scripts.migrate_legacy_parquet --check <path>        # só inspeciona, não escreve

O que muda no formato:
  - Remove as 3 colunas de nível-dataset (`dataset_name`, `dataset_description`,
    `dataset_hash`) — vão para `schema.metadata`.
  - Remove o alias retrocompat `embedding` quando `embedding_v2` está presente
    (mesmo dado, redundante).
  - Adiciona à `schema.metadata` do parquet:
      * `faceaudit:dataset_type` = "face"
      * `faceaudit:dataset_name`, `faceaudit:dataset_description`
      * `faceaudit:column_specs` (JSON do catálogo built-in, filtrado pelas
        colunas presentes no parquet)
      * `faceaudit:column_coverage` (cobertura/min/max/mean/std por coluna)
      * `faceaudit:samples_per_identity`
      * `faceaudit:n_imgs`, `faceaudit:n_ids`, `faceaudit:last_modified`
      * `faceaudit:path_mode` = "absolute"

Operação é idempotente: rodar duas vezes no mesmo arquivo não faz mal — a
segunda execução simplesmente reescreve o mesmo formato com as stats
recomputadas.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq

from faceaudit.models.face_dataset import FaceDataset


def _is_new_format(path: Path) -> bool:
    """Retorna True se o parquet já tem `faceaudit:dataset_type` na metadata."""
    try:
        meta = pq.read_metadata(path).metadata or {}
    except Exception:
        return False
    return any(k.startswith(b"faceaudit:dataset_type") for k in meta.keys())


def _file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def _migrate_single(path: Path, *, dry_run: bool = False) -> Tuple[bool, str]:
    """Migra um parquet. Retorna (sucesso, mensagem)."""
    if not path.exists():
        return False, "Arquivo não encontrado"
    if not path.is_file():
        return False, "Não é um arquivo"
    if path.suffix != ".parquet":
        return False, f"Extensão inesperada ({path.suffix})"

    if _is_new_format(path):
        return True, "Já está no novo formato — pulado"

    size_before = _file_size_mb(path)

    if dry_run:
        return True, f"[CHECK] Migrável (formato legado, {size_before:.1f} MB)"

    # Carrega via FaceDataset.from_parquet (emite o warning legacy automaticamente —
    # silenciamos aqui pra não poluir o relatório).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = FaceDataset.from_parquet(path)
    except Exception as e:
        return False, f"Erro ao ler parquet: {e}"

    if not ds.faces:
        return False, "Dataset vazio — nada a migrar"

    # Salva no mesmo path (overwrite). to_parquet pós-PR 3 escreve no novo formato.
    try:
        ds.to_parquet(path)
    except Exception as e:
        return False, f"Erro ao gravar: {e}"

    size_after = _file_size_mb(path)
    delta = size_after - size_before
    delta_str = (f"{delta:+.1f} MB" if abs(delta) >= 0.05 else "≈ mesmo tamanho")
    return True, (
        f"Migrado: {len(ds.faces)} faces, {len(ds.column_specs)} column_specs, "
        f"{size_before:.1f} → {size_after:.1f} MB ({delta_str})"
    )


def _collect_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            raise SystemExit(f"--dir aponta para algo que não é diretório: {d}")
        paths.extend(sorted(d.glob("*.parquet")))
    paths.extend(Path(p) for p in args.paths)
    if not paths:
        raise SystemExit("Forneça pelo menos um path ou --dir.")
    return paths


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("paths", nargs="*", help="Arquivos .parquet a migrar")
    parser.add_argument("--dir", help="Diretório com arquivos .parquet a migrar (não-recursivo)")
    parser.add_argument("--check", action="store_true",
                        help="Apenas inspeciona, não modifica")
    args = parser.parse_args(argv)

    paths = _collect_paths(args)

    print(f"Modo: {'CHECK (dry-run)' if args.check else 'MIGRAR'}")
    print(f"Total de arquivos: {len(paths)}")
    print()

    n_ok = n_skip = n_fail = 0
    failures: List[Tuple[Path, str]] = []

    for path in paths:
        ok, msg = _migrate_single(path, dry_run=args.check)
        prefix = "✓" if ok else "✗"
        print(f"{prefix} {path.name}: {msg}")
        if ok:
            if "pulado" in msg:
                n_skip += 1
            else:
                n_ok += 1
        else:
            n_fail += 1
            failures.append((path, msg))

    print()
    print(f"Resumo: {n_ok} migrados, {n_skip} pulados (já novos), {n_fail} falhas")
    if failures:
        print()
        print("Falhas:")
        for p, m in failures:
            print(f"  - {p}: {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
