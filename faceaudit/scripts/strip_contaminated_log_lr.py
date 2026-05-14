"""One-shot script para limpar parquets de SIMILARITY_SCORES contaminados
com a coluna `log_lr`.

Bug pré-existente: o ``ScoreCalibrationInstructionExecutor`` mutava
``sample.metadata["log_lr"]`` in-place nos items do dataset de input
(porque o output era criado com as mesmas referências). Quando o Run
All chegava em ``_persist_trained_model_resources_inline``, ele iterava
``trained_resources`` (TODOS os recursos do contexto, inclusive o input
recém-mutado) e regravava qualquer um sem ``artifact_ref`` ainda — o
parquet do input ia parar no disco com a coluna ``log_lr`` populada.

Efeito: a validação ``SCORE_CALIBRATION_STATE(RAW)`` falhava na próxima
re-execução, porque o loader detectava ``log_lr`` em ``metadata`` e
marcava o dataset como CALIBRATED.

O bug está corrigido a partir do commit que introduz cópia de samples
em vez de mutação. Este script é a recuperação dos projetos já
contaminados: remove a coluna ``log_lr`` dos parquets cujo
``artifact_type`` é ``similarity_scores`` e atualiza o hash SHA-256 no
``artifact_ref`` da definição da estratégia correspondente.

Uso::

    python -m faceaudit.scripts.strip_contaminated_log_lr --project <path>
    python -m faceaudit.scripts.strip_contaminated_log_lr --project <path> --apply

Dry-run por padrão; ``--apply`` executa de fato. Idempotente — rodar
duas vezes não faz nada na segunda.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class CleanupAction:
    definition_path: Path
    resource_id: str
    parquet_path: Path
    old_hash: Optional[str]
    new_hash: str
    rows: int


def _sha256_file(path: Path) -> str:
    if hasattr(hashlib, "file_digest"):
        with path.open("rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_log_lr_from_parquet(
    parquet_path: Path,
    *,
    apply: bool,
) -> tuple[bool, int]:
    """Retorna (modified, n_rows). Quando ``apply=False``, apenas
    inspeciona e retorna se ALGUMA modificação seria feita."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    if "log_lr" not in table.column_names:
        return False, table.num_rows

    if apply:
        new_table = table.drop(["log_lr"])
        # Sobrescreve preservando o schema (sem `log_lr`). Não usamos
        # ``SimilarityScoreDataset.to_parquet`` aqui pra evitar re-iterar
        # 5M samples Python — pyarrow grava direto em C.
        pq.write_table(new_table, parquet_path)
    return True, table.num_rows


def _find_contaminated_resources(
    project_root: Path,
) -> List[tuple[Path, dict, dict]]:
    """Retorna [(definition_path, definition_json, resource_json)] para
    cada recurso SIMILARITY_SCORES com parquet existente."""
    out: List[tuple[Path, dict, dict]] = []
    definitions_dir = project_root / "definitions"
    if not definitions_dir.is_dir():
        return out
    for defn_path in sorted(definitions_dir.glob("*_definition.json")):
        try:
            defn = json.loads(defn_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for resource in defn.get("resources", []):
            if resource.get("resource_type") != "SIMILARITY_SCORES":
                continue
            ref = resource.get("artifact_ref") or {}
            rel_path = ref.get("path")
            if not rel_path:
                continue
            # Normaliza Windows backslashes (caso projetos antigos tenham).
            rel_path_norm = rel_path.replace("\\", "/")
            abs_path = project_root / rel_path_norm
            if not abs_path.is_file():
                continue
            if not abs_path.suffix == ".parquet":
                continue
            out.append((defn_path, defn, resource))
    return out


def run(project_root: Path, *, apply: bool) -> int:
    if not project_root.is_dir():
        print(f"Project root not found: {project_root}", file=sys.stderr)
        return 2

    candidates = _find_contaminated_resources(project_root)
    actions: List[CleanupAction] = []
    untouched = 0

    # Cache de definitions agrupadas para sobrescrever uma vez por arquivo.
    by_defn: dict[Path, dict] = {}

    for defn_path, defn, resource in candidates:
        parquet_path = project_root / resource["artifact_ref"]["path"].replace(
            "\\", "/"
        )

        # First inspect (cheap-ish — pyarrow read_table is fast in C).
        import pyarrow.parquet as pq
        table_schema = pq.read_schema(parquet_path)
        has_log_lr = "log_lr" in table_schema.names
        if not has_log_lr:
            untouched += 1
            continue

        # Read to count rows for reporting; needed anyway if --apply.
        old_hash = resource["artifact_ref"].get("hash")
        modified, n_rows = _strip_log_lr_from_parquet(
            parquet_path, apply=apply,
        )
        if not modified:
            untouched += 1
            continue

        # Hash after rewrite — only meaningful when apply=True.
        new_hash = _sha256_file(parquet_path) if apply else "(dry-run)"

        action = CleanupAction(
            definition_path=defn_path,
            resource_id=resource["resource_id"],
            parquet_path=parquet_path,
            old_hash=old_hash,
            new_hash=new_hash,
            rows=n_rows,
        )
        actions.append(action)

        if apply:
            # Update hash in the in-memory definition; write below.
            resource["artifact_ref"]["hash"] = new_hash
            by_defn[defn_path] = defn

    if apply and by_defn:
        for defn_path, defn in by_defn.items():
            defn_path.write_text(
                json.dumps(defn, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    print(f"Project: {project_root}")
    print(f"Resources scanned: {len(candidates)}")
    print(f"Already clean: {untouched}")
    print(f"Contaminated: {len(actions)}")
    print()
    for a in actions:
        print(f"  {a.parquet_path.name}  ({a.rows:,} rows)".replace(",", "."))
        print(f"      definition: {a.definition_path.name}, resource_id={a.resource_id!r}")
        print(f"      hash: {a.old_hash} -> {a.new_hash}")
    if not actions:
        print("(nothing to do)")
    elif not apply:
        print()
        print("Dry-run only. Re-run with --apply to perform the cleanup.")
    else:
        print()
        print("Cleanup complete.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="faceaudit.scripts.strip_contaminated_log_lr",
        description=(
            "Remove a coluna `log_lr` de parquets SIMILARITY_SCORES "
            "contaminados pelo bug de mutação in-place do "
            "ScoreCalibrationInstructionExecutor."
        ),
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Caminho da pasta do projeto FaceAudit (.fa root).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Executar de fato. Sem este flag, apenas reporta o que seria feito.",
    )
    args = parser.parse_args(argv)

    return run(Path(args.project).expanduser(), apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
