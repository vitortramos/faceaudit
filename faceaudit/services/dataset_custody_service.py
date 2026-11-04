"""Entradas da cadeia de custódia para operações do Gerenciador de Datasets.

O `AuditLogService` sabe gravar e verificar a cadeia; este módulo sabe O QUE
cada operação tem de registrar — Juntar (MERGE), Dividir (SPLIT), exclusão de
registros (REMOVE_FACES) e edição de identidade (EDIT_LABEL). A tela só coleta
a decisão do perito (e a justificativa) e chama.

Duas ideias atravessam o módulo:

**Dataset derivado registra a origem pelo SHA-256 do parquet.** É o mesmo
padrão do Processar do pipeline de imagens: quem abrir o derivado precisa
saber, pela cadeia DELE, de quais arquivos ele veio. Junto vai o
`entry_hash` da última entrada da cadeia de cada origem (`audit_chain_head`),
que amarra criptograficamente o derivado à história da origem sem copiá-la
— copiar a cadeia de UMA das duas origens de um Juntar seria contar só
metade da história.

**Entrada de dataset que ainda não está em disco não se perde.** O Juntar
produz um dataset em memória, sem arquivo e portanto sem sidecar. A entrada
fica PENDENTE no próprio objeto (`queue_pending`) e é escrita na cadeia do
arquivo quando ele é gravado (`write_pending`), antes da entrada de
gravação, que traz o hash do arquivo e fecha o elo. `recorded_at` guarda o
instante do ato; o `timestamp` da entrada é o da escrita.
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.services.audit_log_service import (
    AuditLogService,
    require_justification,
    validate_entry_request,
)

# Mesmo teto das outras listas de identificadores na cadeia: listar 5 mil
# linhas não serve a ninguém; o hash do conjunto prova pertencimento.
MAX_AUDIT_IDENTIFIERS = 50

_PENDING_ATTR = "_pending_audit_entries"


# ── Identificação de faces e conjuntos ─────────────────────────────────────


def face_identifier(face: Any) -> str:
    """`hash` da face; na falta, o `image_path`."""
    return (
        getattr(face, "hash", None)
        or getattr(face, "image_path", None)
        or "(sem identificação)"
    )


def set_sha256(values: Iterable[Any]) -> str:
    """SHA-256 do conjunto (ordenado, uma linha por valor)."""
    return hashlib.sha256(
        "\n".join(sorted(str(v) for v in values)).encode("utf-8")
    ).hexdigest()


def summarize_identifiers(identifiers: Sequence[str], prefix: str) -> Dict[str, Any]:
    """Os primeiros identificadores + se a lista foi cortada + hash de todos."""
    ids = list(identifiers)
    return {
        f"{prefix}_identifiers": ids[:MAX_AUDIT_IDENTIFIERS],
        f"{prefix}_identifiers_truncated": len(ids) > MAX_AUDIT_IDENTIFIERS,
        f"{prefix}_set_sha256": set_sha256(ids),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ── Exclusão de registros e edição de identidade ───────────────────────────


def removal_entry(
    faces: Sequence[Any],
    justification: Optional[str],
    *,
    origin: str = "table_or_gallery",
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """(descrição, metadata) de REMOVE_FACES. Recusa justificativa vazia."""
    motivo = require_justification(justification)
    meta: Dict[str, Any] = {
        "removed_count": len(faces),
        "justification": motivo,
        **summarize_identifiers([face_identifier(f) for f in faces], "removed"),
        "pending_until_save": True,
        "origin": origin,
    }
    meta.update(extra or {})
    descricao = (
        f"{len(faces)} registro(s) excluído(s) do dataset. "
        f"Justificativa: {motivo}"
    )
    return descricao, meta


def identity_edit_entry(
    *,
    action: str,
    old_identities: Iterable[str],
    new_identity: str,
    changed_identifiers: Sequence[str],
    justification: Optional[str],
    origin: str = "audit_dialog",
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """(descrição, metadata) de EDIT_LABEL para renomear/mesclar identidades.

    `action` é `"rename_identity"` ou `"merge_identities"`.
    """
    if action not in ("rename_identity", "merge_identities"):
        raise ValueError(f"ação de identidade desconhecida: {action!r}")
    motivo = require_justification(justification)
    antigas = sorted(str(i) for i in old_identities)
    meta: Dict[str, Any] = {
        "action": action,
        "field": "identity",
        "old_identities": antigas,
        "new_identity": str(new_identity),
        "changed_count": len(changed_identifiers),
        "justification": motivo,
        **summarize_identifiers(changed_identifiers, "changed"),
        "pending_until_save": True,
        "origin": origin,
    }
    meta.update(extra or {})
    verbo = "renomeada(s)" if action == "rename_identity" else "mesclada(s)"
    descricao = (
        f"Identidade(s) {', '.join(repr(a) for a in antigas)} {verbo} para "
        f"'{new_identity}' ({len(changed_identifiers)} registro(s)). "
        f"Justificativa: {motivo}"
    )
    return descricao, meta


# ── Origem de dataset derivado ─────────────────────────────────────────────


def describe_source(
    dataset: Any,
    parquet_path: Optional[str | Path],
    *,
    has_unsaved_changes: bool,
    hash_service: Any = None,
    content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Quem é a origem: nome, arquivo, SHA-256, contagens e topo da cadeia.

    Sem arquivo (dataset só em memória), `parquet_sha256` fica None e
    `has_unsaved_changes` True — o derivado não pode afirmar um arquivo que
    não existe. Com alterações não salvas, o SHA-256 é do arquivo como está
    em disco, que já não é exatamente o conteúdo usado; a flag diz isso.
    """
    p = Path(parquet_path) if parquet_path else None
    existe = bool(p and p.exists())
    sha = None
    sha_error = None
    if existe:
        try:
            if hash_service is None:
                from faceaudit.services.hash_service import HashService
                hash_service = HashService()
            sha = hash_service.calculate_file_hash(p)
        except Exception as e:  # registra a falha em vez de omitir em silêncio
            sha_error = str(e)
    cabeca = None
    if existe:
        try:
            cadeia = AuditLogService.load(p)
            cabeca = cadeia[-1].entry_hash if cadeia else None
        except Exception:
            cabeca = None
    out: Dict[str, Any] = {
        "name": getattr(dataset, "name", None),
        "parquet_path": str(p) if p else None,
        "parquet_sha256": sha,
        "has_unsaved_changes": bool(has_unsaved_changes or not existe),
        "rows": len(getattr(dataset, "faces", []) or []),
        "identities": _n_identities(dataset),
        "is_aligned": bool(getattr(dataset, "is_aligned", False)),
        "path_mode": _enum_value(getattr(dataset, "path_mode", None)),
        "audit_chain_head": cabeca,
    }
    if sha_error:
        out["parquet_sha256_error"] = sha_error
    if content:
        out.update(content)
    return out


def merge_entry(
    base: Dict[str, Any],
    other: Dict[str, Any],
    result: Any,
    *,
    mixed_alignment: bool,
    mixed_alignment_confirmed: bool,
) -> Tuple[str, Dict[str, Any]]:
    """(descrição, metadata) do MERGE, primeira entrada do dataset unido."""
    path_mode_result = _enum_value(getattr(result, "path_mode", None))
    meta: Dict[str, Any] = {
        "operation": "dataset_merge",
        "sources": [base, other],
        "rows": len(getattr(result, "faces", []) or []),
        "identities": _n_identities(result),
        "rows_from_sources": [base.get("rows"), other.get("rows")],
        "is_aligned": bool(getattr(result, "is_aligned", False)),
        "path_mode": path_mode_result,
        # Âncoras diferentes viram caminhos absolutos — decisão do código,
        # registrada porque muda o que o arquivo afirma sobre as imagens.
        "paths_absolutized": (
            path_mode_result is not None
            and path_mode_result != base.get("path_mode")
        ),
        # Um só lado alinhado: o perito foi avisado e escolheu unir.
        "mixed_alignment": bool(mixed_alignment),
        "mixed_alignment_confirmed_by_operator": bool(mixed_alignment_confirmed),
        "duplicates_removed": False,
    }
    descricao = (
        f"Dataset unido a partir de '{base.get('name')}' "
        f"({base.get('rows')} registro(s)) e '{other.get('name')}' "
        f"({other.get('rows')} registro(s)); total {meta['rows']}."
    )
    if mixed_alignment:
        descricao += (
            " Só um dos lados era de faces alinhadas; o operador confirmou "
            "a união e o resultado ficou marcado como não alinhado."
        )
    return descricao, meta


def default_split_seed() -> Optional[int]:
    """Semente que `FaceDataset.split_by_identity` usa quando não recebe uma."""
    from faceaudit.models.face_dataset import FaceDataset

    try:
        param = inspect.signature(FaceDataset.split_by_identity).parameters["random_seed"]
        return param.default if param.default is not inspect.Parameter.empty else None
    except (KeyError, TypeError, ValueError):
        return None


def split_entry(
    source: Dict[str, Any],
    *,
    part_label: str,
    part: Any,
    sibling: Any,
    sibling_path: Optional[str | Path],
    mode: str,
    ratio: Optional[float],
    random_seed: Optional[int],
    ids_for_b: Optional[Iterable[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """(descrição, metadata) do SPLIT, primeira entrada de UMA metade.

    `mode` é `"ratio"` (sorteio de identidades) ou `"manual"` (lista de
    identidades de B). A semente só vale no sorteio.
    """
    criterio: Dict[str, Any] = {"mode": mode, "unit": "identity"}
    if mode == "ratio":
        criterio["ratio_identities_to_a"] = ratio
        criterio["random_seed"] = random_seed
    else:
        lista = sorted(str(i) for i in (ids_for_b or []))
        criterio["ids_for_b_count"] = len(lista)
        criterio["ids_for_b"] = lista[:MAX_AUDIT_IDENTIFIERS]
        criterio["ids_for_b_truncated"] = len(lista) > MAX_AUDIT_IDENTIFIERS
        criterio["ids_for_b_sha256"] = set_sha256(lista)
    ids_parte = getattr(part, "unique_identities", None) or []
    meta: Dict[str, Any] = {
        "operation": "dataset_split",
        "part": part_label,
        "source": source,
        "criterion": criterio,
        "rows": len(getattr(part, "faces", []) or []),
        "identities": _n_identities(part),
        "identities_sha256": set_sha256(ids_parte),
        "sibling": {
            "name": getattr(sibling, "name", None),
            "parquet_path": str(sibling_path) if sibling_path else None,
            "rows": len(getattr(sibling, "faces", []) or []),
            "identities": _n_identities(sibling),
        },
    }
    como = (
        f"sorteio de identidades (proporção {ratio} para A, semente {random_seed})"
        if mode == "ratio"
        else f"lista manual de {criterio['ids_for_b_count']} identidade(s) para B"
    )
    descricao = (
        f"Metade {part_label} da divisão de '{source.get('name')}' por {como}: "
        f"{meta['rows']} registro(s), {meta['identities']} identidade(s)."
    )
    return descricao, meta


# ── Entradas pendentes (dataset ainda sem arquivo) ─────────────────────────


def queue_pending(
    dataset: Any,
    operation: AuditOperation,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Guarda no dataset uma entrada a escrever quando ele for gravado.

    Valida JÁ (justificativa obrigatória etc.): recusar só na gravação
    deixaria a operação aplicada sem registro possível.
    """
    validate_entry_request(operation, metadata)
    item = {
        "operation": operation,
        "description": description,
        "metadata": dict(metadata or {}),
        "recorded_at": _utc_now(),
    }
    lista = getattr(dataset, _PENDING_ATTR, None)
    if lista is None:
        lista = []
        setattr(dataset, _PENDING_ATTR, lista)
    lista.append(item)
    return item


def pending_entries(*datasets: Any) -> List[Dict[str, Any]]:
    """Pendentes de todos os datasets dados, sem repetição, em ordem do ato.

    Recebe vários porque o que se grava pode ser um recorte (filtro,
    seleção) do dataset que acumulou as pendências — e elas descrevem a
    história do conteúdo gravado.
    """
    vistos = set()
    out: List[Dict[str, Any]] = []
    for ds in datasets:
        if ds is None:
            continue
        for item in getattr(ds, _PENDING_ATTR, None) or []:
            if id(item) in vistos:
                continue
            vistos.add(id(item))
            out.append(item)
    out.sort(key=lambda it: it.get("recorded_at") or "")
    return out


def inherit_pending(target: Any, *sources: Any) -> None:
    """O derivado em memória herda (copia) as pendências das origens."""
    itens = pending_entries(*sources)
    if not itens:
        return
    lista = getattr(target, _PENDING_ATTR, None)
    if lista is None:
        lista = []
        setattr(target, _PENDING_ATTR, lista)
    ja = {id(i) for i in lista}
    lista.extend(i for i in itens if id(i) not in ja)


def clear_pending(dataset: Any) -> None:
    if dataset is not None and getattr(dataset, _PENDING_ATTR, None):
        setattr(dataset, _PENDING_ATTR, [])


def write_pending(
    parquet_path: str | Path, entries: Sequence[Dict[str, Any]],
) -> int:
    """Escreve as pendências na cadeia do arquivo. Devolve quantas."""
    for item in entries:
        meta = dict(item.get("metadata") or {})
        meta.setdefault("recorded_at", item.get("recorded_at"))
        meta["written_on_save"] = True
        AuditLogService.append(
            parquet_path,
            item["operation"],
            item["description"],
            metadata=meta,
        )
    return len(entries)


# ── Helpers ────────────────────────────────────────────────────────────────


def _n_identities(dataset: Any) -> Optional[int]:
    try:
        return int(dataset.number_of_identities)
    except Exception:
        return None


def _enum_value(v: Any) -> Any:
    return getattr(v, "value", v)
