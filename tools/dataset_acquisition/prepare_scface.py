#!/usr/bin/env python3
"""Preparação do SCface (Surveillance Cameras Face database).

SCface NÃO tem URL pública. O acesso é via request por email após
assinar o EULA. Procedimento:

    1. Acesse https://www.scface.org/
    2. Baixe o EULA, preencha (precisa carimbo institucional) e envie
       assinado por email pro Prof. Mislav Grgic (U.Zagreb). Endereço de
       contato está no site.
    3. Você recebe link de download em ~1-7 dias úteis.
    4. Use este script pra organizar o `SCface_database.zip` que chegar.

Estrutura típica do zip original (130 sujeitos):

    SCface_database/
        mugshot_frontal_cropped_all/
            001_frontal.JPG ... 130_frontal.JPG          ← galeria 'foto-padrão'
        mugshot_frontal_original/
            001_frontal_orig.JPG ...                     ← foto-padrão sem crop
        surveillance_cameras_distance_1/
            001_cam1_1.jpg, 001_cam2_1.jpg, ... 001_cam5_1.jpg
            ... (5 câmeras × 130 sujeitos = 650 imagens)
        surveillance_cameras_distance_2/   ← (idem, mais distante)
        surveillance_cameras_distance_3/   ← (idem, ainda mais distante)
        IR_frontal/                         ← variante IR (opcional)
        IR_distance_1/, IR_distance_2/, IR_distance_3/

Total esperado: 130 mugshots + 130 × 5 × 3 = 1.950 surveillance =
2.080 imagens (mais IR se incluído).

Estrutura produzida no destino:

    <dest>/
        raw/SCface_database.zip
        images/
            mugshot/<identity>/<arquivo>.jpg
            surveillance/<identity>/<arquivo>.jpg
        metadata/
            files.csv    ← image_path, identity, source, camera, distance, modality
            pairs.csv    ← pares prontos para verificação forense
        README.md

Convenção dos pares gerados:

  - GENUINE: mugshot[i] × surveillance[i] em cada câmera/distância
    (= 130 × 5 × 3 = 1950 pares genuínos por modalidade)
  - IMPOSTOR: estratégia round-robin (mugshot[i] × surveillance[j!=i]
    da mesma câmera/distância) — gera n_impostor pares por config.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional


CITATION = """Grgic, M., Delac, K., & Grgic, S. (2011). SCface — surveillance
cameras face database. Multimedia Tools and Applications, 51(3), 863-879.

Site oficial: https://www.scface.org/
"""


# Padrões dos nomes de arquivo:
# - mugshot_frontal_cropped_all/001_frontal.JPG
# - surveillance_cameras_distance_1/001_cam1_1.jpg  (id=001, cam=1, dist=1)
# - IR_distance_1/001_cam1_1.jpg                    (idem, mas IR)
_MUGSHOT_RE = re.compile(r"^(?P<id>\d{3})_frontal(?:_orig)?\.(jpg|jpeg|JPG|JPEG)$")
_SURV_RE = re.compile(
    r"^(?P<id>\d{3})_cam(?P<cam>[1-5])_(?P<dist>[1-3])\.(jpg|jpeg|JPG|JPEG)$"
)


def _classify(rel_path: Path) -> Optional[dict]:
    """Classifica uma imagem pela pasta + nome. None se não bate em
    nenhum padrão conhecido."""
    parent = rel_path.parent.name.lower()
    name = rel_path.name

    m_mug = _MUGSHOT_RE.match(name)
    if m_mug and "mugshot" in parent:
        return {
            "identity": m_mug.group("id"),
            "source": "mugshot",
            "camera": None,
            "distance": None,
            "modality": "VIS",
        }

    m_surv = _SURV_RE.match(name)
    if m_surv:
        modality = "IR" if parent.startswith("ir_") else "VIS"
        if "distance_" in parent or modality == "IR":
            return {
                "identity": m_surv.group("id"),
                "source": "surveillance",
                "camera": int(m_surv.group("cam")),
                "distance": int(m_surv.group("dist")),
                "modality": modality,
            }
    return None


def _extract_zip(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SCface] Extraindo {zip_path.name}…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return target_dir


def _organize(source_dir: Path, dest_images: Path) -> list[dict]:
    """Varre o source extraído e copia cada imagem reconhecida pra
    `dest_images/<source>/<identity>/<arquivo>`. Retorna a lista de
    dicts com metadata pra escrever no CSV."""
    rows: list[dict] = []
    n_mugshot = 0
    n_surv = 0
    n_skipped = 0

    for img in sorted(source_dir.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        rel = img.relative_to(source_dir)
        meta = _classify(rel)
        if meta is None:
            n_skipped += 1
            continue

        dest_dir = dest_images / meta["source"] / meta["identity"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / img.name
        if not dest_path.exists():
            shutil.copy2(img, dest_path)

        rows.append({
            "image_path": str(dest_path),
            "identity": meta["identity"],
            "source": meta["source"],
            "camera": meta["camera"] if meta["camera"] is not None else "",
            "distance": meta["distance"] if meta["distance"] is not None else "",
            "modality": meta["modality"],
            "original_name": img.name,
        })
        if meta["source"] == "mugshot":
            n_mugshot += 1
        else:
            n_surv += 1

    print(f"[SCface] {n_mugshot} mugshots, {n_surv} surveillance, "
          f"{n_skipped} arquivos ignorados.")
    return rows


def _generate_pairs(
    rows: list[dict],
    impostors_per_genuine: int = 5,
    only_modality: Optional[str] = "VIS",
) -> list[dict]:
    """Gera pares forenses mugshot vs surveillance.

    GENUINE: cada mugshot[id] × cada surveillance[id] na mesma câmera/dist.
    IMPOSTOR: round-robin com offset, mesmo (cam, dist), `n` por gen.

    Filtra por `only_modality` (default VIS) pra não misturar com IR.
    """
    mugshots: dict[str, str] = {}
    surv_by_config: dict[tuple[int, int], list[tuple[str, str]]] = {}

    for r in rows:
        if only_modality and r["modality"] != only_modality:
            continue
        if r["source"] == "mugshot":
            mugshots[r["identity"]] = r["image_path"]
        elif r["source"] == "surveillance":
            key = (int(r["camera"]), int(r["distance"]))
            surv_by_config.setdefault(key, []).append(
                (r["identity"], r["image_path"])
            )

    pairs: list[dict] = []
    for (cam, dist), entries in sorted(surv_by_config.items()):
        ids_in_config = [eid for eid, _ in entries]
        for idx, (identity, surv_path) in enumerate(entries):
            mug_path = mugshots.get(identity)
            if mug_path is None:
                continue
            pairs.append({
                "probe_path": surv_path,
                "reference_path": mug_path,
                "probe_identity": identity,
                "reference_identity": identity,
                "same_person": True,
                "camera": cam,
                "distance": dist,
            })
            # Impostores: pega os próximos N ids diferentes no mesmo config.
            n_added = 0
            offset = 1
            while n_added < impostors_per_genuine and offset < len(ids_in_config):
                other_idx = (idx + offset) % len(ids_in_config)
                other_id, _ = entries[other_idx]
                if other_id != identity:
                    other_mug = mugshots.get(other_id)
                    if other_mug is not None:
                        pairs.append({
                            "probe_path": surv_path,
                            "reference_path": other_mug,
                            "probe_identity": identity,
                            "reference_identity": other_id,
                            "same_person": False,
                            "camera": cam,
                            "distance": dist,
                        })
                        n_added += 1
                offset += 1

    n_gen = sum(1 for p in pairs if p["same_person"])
    n_imp = len(pairs) - n_gen
    print(f"[SCface] Pares gerados: {n_gen} genuínos, {n_imp} impostores "
          f"(modalidade {only_modality or 'all'}).")
    return pairs


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"[SCface] {path.name} escrito ({len(rows)} linhas).")


def _write_readme(dest: Path, src_zip: Path) -> None:
    readme = dest / "README.md"
    readme.write_text(
        f"""# SCface (Surveillance Cameras Face database)

Origem: arquivo recebido por email do mantenedor após EULA assinado.
Arquivo bruto: `{src_zip.name}` (preservado em `raw/`).
Organizado por `tools/dataset_acquisition/prepare_scface.py`.

## Conteúdo

- 130 sujeitos
- 1 mugshot frontal por sujeito (galeria 'foto-padrão')
- 5 câmeras × 3 distâncias = 15 surveillance shots por sujeito
- Versão IR opcional (mesma estrutura, modalidade IR)

## Pares forenses

`metadata/pairs.csv` traz pares prontos pra verificação:
- Genuínos: mugshot[i] × surveillance[i, cam, dist]
- Impostores: round-robin de mugshot[j!=i] × surveillance[i, cam, dist],
  N impostores por genuíno (default 5)

Útil pra simular o cenário forense canônico:
"foto-padrão de identificação civil × frame de CFTV".

## Citação

{CITATION}

## Licença

Uso ESTRITAMENTE acadêmico, conforme EULA assinado. Republicar imagens
do dataset é proibido (incluindo em papers/posters). Respeite os
termos.
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Organiza SCface a partir do zip recebido por email.",
    )
    ap.add_argument(
        "--zip",
        required=True,
        help="Caminho local do SCface_database.zip recebido por email.",
    )
    ap.add_argument("--dest", required=True, help="Pasta de destino.")
    ap.add_argument(
        "--impostors-per-genuine",
        type=int,
        default=5,
        help="Quantos impostores gerar por par genuíno (default: 5).",
    )
    ap.add_argument(
        "--modality",
        choices=["VIS", "IR", "all"],
        default="VIS",
        help="Modalidade pra pares (default VIS).",
    )
    args = ap.parse_args()

    src_zip = Path(args.zip).expanduser().resolve()
    if not src_zip.exists():
        print(f"[SCface] Arquivo não encontrado: {src_zip}", file=sys.stderr)
        print(
            "         SCface não tem URL pública. Veja o cabeçalho deste script "
            "para o processo de request por email.",
            file=sys.stderr,
        )
        return 1

    dest = Path(args.dest).expanduser().resolve()
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Preserva o arquivo bruto
    raw_zip = raw_dir / src_zip.name
    if raw_zip != src_zip and not raw_zip.exists():
        shutil.copy2(src_zip, raw_zip)

    # Extrai
    work_dir = dest / "_extract"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    root = _extract_zip(raw_zip, work_dir)

    # Organiza e gera CSVs
    rows = _organize(root, dest / "images")
    _write_csv(
        rows,
        dest / "metadata" / "files.csv",
        ["image_path", "identity", "source", "camera", "distance",
         "modality", "original_name"],
    )

    modality = None if args.modality == "all" else args.modality
    pairs = _generate_pairs(rows, args.impostors_per_genuine, modality)
    _write_csv(
        pairs,
        dest / "metadata" / "pairs.csv",
        ["probe_path", "reference_path", "probe_identity",
         "reference_identity", "same_person", "camera", "distance"],
    )

    _write_readme(dest, src_zip)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[SCface] Concluído. Destino: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
