#!/usr/bin/env python3
"""Download e organização do CPLFW (Cross-Pose LFW).

O dataset CPLFW é distribuído pelo grupo do Prof. Weihong Deng (Beihang)
via http://www.whdeng.cn/CPLFW/  — a página tem o link de download direto
(geralmente Google Drive ou link Beihang) que pode mudar com o tempo.
Este script NÃO chuta URL: você cola o link em `--url` ou aponta `--zip`
pra um zip já baixado.

Estrutura produzida no destino:

    <dest>/
        raw/<arquivo-original>           ← zip bruto, preservado
        images/<identity>/<arquivo.jpg>  ← imagens reorganizadas por id
        metadata/
            pairs.txt                    ← protocolo de pares (mesmo do LFW)
            files.csv                    ← image_path, identity, original_name
        README.md                        ← origem, citação, licença

Citação obrigatória ao usar:
    Zheng, T., & Deng, W. (2018). Cross-Pose LFW: A Database for Studying
    Cross-Pose Face Recognition in Unconstrained Environments. Beijing
    University of Posts and Telecommunications, Technical Report 18-01.

Uso típico:
    # com URL obtida manualmente do site oficial
    python tools/dataset_acquisition/download_cplfw.py \\
        --url 'https://drive.google.com/file/d/<ID>/view' \\
        --dest ~/datasets/cplfw

    # com zip já baixado
    python tools/dataset_acquisition/download_cplfw.py \\
        --zip ~/Downloads/CPLFW.zip \\
        --dest ~/datasets/cplfw
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


CITATION = """Zheng, T., & Deng, W. (2018). Cross-Pose LFW: A Database for
Studying Cross-Pose Face Recognition in Unconstrained Environments.
Beijing University of Posts and Telecommunications, Technical Report 18-01.

Site oficial: http://www.whdeng.cn/CPLFW/
"""


def _http_download(url: str, dest_path: Path) -> None:
    """Download via urllib com barra de progresso ASCII simples.

    Não usa requests/tqdm — quer rodar sem dependências extras. Falha
    explícita se for Google Drive (precisa confirmação de download
    para arquivos grandes; instrua o usuário a baixar manualmente).
    """
    if "drive.google.com" in url:
        raise RuntimeError(
            "URL parece ser Google Drive — arquivos > 100MB pedem confirmação "
            "interativa de antivírus e não baixam direto via urllib. Baixe "
            "manualmente pelo navegador e use --zip <caminho-do-arquivo>."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CPLFW] Baixando: {url}")
    print(f"        Destino: {dest_path}")

    def _report(block_n: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, 100.0 * block_n * block_size / total)
            mb = block_n * block_size / 1e6
            total_mb = total / 1e6
            sys.stdout.write(
                f"\r        {mb:7.1f} / {total_mb:7.1f} MB ({pct:5.1f}%)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_report)
    sys.stdout.write("\n")


def _extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extrai e retorna a pasta raiz do dataset extraído."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CPLFW] Extraindo {zip_path.name} em {target_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    # Detecta a pasta raiz: o zip oficial costuma vir como CPLFW/ ou
    # aligned images/ no topo. Pega o primeiro subdir não-vazio.
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return target_dir


_FNAME_RE = re.compile(r"^(?P<identity>[^_]+(?:_[^_0-9]+)*)_(?P<idx>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _identity_from_filename(fname: str) -> Optional[str]:
    """Extrai a identidade do nome do arquivo no padrão LFW.

    Convenção LFW/CPLFW: `Aaron_Eckhart_0001.jpg` → identidade
    `Aaron_Eckhart`. O sufixo `_NNNN` é o índice da imagem dentro
    da identidade.
    """
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return m.group("identity")


def _organize_by_identity(
    source_dir: Path, dest_images: Path
) -> list[tuple[Path, str, str]]:
    """Copia imagens pra `dest/<identity>/<file>` e devolve a lista
    `(novo_path, identity, nome_original)` pra escrita do CSV.

    Imagens cujo nome não bate com o padrão LFW vão pra `__unparsed/`
    com identity "unknown" — não silencia, pra você poder inspecionar
    o que sobrou (geralmente metadata files dentro do zip).
    """
    dest_images.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Path, str, str]] = []
    n_known = 0
    n_unparsed = 0

    for img in sorted(source_dir.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        identity = _identity_from_filename(img.name)
        if identity is None:
            identity = "unknown"
            n_unparsed += 1
        else:
            n_known += 1
        dest_id_dir = dest_images / identity
        dest_id_dir.mkdir(exist_ok=True)
        dest_path = dest_id_dir / img.name
        if not dest_path.exists():
            shutil.copy2(img, dest_path)
        rows.append((dest_path, identity, img.name))

    print(f"[CPLFW] Imagens organizadas: {n_known} reconhecidas, "
          f"{n_unparsed} em 'unknown/'")
    return rows


def _copy_pairs_file(source_dir: Path, dest_metadata: Path) -> Optional[Path]:
    """Procura `pairs.txt` (ou variantes) e copia pro destino. Retorna
    None se não encontrar — alguns mirrors distribuem só as imagens."""
    candidates = list(source_dir.rglob("pairs*.txt"))
    if not candidates:
        return None
    src = candidates[0]
    dest_metadata.mkdir(parents=True, exist_ok=True)
    dst = dest_metadata / "pairs.txt"
    shutil.copy2(src, dst)
    print(f"[CPLFW] Copiado pairs.txt ({src.stat().st_size} bytes)")
    return dst


def _write_files_csv(rows: list[tuple[Path, str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "identity", "original_name"])
        for p, identity, original in rows:
            w.writerow([str(p), identity, original])
    print(f"[CPLFW] CSV escrito: {csv_path} ({len(rows)} linhas)")


def _write_readme(dest: Path, source_label: str) -> None:
    readme = dest / "README.md"
    readme.write_text(
        f"""# CPLFW (Cross-Pose LFW)

Fonte: {source_label}
Organizado por `tools/dataset_acquisition/download_cplfw.py`.

## Citação

{CITATION}

## Estrutura

- `raw/`       — arquivo zip original
- `images/`    — imagens organizadas em `<identity>/<arquivo>.jpg`
- `metadata/`
  - `pairs.txt`  — protocolo oficial de pares (mesmo formato LFW)
  - `files.csv`  — image_path, identity, original_name

## Licença

CPLFW segue a mesma licença do LFW: uso restrito a pesquisa não-comercial.
Releia os termos no site oficial antes de usar.
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Download/organização do CPLFW.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="URL direta do zip (não Google Drive).")
    src.add_argument("--zip", help="Caminho local pra zip já baixado.")
    ap.add_argument("--dest", required=True, help="Pasta de destino.")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        zip_name = args.url.rstrip("/").split("/")[-1] or "CPLFW.zip"
        zip_path = raw_dir / zip_name
        if zip_path.exists():
            print(f"[CPLFW] {zip_path} já existe — pulando download.")
        else:
            _http_download(args.url, zip_path)
        source_label = args.url
    else:
        src_zip = Path(args.zip).expanduser().resolve()
        if not src_zip.exists():
            print(f"[CPLFW] zip não encontrado: {src_zip}", file=sys.stderr)
            return 1
        zip_path = raw_dir / src_zip.name
        if zip_path != src_zip:
            shutil.copy2(src_zip, zip_path)
        source_label = f"arquivo local {src_zip}"

    work_dir = dest / "_extract"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    root = _extract_zip(zip_path, work_dir)

    rows = _organize_by_identity(root, dest / "images")
    _copy_pairs_file(root, dest / "metadata")
    _write_files_csv(rows, dest / "metadata" / "files.csv")
    _write_readme(dest, source_label)

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[CPLFW] Concluído. Destino: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
