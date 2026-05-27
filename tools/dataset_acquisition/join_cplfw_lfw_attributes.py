#!/usr/bin/env python3
"""Cruza imagens do CPLFW com atributos do LFW Attributes Database.

LFW Attributes Database (Kumar et al., Columbia):
    Kumar, N., Berg, A. C., Belhumeur, P. N., & Nayar, S. K. (2011).
    Describable Visual Attributes for Face Verification and Image Search.
    IEEE PAMI 33(10): 1962-1977.

O arquivo `lfw_attributes.txt` é distribuído pela Columbia University,
em texto plano com 73 atributos por (person_name, imagenum). Como o CPLFW
é um subset reorganizado do LFW (mesmas pessoas, mesma convenção de
nome de arquivo `<Person_Name>_<NNNN>.jpg`), o join é direto por nome
da pessoa + número da imagem.

Procedimento:

    1. Baixar `lfw_attributes.txt` da página da Columbia (busque por
       "LFW attribute scores file" no site cs.columbia.edu).
    2. Rodar:

           python tools/dataset_acquisition/join_cplfw_lfw_attributes.py \\
               --cplfw ~/Documents/bases/cplfw \\
               --attributes ~/Downloads/lfw_attributes.txt \\
               --out ~/Documents/bases/cplfw/metadata/demographics.csv

Output CSV (`demographics.csv`):

    Identificação:
        image_path, identity, image_num, is_matched
    Categorias derivadas (conveniência pra estratificação):
        gender, gender_score
        race_label, race_score, race_scores_json
        age_band, age_scores_json
    Atributos brutos (default: inclui todos os 73 — ver --no-raw-attributes):
        male, asian, white, black, baby, child, youth, middle_aged, senior,
        black_hair, blond_hair, brown_hair, bald, no_eyewear, eyeglasses,
        sunglasses, mustache, smiling, frowning, chubby, blurry,
        harsh_lighting, flash, soft_lighting, outdoor, curly_hair,
        wavy_hair, straight_hair, receding_hairline, bangs, sideburns,
        fully_visible_forehead, partially_visible_forehead,
        obstructed_forehead, bushy_eyebrows, arched_eyebrows, narrow_eyes,
        eyes_open, big_nose, pointy_nose, big_lips, mouth_closed,
        mouth_slightly_open, mouth_wide_open, teeth_not_visible, no_beard,
        goatee, round_jaw, double_chin, wearing_hat, oval_face, square_face,
        round_face, color_photo, posed_photo, attractive_man,
        attractive_woman, indian, gray_hair, bags_under_eyes, heavy_makeup,
        rosy_cheeks, shiny_skin, pale_skin, five_oclock_shadow,
        strong_nose_mouth_lines, wearing_lipstick, flushed_face,
        high_cheekbones, brown_eyes, wearing_earrings, wearing_necktie,
        wearing_necklace

Convenções derivadas dos scores contínuos do Columbia:

    gender = "male" se Male > 0, "female" se Male < 0,
             "unknown" se atributo ausente
    race_label = argmax{Asian, White, Black, Indian} — se o segundo
        colocado fica a < 0.3 do primeiro, marca como "mixed" (decisão
        deliberadamente conservadora pra evitar rótulo único forte
        em casos ambíguos)
    age_band = argmax{Baby, Child, Youth, Middle Aged, Senior}

Cobertura observada (CPLFW v1, 11648 imagens, 3929 identidades):

    Match exato (identity, image_num):   51.4% das imagens
    Match por identidade (qualquer img): 99.3% das imagens

Por que o gap? O CPLFW REINDEXOU as imagens (cada pessoa tem `_1`,
`_2`, `_3` — uniforme), enquanto o LFW original mantém numeração
esparsa por imagem (George W. Bush vai até `_0530`). Resultado:
`Bush_1.jpg` do CPLFW é uma imagem diferente do `Bush_1.jpg` do LFW
— mas a pessoa é a mesma.

Para subir a cobertura, use `--by-identity-mean`: pra imagens sem
match exato, usa a MÉDIA dos scores de todas as imagens dessa pessoa
no arquivo Columbia. **Trade-off importante**:

    Atributos invariantes da pessoa (gender, race, age aprox.)   → OK
    Atributos por imagem (smiling, eyeglasses, lighting, pose)   → ruidoso

A flag `is_matched` continua marcando match exato (True) vs fallback
por identidade (False) pra você filtrar depois.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Os 73 atributos do Columbia. Usamos um subset pra derivar gender /
# race / age; os restantes ficam preservados como colunas brutas no
# CSV (slug em snake_case).
_RACE_ATTRS = ("Asian", "White", "Black", "Indian")
_AGE_ATTRS = ("Baby", "Child", "Youth", "Middle Aged", "Senior")


def _slug(name: str) -> str:
    """Header do Columbia → identificador snake_case.

    'Male'                    → 'male'
    'Black Hair'              → 'black_hair'
    "5 o' Clock Shadow"       → 'five_oclock_shadow'
    'Strong Nose-Mouth Lines' → 'strong_nose_mouth_lines'

    Substitui dígito inicial por palavra em inglês (`5` → `five`) pra
    não gerar nomes de coluna que comecem com dígito — incômodo em SQL
    e em pandas com `df.col_name`.
    """
    digit_words = {
        "0": "zero", "1": "one",   "2": "two",   "3": "three", "4": "four",
        "5": "five", "6": "six",   "7": "seven", "8": "eight", "9": "nine",
    }
    s = name.strip().lower()
    if s and s[0] in digit_words:
        s = digit_words[s[0]] + s[1:]
    # remove apóstrofos e caracteres não-alfanuméricos
    cleaned: List[str] = []
    for ch in s:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in (" ", "-", "_"):
            cleaned.append("_")
        # outros (apóstrofo, etc.) viram nada
    out = "".join(cleaned)
    # colapsa múltiplos underscores
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# nome do arquivo CPLFW: `Aaron_Eckhart_0001.jpg`
_FNAME_RE = re.compile(r"^(?P<name>.+)_(?P<num>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _list_cplfw_images(cplfw_root: Path) -> List[Tuple[Path, str, int]]:
    """Retorna (path, person_name, image_num) pra cada imagem em
    `images/` (preferindo o subdir `images/` se existir, senão usa
    `aligned images/`).

    O nome da pessoa volta com underscores (`Aaron_Eckhart`) — bate
    com a convenção do arquivo Columbia que também usa underscores.
    """
    cand = [cplfw_root / "images", cplfw_root / "aligned images"]
    src = next((d for d in cand if d.is_dir()), None)
    if src is None:
        raise FileNotFoundError(
            f"Nenhuma das pastas {cand} existe sob {cplfw_root}."
        )

    rows: List[Tuple[Path, str, int]] = []
    for img in sorted(src.iterdir()):
        if not img.is_file():
            continue
        m = _FNAME_RE.match(img.name)
        if not m:
            continue
        rows.append((img, m.group("name"), int(m.group("num"))))
    print(f"[join] CPLFW: {len(rows)} imagens em {src.name}/")
    return rows


def _parse_attributes_file(
    path: Path,
) -> Tuple[List[str], Dict[Tuple[str, int], List[float]]]:
    """Parser tolerante do `lfw_attributes.txt`.

    Formato típico:
        # comentários iniciados por #
        person   imagenum   <attr1>   <attr2>   ... <attrN>
        Aaron_Eckhart   1   1.45   -1.23   0.87   ...

    Separador é variável (TABs e/ou espaços múltiplos). Linhas
    iniciadas em `#` viram comentário. Nome de pessoa pode vir com
    underscore OU com espaço — normalizamos pra underscore.
    """
    headers: Optional[List[str]] = None
    table: Dict[Tuple[str, int], List[float]] = {}
    n_rows = 0
    n_bad = 0

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            stripped = raw_line.rstrip("\r\n")
            if not stripped.strip():
                continue

            # Linha de comentário pode SER o cabeçalho! O formato
            # oficial do Columbia tem `#\tperson\timagenum\t<73 attrs>`
            # na 2ª linha. Detecta pelo padrão "person...imagenum" em
            # tokens TAB-separados após remover o `#` inicial.
            if stripped.startswith("#"):
                if headers is not None:
                    continue
                inner = stripped.lstrip("#").lstrip()
                tab_tokens = inner.split("\t") if "\t" in inner else re.split(r"\s+", inner)
                lower = [t.strip().lower() for t in tab_tokens]
                if "person" in lower and "imagenum" in lower:
                    # pula os 2 primeiros campos (person, imagenum)
                    headers = [t.strip() for t in tab_tokens[2:]]
                    print(f"[join] {len(headers)} atributos no header.")
                continue

            tab_data = stripped.split("\t") if "\t" in stripped else re.split(r"\s+", stripped)

            # Se ainda não temos headers e a linha começa com 'person',
            # também é um header (formato sem `#`).
            if headers is None:
                lower = [t.strip().lower() for t in tab_data[:2]]
                if lower and lower[0] in ("person", "person_name", "name"):
                    headers = [t.strip() for t in tab_data[2:]]
                    print(f"[join] {len(headers)} atributos no header.")
                    continue
                # Sem cabeçalho legível — atribui placeholders e segue
                headers = [f"attr_{i}" for i in range(max(0, len(tab_data) - 2))]

            try:
                person = tab_data[0].strip().replace(" ", "_")
                imagenum = int(tab_data[1])
                scores = [float(x) for x in tab_data[2:2 + len(headers)]]
                if len(scores) != len(headers):
                    raise ValueError(
                        f"esperava {len(headers)} scores, obteve {len(scores)}"
                    )
                table[(person, imagenum)] = scores
                n_rows += 1
            except (ValueError, IndexError) as exc:
                n_bad += 1
                if n_bad <= 3:
                    print(f"[join] linha ignorada: {exc} — '{stripped[:80]}…'")

    print(f"[join] LFW attributes: {n_rows} linhas válidas, {n_bad} ignoradas.")
    if not headers:
        raise ValueError("não consegui detectar cabeçalho de atributos.")
    return headers, table


def _attr_idx(headers: List[str], name: str) -> Optional[int]:
    """Busca case-insensitive + tolerante a espaços. Retorna None se
    o atributo não existe no header (pra arquivo legacy)."""
    target = name.lower().replace(" ", "")
    for i, h in enumerate(headers):
        if h.lower().replace(" ", "") == target:
            return i
    return None


def _derive_demographics(
    scores: List[float],
    headers: List[str],
    race_ambiguous_margin: float = 0.3,
) -> Dict[str, object]:
    """Deriva gender / race / age dos scores brutos.

    Decisões deliberadamente conservadoras: race vira "mixed" quando
    o gap entre top1 e top2 é < `race_ambiguous_margin`. Em rótulos
    forenses preferimos "não-categórico" a categórico errado.
    """
    out: Dict[str, object] = {}

    # Gender
    gi = _attr_idx(headers, "Male")
    if gi is not None:
        gs = scores[gi]
        out["gender"] = "male" if gs > 0 else "female"
        out["gender_score"] = round(gs, 4)
    else:
        out["gender"] = "unknown"
        out["gender_score"] = None

    # Race (argmax com margin)
    race_scores: Dict[str, float] = {}
    for attr in _RACE_ATTRS:
        idx = _attr_idx(headers, attr)
        if idx is not None:
            race_scores[attr.lower()] = round(scores[idx], 4)
    if race_scores:
        ranked = sorted(race_scores.items(), key=lambda kv: kv[1], reverse=True)
        top, top_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
        if top_score <= 0:
            out["race_label"] = "unknown"
        elif (top_score - runner_up_score) < race_ambiguous_margin:
            out["race_label"] = "mixed"
        else:
            out["race_label"] = top
        out["race_score"] = top_score
    else:
        out["race_label"] = "unknown"
        out["race_score"] = None
    out["race_scores_json"] = json.dumps(race_scores) if race_scores else ""

    # Age band (argmax)
    age_scores: Dict[str, float] = {}
    for attr in _AGE_ATTRS:
        idx = _attr_idx(headers, attr)
        if idx is not None:
            age_scores[attr.lower().replace(" ", "_")] = round(scores[idx], 4)
    if age_scores:
        top_age = max(age_scores.items(), key=lambda kv: kv[1])
        out["age_band"] = top_age[0] if top_age[1] > 0 else "unknown"
    else:
        out["age_band"] = "unknown"
    out["age_scores_json"] = json.dumps(age_scores) if age_scores else ""

    return out


def _coverage_summary(rows: Iterable[dict]) -> None:
    """Imprime contagens por gender / race / age separando match
    exato de fallback por identidade."""
    by_gender: Dict[str, int] = defaultdict(int)
    by_race: Dict[str, int] = defaultdict(int)
    by_age: Dict[str, int] = defaultdict(int)
    n_total = 0
    n_exact = 0
    n_fallback = 0
    n_none = 0
    for r in rows:
        n_total += 1
        kind = r.get("match_kind", "none")
        if kind == "exact":
            n_exact += 1
        elif kind == "by_identity":
            n_fallback += 1
        else:
            n_none += 1
        if kind in ("exact", "by_identity"):
            by_gender[r["gender"]] += 1
            by_race[r["race_label"]] += 1
            by_age[r["age_band"]] += 1

    covered = n_exact + n_fallback
    print(
        f"\n[join] Cobertura: {covered}/{n_total} imagens "
        f"({100.0*covered/max(1,n_total):.1f}%) "
        f"[exato={n_exact}, fallback={n_fallback}, sem={n_none}]"
    )

    def _print_dist(label: str, d: Dict[str, int]) -> None:
        total = sum(d.values()) or 1
        print(f"  {label}:")
        for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {k:15s} {v:6d}  ({100.0*v/total:5.1f}%)")

    _print_dist("Gender",   by_gender)
    _print_dist("Race",     by_race)
    _print_dist("Age band", by_age)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cplfw", required=True,
        help="Pasta raiz do CPLFW (contendo images/ ou aligned images/).",
    )
    ap.add_argument(
        "--attributes", required=True,
        help="Caminho local do arquivo lfw_attributes.txt (Columbia).",
    )
    ap.add_argument(
        "--out", required=True,
        help="CSV de saída com rótulos demográficos por imagem.",
    )
    ap.add_argument(
        "--race-margin", type=float, default=0.3,
        help="Margem mínima top1−top2 pra cravar etnia (default 0.3).",
    )
    ap.add_argument(
        "--no-raw-attributes", action="store_true",
        help="Não inclui os 73 atributos brutos como colunas (CSV enxuto).",
    )
    ap.add_argument(
        "--by-identity-mean", action="store_true",
        help=(
            "Fallback por identidade: pra imagens sem match exato em "
            "(identity, image_num), usa a média dos scores de todas as "
            "imagens dessa pessoa. Útil pra gender/race/age; ruidoso pra "
            "atributos por-imagem (smiling, lighting, etc)."
        ),
    )
    args = ap.parse_args()

    cplfw = Path(args.cplfw).expanduser().resolve()
    attrs_path = Path(args.attributes).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not attrs_path.exists():
        print(f"[join] arquivo de atributos não encontrado: {attrs_path}",
              file=sys.stderr)
        return 1

    images = _list_cplfw_images(cplfw)
    headers, table = _parse_attributes_file(attrs_path)

    # Coleta os slugs dos atributos brutos preservando a ordem original
    # do header. Detecta colisões (improvável dado o header oficial,
    # mas o slugify é defensivo).
    raw_slugs: List[str] = []
    seen: set = set()
    for h in headers:
        slug = _slug(h)
        if slug in seen:
            # raríssimo — sufixa com índice pra preservar todos
            i = 2
            while f"{slug}_{i}" in seen:
                i += 1
            slug = f"{slug}_{i}"
        seen.add(slug)
        raw_slugs.append(slug)
    include_raw = not args.no_raw_attributes
    if include_raw:
        print(f"[join] Incluindo {len(raw_slugs)} colunas de atributos brutos "
              f"(use --no-raw-attributes pra CSV enxuto).")

    # Índice agregado por identidade — médias dos scores de TODAS as
    # imagens dessa pessoa no Columbia. Útil pro fallback que usa
    # gender/race/age robustos quando o image_num não bate.
    per_identity_mean: Dict[str, List[float]] = {}
    if args.by_identity_mean:
        per_identity_sum: Dict[str, List[float]] = {}
        per_identity_count: Dict[str, int] = {}
        for (person, _num), scores in table.items():
            if person not in per_identity_sum:
                per_identity_sum[person] = list(scores)
                per_identity_count[person] = 1
            else:
                acc = per_identity_sum[person]
                for i, s in enumerate(scores):
                    acc[i] += s
                per_identity_count[person] += 1
        for person, acc in per_identity_sum.items():
            n = per_identity_count[person]
            per_identity_mean[person] = [x / n for x in acc]
        print(f"[join] Fallback por identidade ativo — {len(per_identity_mean)} "
              "identidades agregadas.")

    out_rows: List[dict] = []
    n_exact = 0
    n_fallback = 0
    for img_path, person, imagenum in images:
        scores = table.get((person, imagenum))
        match_kind: Optional[str] = None  # "exact" | "by_identity" | None
        if scores is not None:
            match_kind = "exact"
            n_exact += 1
        elif args.by_identity_mean and person in per_identity_mean:
            scores = per_identity_mean[person]
            match_kind = "by_identity"
            n_fallback += 1

        if scores is None:
            row = {
                "image_path": str(img_path),
                "identity": person,
                "image_num": imagenum,
                "gender": "unknown",
                "gender_score": "",
                "race_label": "unknown",
                "race_score": "",
                "race_scores_json": "",
                "age_band": "unknown",
                "age_scores_json": "",
                "is_matched": False,
                "match_kind": "none",
            }
            if include_raw:
                for slug in raw_slugs:
                    row[slug] = ""
            out_rows.append(row)
            continue

        demo = _derive_demographics(scores, headers, args.race_margin)
        row = {
            "image_path": str(img_path),
            "identity": person,
            "image_num": imagenum,
            "gender": demo["gender"],
            "gender_score": demo["gender_score"] if demo["gender_score"] is not None else "",
            "race_label": demo["race_label"],
            "race_score": demo["race_score"] if demo["race_score"] is not None else "",
            "race_scores_json": demo["race_scores_json"],
            "age_band": demo["age_band"],
            "age_scores_json": demo["age_scores_json"],
            # is_matched=True só pra match exato — preserva semântica
            # original. Fallback é flag separado.
            "is_matched": match_kind == "exact",
            "match_kind": match_kind,
        }
        if include_raw:
            for slug, val in zip(raw_slugs, scores):
                row[slug] = round(val, 4)
        out_rows.append(row)

    if args.by_identity_mean:
        print(f"[join] Match: {n_exact} exatos + {n_fallback} por identidade "
              f"({n_exact + n_fallback}/{len(images)} = "
              f"{100*(n_exact+n_fallback)/max(1,len(images)):.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "image_path", "identity", "image_num",
        "gender", "gender_score",
        "race_label", "race_score", "race_scores_json",
        "age_band", "age_scores_json",
        "is_matched", "match_kind",
    ]
    if include_raw:
        cols.extend(raw_slugs)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[join] CSV escrito: {out_path} ({len(cols)} colunas)")

    _coverage_summary(out_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
