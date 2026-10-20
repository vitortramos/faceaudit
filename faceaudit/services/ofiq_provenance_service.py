"""De onde vieram as métricas do OFIQ: versão, modo de chamada e modelos.

O `OFIQSampleApp` não informa a própria versão (não tem `--version`; `-h` é
recusado), e a v1.0.3 — a implementação de referência da ISO/IEC 29794-5 — e
a v1.2.x — implementação conforme — produzem CSVs que um laudo não pode
confundir. A versão sai do `Version.txt` que toda instalação do OFIQ traz na
raiz, o mesmo arquivo que o CMake do projeto lê para numerar o build.

Os modelos entram pela impressão digital: são 22 arquivos baixados à parte da
compilação, e trocar um deles muda as métricas sem mudar a versão.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

# Até onde subir a partir do executável procurando o `Version.txt`. Na
# instalação padrão ele está três níveis acima de
# `install_x86_64/Release/bin/OFIQSampleApp.exe`.
_NIVEIS_ACIMA_DO_EXECUTAVEL = 5

ROTULO_VERSAO_DESCONHECIDA = "desconhecida"


def detect_ofiq_version(exe_path: Optional[str]) -> Optional[str]:
    """Versão do OFIQ dona deste executável (ex.: "1.0.3"), ou None."""
    if not exe_path or not os.path.isabs(exe_path):
        # Nome solto ("OFIQSampleApp.exe" a ser achado no PATH): sem pasta
        # de instalação para consultar.
        return None
    pasta = Path(exe_path).parent
    for _ in range(_NIVEIS_ACIMA_DO_EXECUTAVEL):
        candidato = pasta / "Version.txt"
        if candidato.is_file():
            return parse_version_file(candidato.read_text(encoding="utf-8", errors="replace"))
        if pasta.parent == pasta:
            break
        pasta = pasta.parent
    return None


def parse_version_file(texto: str) -> Optional[str]:
    """`VERSION_MAJOR 1` / `VERSION_MINOR 0` / ... → "1.0.3" (com sufixo, se houver)."""
    partes = {}
    for chave in ("MAJOR", "MINOR", "PATCH", "SUFFIX"):
        m = re.search(rf"^\s*VERSION_{chave}[ \t]+(\S*)[ \t]*$", texto, re.MULTILINE)
        partes[chave] = m.group(1) if m else ""
    if not (partes["MAJOR"] and partes["MINOR"] and partes["PATCH"]):
        return None
    versao = f'{partes["MAJOR"]}.{partes["MINOR"]}.{partes["PATCH"]}'
    if partes["SUFFIX"]:
        versao += f'-{partes["SUFFIX"]}'
    return versao


def models_dir_for(data_path: Optional[str]) -> Optional[Path]:
    """Pasta `models` a partir do que a config chama de dados do OFIQ.

    Aceita a pasta `data` ou o próprio `ofiq_config.jaxn` dentro dela —
    o `-c` do `OFIQSampleApp` aceita os dois.
    """
    if not data_path:
        return None
    p = Path(data_path)
    base = p.parent if p.suffix.lower() == ".jaxn" else p
    models = base / "models"
    return models if models.is_dir() else None


def models_fingerprint(data_path: Optional[str]) -> Optional[str]:
    """SHA-256 do conjunto de modelos: caminho relativo + SHA-256 de cada arquivo.

    Ignora os `.md` (README e licenças), que mudam entre distribuições sem
    mudar o que o OFIQ calcula. None se não houver pasta de modelos ou se
    ela estiver vazia — que é exatamente o caso de uma instalação copiada
    sem os modelos, e não pode passar por "conjunto válido".
    """
    models = models_dir_for(data_path)
    if models is None:
        return None
    arquivos = sorted(
        f for f in models.rglob("*")
        if f.is_file() and f.suffix.lower() != ".md"
    )
    if not arquivos:
        return None
    conjunto = hashlib.sha256()
    for arquivo in arquivos:
        relativo = arquivo.relative_to(models).as_posix()
        with open(arquivo, "rb") as fh:
            digest = hashlib.file_digest(fh, "sha256").hexdigest()
        conjunto.update(f"{relativo}\t{digest}\n".encode("utf-8"))
    return conjunto.hexdigest()


def version_label(version: Optional[str], backend: Optional[str]) -> str:
    """Rótulo legível de uma origem: "1.0.3 (executable)" ou "desconhecida"."""
    if not version:
        return ROTULO_VERSAO_DESCONHECIDA
    return f"{version} ({backend})" if backend else version


def summarize_ofiq_versions(faces: Iterable) -> Dict[str, int]:
    """Quantas faces com métrica do OFIQ vieram de cada versão/modo.

    Faces sem `ofiq_info` não entram: não têm métrica a atribuir.
    """
    resumo: Dict[str, int] = {}
    for face in faces:
        info = getattr(face, "ofiq_info", None)
        if info is None:
            continue
        rotulo = version_label(getattr(info, "version", None), getattr(info, "backend", None))
        resumo[rotulo] = resumo.get(rotulo, 0) + 1
    return resumo


def mixed_versions_message(resumo: Dict[str, int]) -> Optional[str]:
    """Texto de aviso quando o dataset tem métricas de mais de uma origem.

    Misturar não é erro — enriquecer só o que falta é o comportamento
    normal —, mas compara números que podem não ser comparáveis: da v1.0.3
    para a v1.2.0 os valores brutos de BackgroundUniformity e
    NoHeadCoverings mudam, e uma origem "desconhecida" pode ser qualquer
    versão.
    """
    if len(resumo) <= 1:
        return None
    partes = ", ".join(
        f"{rotulo}: {n} face(s)" for rotulo, n in sorted(resumo.items())
    )
    return (
        "As métricas do OFIQ deste dataset vieram de mais de uma origem — "
        f"{partes}. Valores de versões diferentes podem não ser comparáveis "
        "entre si; para um conjunto homogêneo, reprocesse o OFIQ em todas as "
        "faces."
    )
