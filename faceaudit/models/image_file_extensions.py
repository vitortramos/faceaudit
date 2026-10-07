"""Quais extensões de arquivo o sistema aceita como imagem.

A lista estava escrita três vezes, em três lugares que não se conheciam: a
ingestão de pasta, a geração de imagens pasta a pasta e o seletor de
arquivo do fluxo de vídeo. Nenhuma delas trazia `.jfif`, que é como o
Windows costuma nomear imagem salva do navegador — e JFIF é o próprio
formato de arquivo do JPEG, lido sem problema por Pillow e por
`cv2.imdecode`, que decidem pelo conteúdo e não pela extensão. Eram
imagens perfeitamente legíveis que o sistema nunca listava.

Pior que a falta era o silêncio: o filtro descartava o que não casava sem
contar nem avisar. Uma pasta só de `.jfif` rendia um dataset de zero faces
sem nenhuma mensagem, e uma pasta mista rendia menos faces do que tinha
arquivos, sem nada apontando a diferença. Daí `classificar`, que devolve o
que ficou de fora agrupado por extensão para quem chama poder reportar.

Uma lista só, num lugar só, é o que impede o próximo formato de entrar em
duas das três e sumir na terceira.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Todas em minúsculas e com ponto. `.jfif`, `.jfi` e `.jif` são JPEG com
# outro nome; `.jpe` idem. `.tif` faltava ao lado de `.tiff` pelo mesmo
# descuido de lista.
EXTENSOES_DE_IMAGEM: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".jpe", ".jfif", ".jfi", ".jif",
    ".png", ".bmp", ".webp", ".tif", ".tiff",
)

_CONJUNTO = frozenset(EXTENSOES_DE_IMAGEM)


def e_imagem(caminho: Path | str) -> bool:
    """True quando a extensão está entre as aceitas."""
    return Path(caminho).suffix.lower() in _CONJUNTO


@dataclass(frozen=True)
class TriagemDeArquivos:
    """O que entrou, o que ficou de fora e por qual extensão."""

    imagens: Tuple[Path, ...] = ()
    ignorados_por_extensao: Dict[str, int] = field(default_factory=dict)

    @property
    def n_ignorados(self) -> int:
        return sum(self.ignorados_por_extensao.values())

    def aviso(self) -> str:
        """Frase única para o log, ou vazio quando nada ficou de fora.

        Lista as extensões em ordem decrescente de quantidade: quem olha
        quer saber primeiro o que pode ter perdido em volume.
        """
        if not self.ignorados_por_extensao:
            return ""
        partes = ", ".join(
            f"{ext or '(sem extensão)'}: {n}"
            for ext, n in sorted(
                self.ignorados_por_extensao.items(),
                key=lambda par: (-par[1], par[0]),
            )
        )
        plural = "s" if self.n_ignorados != 1 else ""
        return (
            f"⚠️  {self.n_ignorados} arquivo{plural} na pasta "
            f"não entrou por causa da extensão ({partes}). "
            f"Aceitas: {', '.join(EXTENSOES_DE_IMAGEM)}."
        )


def classificar(caminhos: Iterable[Path | str]) -> TriagemDeArquivos:
    """Separa as imagens do resto, contando o resto por extensão.

    Pastas são ignoradas sem entrar na contagem — não são candidatas a
    imagem e poluiriam o aviso. Tudo mais entra, inclusive `Thumbs.db` e
    afins: eles viram uma linha de `.db: 1`, fácil de descartar a olho, e
    filtrá-los por uma lista de "lixo conhecido" esconderia justamente o
    caso que o aviso existe para pegar.
    """
    imagens: List[Path] = []
    ignorados: Dict[str, int] = {}
    for bruto in caminhos:
        p = Path(bruto)
        if p.is_dir():
            continue
        if e_imagem(p):
            imagens.append(p)
        else:
            ext = p.suffix.lower()
            ignorados[ext] = ignorados.get(ext, 0) + 1
    return TriagemDeArquivos(tuple(imagens), ignorados)


def filtro_de_dialogo(rotulo: str = "Imagens") -> str:
    """Filtro para `QFileDialog`, derivado da mesma lista.

    Escrito à mão, o filtro do seletor divergia da lista da ingestão: o
    usuário conseguia escolher um arquivo que o sistema depois descartava,
    ou o contrário.
    """
    padroes = " ".join(f"*{ext}" for ext in EXTENSOES_DE_IMAGEM)
    return f"{rotulo} ({padroes});;Todos (*.*)"
