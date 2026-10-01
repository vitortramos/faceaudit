"""Metadados de captura de um arquivo de VÍDEO.

Duas coisas distinguem isto dos metadados de uma foto.

A primeira: todas as faces de um vídeo vêm do MESMO arquivo, então a
procedência é a mesma para todas — mas cada face tem o seu instante
(`timestamp_s_from_fps`). Com um início de gravação confiável, a data de
captura de cada face é `início + instante`, e isso responde a uma pergunta
que a data do arquivo não responde: **quando esta pessoa foi filmada**. Num
vídeo de duas horas, a diferença entre a primeira e a última face é de duas
horas.

A segunda: um vídeo declara a sua **orientação** no contêiner. O OpenCV
ignora essa etiqueta — vídeo de celular gravado em retrato é entregue
deitado. Quem lê o quadro precisa saber disso, e a etiqueta vem na mesma
leitura dos metadados, de graça.

**O que NÃO se faz aqui**: inventar fuso. A data de início só é usada quando
a etiqueta traz o deslocamento explícito, a mesma regra das fotos. O
`CreateDate` do QuickTime costuma estar em UTC sem dizer, e tratá-lo como
hora local erraria por horas, em silêncio, numa coluna que depois apareceria
num laudo. Sem fuso declarado, a data crua é preservada e a calculada fica
vazia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from faceaudit.services.image_rotation_service import normalizar_graus

# Etiquetas de início de gravação, em ordem de preferência. As primeiras
# trazem o deslocamento embutido no próprio valor; `CreateDate` quase sempre
# não traz (QuickTime grava em UTC sem declarar), e por isso vem por último.
_ETIQUETAS_INICIO: Tuple[str, ...] = (
    "CreationDate",            # QuickTime/Apple — "2026:09:20 14:32:10-03:00"
    "DateTimeOriginal",
    "ContentCreateDate",
    "DateCreated",
    "MediaCreateDate",
    "TrackCreateDate",
    "CreateDate",
)

# Rotação declarada no contêiner.
_ETIQUETAS_ROTACAO: Tuple[str, ...] = (
    "Rotation", "VideoRotation", "TrackRotation", "ImageRotation",
)

# Como a data de captura de uma linha foi obtida. Sem isto, dois valores ISO
# de aparência idêntica significariam coisas diferentes entre datasets — a
# ambiguidade exata que uma coluna de laudo não pode ter.
ORIGEM_ETIQUETA_DO_ARQUIVO = "FILE_TAG"
ORIGEM_INICIO_MAIS_QUADRO = "VIDEO_START_PLUS_FRAME"


@dataclass(frozen=True)
class VideoCaptureMetadata:
    """Leitura única por arquivo de vídeo."""

    tags: Dict[str, Any] = field(default_factory=dict)
    # Início da gravação, COM fuso. None quando nenhuma etiqueta trouxe
    # deslocamento explícito — e aí não há data por quadro a calcular.
    start_datetime: Optional[datetime] = None
    # O valor cru da etiqueta, preservado mesmo sem fuso: é o que o arquivo
    # declara, e sumir com isso seria pior que não poder normalizá-lo.
    start_raw: Optional[str] = None
    start_tag: Optional[str] = None
    rotation_degrees: int = 0
    error: Optional[str] = None

    @classmethod
    def from_manifest_entry(cls, entrada: Dict[str, Any]) -> "VideoCaptureMetadata":
        """Reconstrói a partir do que a extração gravou no manifesto.

        Existe como método público porque o importador precisa disto e não
        deve alcançar o interpretador de data por dentro do módulo — função
        privada compartilhada entre módulos é o acoplamento que faz os dois
        divergirem no primeiro ajuste de qualquer um deles.
        """
        if not isinstance(entrada, dict):
            return cls()
        return cls(
            tags=dict(entrada.get("tags") or {}),
            start_datetime=_parse_inicio_com_fuso(
                entrada.get("start_datetime_iso") or entrada.get("start_datetime_raw")
            ),
            start_raw=entrada.get("start_datetime_raw"),
            start_tag=entrada.get("start_datetime_tag"),
            rotation_degrees=_parse_rotacao(entrada.get("container_rotation")),
            error=entrada.get("error"),
        )

    def to_manifest_entry(self, incluir_tags: bool = True) -> Dict[str, Any]:
        """O que a extração grava no manifesto. Par de `from_manifest_entry`.

        `incluir_tags=False` grava só o que a extração USOU (rotação e
        início): a rotação é sempre lida, mas o dump completo só interessa a
        quem pediu os metadados.
        """
        entrada: Dict[str, Any] = {
            "container_rotation": self.rotation_degrees,
            "start_datetime_raw": self.start_raw,
            "start_datetime_tag": self.start_tag,
            "start_datetime_iso": (
                self.start_datetime.isoformat() if self.start_datetime else None
            ),
            "timezone_declared": self.start_datetime is not None,
        }
        if self.error:
            entrada["error"] = self.error
        if incluir_tags:
            entrada["tags"] = self.tags
        return entrada

    @property
    def tem_inicio_confiavel(self) -> bool:
        return self.start_datetime is not None

    @property
    def declara_rotacao(self) -> bool:
        return self.rotation_degrees != 0


def probe_video_capture_metadata(
    exif_service, video_path,
) -> VideoCaptureMetadata:
    """Uma leitura do ExifTool por vídeo. Nunca levanta.

    Falha de leitura devolve um objeto com `error` preenchido e os demais
    campos vazios: a extração de vídeo não pode ser interrompida porque o
    metadado não pôde ser lido.
    """
    try:
        metadata = exif_service.extract(video_path)
    except Exception as e:                       # noqa: BLE001
        return VideoCaptureMetadata(error=str(e))

    tags = dict(getattr(metadata, "tags", None) or {})
    erro = getattr(metadata, "error", None)

    bruto, etiqueta = _primeiro_valor(metadata, _ETIQUETAS_INICIO)
    inicio = _parse_inicio_com_fuso(bruto)
    rotacao_bruta, _ = _primeiro_valor(metadata, _ETIQUETAS_ROTACAO)

    return VideoCaptureMetadata(
        tags=tags,
        start_datetime=inicio,
        start_raw=str(bruto).strip() if bruto else None,
        start_tag=etiqueta,
        rotation_degrees=_parse_rotacao(rotacao_bruta),
        error=erro,
    )


def columns_for_frame(
    meta: VideoCaptureMetadata,
    timestamp_s: Optional[float],
    base_columns: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Colunas de captura de UMA face, a partir do vídeo de origem.

    `base_columns` é o resultado de `build_capture_columns` para o vídeo —
    idêntico para todas as faces do mesmo arquivo. Quem chama deve calculá-lo
    UMA vez e passar aqui: além do tempo, isso faz todas as linhas
    compartilharem a mesma instância de string do dump completo, em vez de
    uma cópia por face.

    O que varia por face é a data: `início + instante`. Sem início confiável,
    `capture_datetime` fica vazia e a crua do vídeo é preservada.
    """
    colunas: Dict[str, Any] = dict(base_columns or {})

    # A data do ARQUIVO não vale para o quadro: ela é o começo da gravação, e
    # usá-la em todas as faces afirmaria que todas foram capturadas no mesmo
    # instante. Só a calculada entra; a crua permanece em `_raw`.
    colunas.pop("capture_datetime", None)
    colunas.pop("capture_datetime_origin", None)

    if meta.start_raw and not colunas.get("capture_datetime_raw"):
        colunas["capture_datetime_raw"] = meta.start_raw

    calculada = datetime_do_quadro(meta, timestamp_s)
    if calculada:
        colunas["capture_datetime"] = calculada
        colunas["capture_datetime_origin"] = ORIGEM_INICIO_MAIS_QUADRO

    return colunas


def datetime_do_quadro(
    meta: VideoCaptureMetadata, timestamp_s: Optional[float],
) -> Optional[str]:
    """`início + instante` em ISO-8601 com fuso, ou None.

    None quando não há início com fuso explícito OU quando a face não tem
    instante — nos dois casos, o que falta é dado, e preencher seria
    afirmar o que não se sabe.
    """
    if meta.start_datetime is None or timestamp_s is None:
        return None
    try:
        deslocamento = float(timestamp_s)
    except (TypeError, ValueError):
        return None
    if deslocamento < 0:
        return None
    return (meta.start_datetime + timedelta(seconds=deslocamento)).isoformat()


# ── Leitura das etiquetas ────────────────────────────────────────────────

def _primeiro_valor(metadata, nomes: Tuple[str, ...]):
    """Primeiro nome com valor, e qual foi. Respeita a ordem dada.

    Usa `get_first` do `ImageMetadata`, que resolve nomes com prefixo de
    grupo (`QuickTime:CreationDate` casa com `CreationDate`).
    """
    for nome in nomes:
        valor = metadata.get_first(nome)
        if valor not in (None, "", []):
            return valor, nome
    return None, None


def _parse_inicio_com_fuso(bruto: Any) -> Optional[datetime]:
    """Converte a etiqueta num `datetime` COM fuso, ou devolve None.

    Aceita os formatos que o ExifTool emite para vídeo:
      "2026:09:20 14:32:10-03:00"   → ok
      "2026:09:20 14:32:10+0000"    → ok (normaliza o deslocamento)
      "2026:09:20 14:32:10Z"        → ok
      "2026:09:20 14:32:10"         → None (sem fuso declarado)
    """
    if not bruto:
        return None
    texto = str(bruto).strip()
    if not texto:
        return None

    # EXIF grava a data com ":"; o ISO quer "-" (e só nos dois primeiros).
    if len(texto) >= 10 and texto[4] == ":" and texto[7] == ":":
        texto = texto[:4] + "-" + texto[5:7] + "-" + texto[8:]
    texto = texto.replace(" ", "T", 1)

    if texto.endswith(("Z", "z")):
        texto = texto[:-1] + "+00:00"
    # Deslocamento sem dois-pontos ("+0000" → "+00:00"). Procura o sinal
    # depois da parte de hora para não confundir com a data.
    corte = max(texto.rfind("+"), texto.rfind("-"))
    if corte > 10:
        desl = texto[corte:]
        if len(desl) == 5 and desl[1:].isdigit():
            texto = texto[:corte] + desl[:3] + ":" + desl[3:]

    try:
        resultado = datetime.fromisoformat(texto)
    except ValueError:
        return None
    # A regra: sem deslocamento explícito, não há data normalizada. O
    # `CreateDate` do QuickTime quase sempre está em UTC sem dizer, e
    # tratá-lo como hora local erraria por horas, em silêncio.
    return resultado if resultado.tzinfo is not None else None


def _parse_rotacao(bruto: Any) -> int:
    """Rotação declarada, em múltiplo de 90°. Qualquer outra coisa → 0.

    Vale 0 também para valor inesperado: girar por um palpite seria pior
    que não girar, porque sairia um dataset torto sem explicação.
    """
    if bruto in (None, "", []):
        return 0
    texto = str(bruto).strip().rstrip("°").split()[0] if str(bruto).strip() else ""
    try:
        return normalizar_graus(int(float(texto)))
    except Exception:                            # noqa: BLE001
        return 0
