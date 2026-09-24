"""Extrai os metadados embutidos nos arquivos de imagem.

Dois backends, escolhidos por disponibilidade:

- **ExifTool** (Phil Harvey) — o padrão de fato em perícia. Lê EXIF, XMP,
  IPTC, ICC, *maker notes* dos fabricantes e as assinaturas de proveniência,
  e não interpreta valores por conta própria. É o backend preferido; o
  caminho do executável vem da configuração (`SystemConfig.exiftool`),
  porque na máquina do perito ele costuma vir de um zip solto, fora do PATH.
- **Pillow** — alternativa quando o ExifTool não está disponível. Lê só o
  EXIF padrão, sem XMP/IPTC/maker notes. Serve para não deixar o fluxo
  parado, mas cobre bem menos, e o resultado diz qual backend foi usado.

Duas decisões que valem para os dois backends:

- **O dump completo é preservado.** O serviço devolve TODAS as tags lidas;
  quem consome decide o que vira coluna de primeira classe e o que fica no
  JSON. Descartar no extrator seria perder informação que não se recupera
  depois — a foto original pode nem acompanhar o dataset.
- **Metadado é declaração do arquivo, não fato medido.** O serviço não
  "corrige" data sem fuso nem valida coerência; entrega o que está gravado.
  A avaliação é do perito.

Sem Qt e sem estado — o progresso sai por callback.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Tags que o ExifTool inventa sobre a leitura em si, não sobre a imagem.
# Removê-las evita poluir o dump com ruído do próprio extrator.
_TAGS_DESCARTAVEIS = {"SourceFile", "ExifToolVersion"}

# Teto de arquivos por invocação do ExifTool. Uma chamada por arquivo
# pagaria a subida do processo N vezes; uma única chamada com 20 mil
# caminhos estoura o limite de linha de comando do Windows.
_LOTE_PADRAO = 200


@dataclass
class ImageMetadata:
    """Metadados lidos de UM arquivo."""

    source_path: str
    tags: Dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.tags)

    def get_first(self, *nomes: str) -> Optional[Any]:
        """Primeiro valor presente entre `nomes`, ignorando maiúsculas.

        O ExifTool devolve a tag com o nome do grupo quando há ambiguidade
        (`EXIF:Make` vs `Make`), e nem todo arquivo traz a mesma variante —
        procurar por uma lista de sinônimos é o que torna o consumo estável.
        """
        indice = {str(k).lower(): v for k, v in self.tags.items()}
        for nome in nomes:
            alvo = nome.lower()
            if alvo in indice:
                return indice[alvo]
            # `EXIF:Make` casa com quem pediu `Make`.
            for chave, valor in indice.items():
                if chave.rsplit(":", 1)[-1] == alvo:
                    return valor
        return None


class ExifExtractionService:
    """Stateless. O caminho do ExifTool entra por injeção."""

    BACKEND_EXIFTOOL = "exiftool"
    BACKEND_PILLOW = "pillow"

    def __init__(
        self,
        exiftool_path: Optional[str] = "exiftool",
        *,
        batch_size: int = _LOTE_PADRAO,
    ) -> None:
        self._exiftool_path = self._resolver_exiftool(exiftool_path)
        self._batch_size = max(1, int(batch_size))

    # ── Disponibilidade ──────────────────────────────────────────────────

    @staticmethod
    def _resolver_exiftool(caminho: Optional[str]) -> Optional[str]:
        """Caminho utilizável do ExifTool, ou None.

        Aceita tanto um executável explícito quanto um nome resolvido pelo
        PATH. Vazio/None desliga o backend deliberadamente.
        """
        if not caminho:
            return None
        bruto = str(caminho).strip()
        if not bruto:
            return None
        arquivo = Path(bruto)
        if arquivo.is_file():
            return str(arquivo)
        encontrado = shutil.which(bruto)
        return encontrado if encontrado else None

    @property
    def exiftool_available(self) -> bool:
        return self._exiftool_path is not None

    @property
    def backend_name(self) -> str:
        return self.BACKEND_EXIFTOOL if self.exiftool_available else self.BACKEND_PILLOW

    def exiftool_version(self) -> Optional[str]:
        """Versão do ExifTool configurado, ou None se indisponível."""
        if not self._exiftool_path:
            return None
        try:
            saida = subprocess.run(
                [self._exiftool_path, "-ver"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            versao = (saida.stdout or "").strip()
            return versao or None
        except Exception:
            return None

    # ── API ──────────────────────────────────────────────────────────────

    def extract(self, image_path) -> ImageMetadata:
        """Metadados de um arquivo."""
        return self.extract_many([image_path])[0]

    def extract_many(
        self,
        image_paths: Sequence,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[ImageMetadata]:
        """Metadados de vários arquivos, preservando a ordem de entrada.

        Com o ExifTool, os arquivos são processados em lotes: uma chamada
        por arquivo pagaria a subida do processo a cada imagem, e um lote
        único estouraria o limite de linha de comando.
        """
        caminhos = [str(p) for p in image_paths]
        if not caminhos:
            return []

        if self.exiftool_available:
            return self._extract_com_exiftool(caminhos, progress_callback)
        return self._extract_com_pillow(caminhos, progress_callback)

    # ── Backend ExifTool ─────────────────────────────────────────────────

    def _extract_com_exiftool(
        self,
        caminhos: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> List[ImageMetadata]:
        por_arquivo: Dict[str, ImageMetadata] = {}
        total = len(caminhos)

        for inicio in range(0, total, self._batch_size):
            lote = caminhos[inicio:inicio + self._batch_size]
            if progress_callback:
                progress_callback(inicio, total, lote[0])
            for metadata in self._rodar_exiftool(lote):
                por_arquivo[self._normalizar(metadata.source_path)] = metadata

        if progress_callback:
            progress_callback(total, total, "")

        resultado = []
        for caminho in caminhos:
            metadata = por_arquivo.get(self._normalizar(caminho))
            if metadata is None:
                metadata = ImageMetadata(
                    source_path=caminho,
                    backend=self.BACKEND_EXIFTOOL,
                    error="ExifTool não retornou metadados para o arquivo",
                )
            resultado.append(metadata)
        return resultado

    def _rodar_exiftool(self, lote: List[str]) -> List[ImageMetadata]:
        """Uma invocação do ExifTool sobre `lote`, em JSON.

        Os caminhos NÃO vão na linha de comando: no Windows eles chegariam
        ao ExifTool na página de código do sistema, e todo caminho com
        acento viraria "File not found" — o que atingiria praticamente todos
        os arquivos aqui, que vivem sob "Perícias". A forma suportada
        é a lista de argumentos em arquivo (`-@`), gravada em UTF-8 e lida
        com `-charset filename=UTF8`.
        """
        import tempfile

        def _falha(mensagem: str) -> List[ImageMetadata]:
            return [
                ImageMetadata(
                    source_path=c, backend=self.BACKEND_EXIFTOOL, error=mensagem,
                )
                for c in lote
            ]

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".args",
                delete=False, newline="\n",
            ) as arquivo_args:
                arquivo_args.write("\n".join(lote))
                caminho_args = arquivo_args.name
        except Exception as exc:
            return _falha(f"não foi possível preparar a lista de arquivos: {exc}")

        comando = [
            self._exiftool_path,
            "-json",        # saída estruturada
            "-a",           # inclui tags duplicadas (grupos distintos)
            "-G",           # prefixa o grupo (EXIF:, XMP:, IPTC:, …)
            "-n",           # valores crus, sem formatação de exibição
            "-charset", "filename=UTF8",
            "-@", caminho_args,
        ]
        try:
            saida = subprocess.run(
                comando, capture_output=True, timeout=300, check=False,
            )
        except Exception as exc:
            return _falha(f"falha ao executar ExifTool: {exc}")
        finally:
            try:
                Path(caminho_args).unlink(missing_ok=True)
            except Exception:
                pass  # temporário órfão não impede a leitura

        texto = (saida.stdout or b"").decode("utf-8", errors="replace").strip()
        if not texto:
            erro = (saida.stderr or b"").decode("utf-8", errors="replace").strip()
            return _falha(erro or "ExifTool não produziu saída")

        try:
            registros = json.loads(texto)
        except json.JSONDecodeError as exc:
            return _falha(f"saída do ExifTool ilegível: {exc}")

        saidas = []
        for registro in registros:
            origem = registro.get("SourceFile") or ""
            tags = {
                chave: valor for chave, valor in registro.items()
                if chave not in _TAGS_DESCARTAVEIS
            }
            # O ExifTool reporta erro POR ARQUIVO numa tag própria, em vez
            # de falhar a invocação inteira — um arquivo ilegível no meio
            # do lote não pode passar por "sem metadados".
            erro_arquivo = registro.get("ExifTool:Error") or registro.get("Error")
            saidas.append(ImageMetadata(
                source_path=origem,
                tags={} if erro_arquivo else tags,
                backend=self.BACKEND_EXIFTOOL,
                error=(
                    str(erro_arquivo) if erro_arquivo
                    else (None if tags else "arquivo sem metadados legíveis")
                ),
            ))
        return saidas

    # ── Backend Pillow ───────────────────────────────────────────────────

    def _extract_com_pillow(
        self,
        caminhos: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> List[ImageMetadata]:
        resultado = []
        total = len(caminhos)
        for indice, caminho in enumerate(caminhos):
            if progress_callback:
                progress_callback(indice, total, caminho)
            resultado.append(self._ler_com_pillow(caminho))
        if progress_callback:
            progress_callback(total, total, "")
        return resultado

    def _ler_com_pillow(self, caminho: str) -> ImageMetadata:
        try:
            from PIL import ExifTags, Image
        except ImportError as exc:  # pragma: no cover - Pillow é dependência
            return ImageMetadata(
                source_path=caminho, backend=self.BACKEND_PILLOW,
                error=f"Pillow indisponível: {exc}",
            )

        try:
            with Image.open(caminho) as imagem:
                bruto = imagem.getexif()
        except Exception as exc:
            return ImageMetadata(
                source_path=caminho, backend=self.BACKEND_PILLOW,
                error=f"leitura falhou: {exc}",
            )

        tags: Dict[str, Any] = {}
        for chave, valor in (bruto or {}).items():
            nome = ExifTags.TAGS.get(chave, str(chave))
            tags[f"EXIF:{nome}"] = self._simplificar(valor)

        try:
            gps = bruto.get_ifd(ExifTags.IFD.GPSInfo) if bruto else None
        except Exception:
            gps = None
        for chave, valor in (gps or {}).items():
            nome = ExifTags.GPSTAGS.get(chave, str(chave))
            tags[f"GPS:{nome}"] = self._simplificar(valor)

        return ImageMetadata(
            source_path=caminho,
            tags=tags,
            backend=self.BACKEND_PILLOW,
            error=None if tags else "arquivo sem metadados legíveis",
        )

    @staticmethod
    def _simplificar(valor: Any) -> Any:
        """Converte tipos do Pillow em algo serializável em JSON."""
        if isinstance(valor, bytes):
            return valor.decode("utf-8", errors="replace")
        if isinstance(valor, tuple):
            return [ExifExtractionService._simplificar(v) for v in valor]
        try:
            from PIL.TiffImagePlugin import IFDRational

            if isinstance(valor, IFDRational):
                return float(valor)
        except Exception:
            pass
        return valor

    @staticmethod
    def _normalizar(caminho: str) -> str:
        """Chave estável para casar a saída do ExifTool com a entrada.

        O ExifTool devolve o caminho com barras normais mesmo no Windows,
        e a comparação precisa ignorar isso e a caixa das letras.
        """
        return str(Path(caminho)).replace("\\", "/").lower()


# ── Promoção para colunas de dataset ─────────────────────────────────────

# Nome da coluna → tags que a alimentam, em ordem de preferência. Mais de
# um sinônimo porque o mesmo dado aparece com nomes distintos conforme o
# fabricante e o grupo (EXIF, XMP, QuickTime, Composite).
_MAPA_COLUNAS = {
    "capture_datetime_raw": ("DateTimeOriginal", "CreateDate", "DateTimeCreated"),
    "capture_timezone_offset": ("OffsetTimeOriginal", "OffsetTime", "TimeZone"),
    "camera_make": ("Make",),
    "camera_model": ("Model",),
    "camera_serial": ("SerialNumber", "BodySerialNumber", "InternalSerialNumber"),
    "lens_model": ("LensModel", "Lens", "LensID"),
    "iso": ("ISO", "ISOSpeedRatings", "PhotographicSensitivity"),
    "f_number": ("FNumber", "ApertureValue"),
    "exposure_time": ("ExposureTime", "ShutterSpeedValue"),
    "focal_length_mm": ("FocalLength",),
    "focal_length_35mm": ("FocalLengthIn35mmFormat", "FocalLength35efl"),
    "orientation": ("Orientation",),
    "image_width": ("ExifImageWidth", "ImageWidth"),
    "image_height": ("ExifImageHeight", "ImageHeight"),
    "software": ("Software", "ProcessingSoftware", "CreatorTool"),
    "gps_latitude": ("GPSLatitude",),
    "gps_longitude": ("GPSLongitude",),
    "gps_altitude": ("GPSAltitude",),
}

_COLUNAS_INT = {"iso", "orientation", "image_width", "image_height"}
_COLUNAS_FLOAT = {
    "f_number", "focal_length_mm", "focal_length_35mm",
    "gps_latitude", "gps_longitude", "gps_altitude",
}


def build_capture_columns(metadata: "ImageMetadata") -> Dict[str, Any]:
    """Traduz as tags lidas nas colunas de captura do dataset.

    Devolve só as colunas com valor — ausência vira coluna nula, não string
    vazia. `exif_json` carrega o dump íntegro: as demais colunas são apenas
    campos promovidos para consulta rápida, e nada é descartado.

    `capture_datetime` (normalizada) só é preenchida quando o arquivo
    declara o fuso EXPLICITAMENTE. Atribuir o fuso da máquina a uma foto de
    origem desconhecida seria inventar informação — e data/hora costuma ser
    exatamente o que está em disputa.
    """
    if not metadata.tags:
        return {}

    colunas: Dict[str, Any] = {}
    for nome, sinonimos in _MAPA_COLUNAS.items():
        valor = metadata.get_first(*sinonimos)
        if valor is None or valor == "":
            continue
        colunas[nome] = _converter(nome, valor)

    normalizada = _datetime_normalizado(
        colunas.get("capture_datetime_raw"),
        colunas.get("capture_timezone_offset"),
    )
    if normalizada:
        colunas["capture_datetime"] = normalizada

    colunas["exif_json"] = json.dumps(
        metadata.tags, ensure_ascii=False, default=str, separators=(",", ":"),
    )
    return colunas


def _converter(nome: str, valor: Any) -> Any:
    """Converte para o dtype declarado no catálogo, sem estourar."""
    if isinstance(valor, (list, tuple)):
        valor = valor[0] if valor else None
    if valor is None:
        return None
    try:
        if nome in _COLUNAS_INT:
            return int(float(valor))
        if nome in _COLUNAS_FLOAT:
            return float(valor)
    except (TypeError, ValueError):
        # Valor fora do formato esperado: preserva como texto em vez de
        # descartar — é informação do arquivo, e sumir com ela é pior.
        return str(valor)
    return str(valor).strip()


def _datetime_normalizado(bruto: Any, offset: Any) -> Optional[str]:
    """ISO-8601 com fuso, ou None quando o offset não foi declarado."""
    if not bruto or not offset:
        return None
    texto = str(bruto).strip()
    deslocamento = str(offset).strip()
    # EXIF grava "2026:09:24 10:30:00"; ISO quer "2026-09-24T10:30:00".
    if len(texto) >= 19 and texto[4] == ":" and texto[7] == ":":
        texto = texto[:4] + "-" + texto[5:7] + "-" + texto[8:]
    texto = texto.replace(" ", "T", 1)
    if deslocamento.upper() in {"Z", "UTC"}:
        deslocamento = "+00:00"
    if deslocamento and deslocamento[0] not in "+-":
        return None            # offset em formato não reconhecido
    if len(deslocamento) == 5 and ":" not in deslocamento:
        deslocamento = deslocamento[:3] + ":" + deslocamento[3:]
    candidato = f"{texto}{deslocamento}"
    try:
        from datetime import datetime

        datetime.fromisoformat(candidato)
    except ValueError:
        return None            # não afirma o que não consegue validar
    return candidato
