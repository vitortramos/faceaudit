"""DatasetLocalizationService — converte `FaceDataset` de paths
absolutos pra paths relativos, opcionalmente copiando imagens (PR L1c).

Stateless. Chamado pela UI do Dataset Manager (via L1d) e futuramente
pelo flow "Selar projeto" do Strategy Manager V2.

Modos:
  - REWRITE_RELATIVE: não copia; reescreve `image_path` como relativo
    a `target_dir`. Requer que TODAS as imagens já estejam dentro de
    `target_dir`. Falhas (imagens fora) ficam em `failures`.
  - COPY_ORIGINALS: copia imagens originais pra `target_dir/images/`
    e cria dataset novo com paths relativos. 3 estratégias de naming
    (PRESERVE/HASH/BY_IDENTITY).

Decisões de design:
  - **Sem atomicidade transacional** (escolha B do plano): cópia direta;
    em caso de erro, `skip_existing=True` permite reentrância natural.
    Roda de novo pra continuar de onde parou.
  - **Hash do conteúdo é estável**: cópia preserva bytes; `face.hash`
    continua válido sem recalcular.
  - **Não muta o dataset de entrada**: clona `FaceInfo`s com novos
    paths.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional, Tuple, Union

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.localization_result import LocalizationResult
from faceaudit.services.image_path_resolver import resolve_image_path


# Subpasta onde imagens são copiadas em COPY_ORIGINALS.
_COPY_SUBDIR = "images"


class DatasetLocalizationService:
    """Stateless. Use uma instância por chamada ou compartilhe.

    Lida exclusivamente com path management — REWRITE_RELATIVE,
    COPY_ORIGINALS, REBASE_PREFIX. Para alinhamento de faces (que
    é transformação de conteúdo), use o Editor de Pipeline de
    Imagens com `Origem = Face alinhada (112×112)` e zero steps.
    """

    def __init__(self):
        pass

    @staticmethod
    def detect_path_prefix(dataset: FaceDataset) -> Optional[str]:
        """Auto-detecta o prefixo comum dos `image_path` do dataset.

        Útil pro modo REBASE_PREFIX — quando todas as imagens vivem sob
        um diretório raiz na máquina original, este método infere esse
        diretório (longest common prefix por componente de path).

        Retorna None se o dataset estiver vazio, se os paths divergirem
        a partir do primeiro componente, ou se o dataset usa paths
        relativos (rebase só faz sentido em paths absolutos).
        """
        paths = [
            f.image_path for f in dataset.faces
            if f.image_path
        ]
        if not paths:
            return None

        # Normaliza separadores antes de comparar (Win '\\' → '/').
        norm = [p.replace("\\", "/") for p in paths]
        # Rebase só faz sentido em paths absolutos. Se um deles é
        # relativo, abortamos (caller deve usar outro modo).
        if any(not (p.startswith("/") or (len(p) > 2 and p[1] == ":")) for p in norm):
            return None

        # Componentes do primeiro path como referência.
        ref = norm[0].split("/")
        common: list[str] = []
        for i, comp in enumerate(ref):
            if all(len(p.split("/")) > i and p.split("/")[i] == comp for p in norm):
                common.append(comp)
            else:
                break

        if len(common) <= 1:  # só "/" ou drive letter — sem utilidade
            return None

        return "/".join(common)

    def localize(
        self,
        dataset: FaceDataset,
        target_dir: Union[str, Path],
        *,
        mode: LocalizationMode,
        naming: NamingStrategy = NamingStrategy.PRESERVE,
        skip_existing: bool = True,
        old_prefix: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> LocalizationResult:
        """Localiza um dataset.

        Args:
            dataset: dataset de entrada (não é mutado).
            target_dir: pasta âncora. Em COPY_ORIGINALS, imagens são
                copiadas pra `target_dir/images/`. Em REWRITE_RELATIVE,
                paths viram relativos a `target_dir`.
            mode: REWRITE_RELATIVE ou COPY_ORIGINALS.
            naming: estratégia de nome de arquivo (só relevante em
                COPY_ORIGINALS; ignorada em REWRITE_RELATIVE).
            skip_existing: idempotência — em COPY_ORIGINALS, pula faces
                cujo destino já existe com mesmo hash.
            progress_callback: chamado por face com `(idx, total, msg)`.

        Returns:
            `LocalizationResult` com o novo dataset + métricas.
        """
        target = Path(target_dir).resolve()
        # REBASE_PREFIX só reescreve paths — não cria diretório
        # destino. Se a pasta nem existe, é sinal de prefix errado
        # e mkdir mascararia o problema criando uma pasta vazia.
        if mode != LocalizationMode.REBASE_PREFIX:
            target.mkdir(parents=True, exist_ok=True)

        start = time.time()
        if mode == LocalizationMode.REWRITE_RELATIVE:
            result = self._rewrite_relative(dataset, target, progress_callback)
        elif mode == LocalizationMode.COPY_ORIGINALS:
            result = self._copy_originals(
                dataset, target, naming, skip_existing, progress_callback,
            )
        elif mode == LocalizationMode.REBASE_PREFIX:
            if not old_prefix:
                raise ValueError(
                    "REBASE_PREFIX exige `old_prefix` — use "
                    "DatasetLocalizationService.detect_path_prefix(dataset) "
                    "para auto-detectar."
                )
            if not target.exists():
                raise ValueError(
                    f"REBASE_PREFIX: pasta destino '{target}' não existe. "
                    f"Verifique se o caminho está correto e se as imagens "
                    f"realmente estão nesse local."
                )
            result = self._rebase_prefix(
                dataset, old_prefix, target, progress_callback,
            )
        else:
            raise ValueError(f"Modo não suportado: {mode}")

        # Re-empacota com duration_seconds preenchido. (LocalizationResult
        # é frozen, então recriamos o tuple-like.)
        from dataclasses import replace
        return replace(result, duration_seconds=time.time() - start)

    # ── REWRITE_RELATIVE ────────────────────────────────────────────────────

    def _rewrite_relative(
        self,
        dataset: FaceDataset,
        target_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> LocalizationResult:
        """Reescreve paths como relativos a `target_dir`. Sem cópia.

        Para cada face:
          1. Resolve o path absoluto atual.
          2. Tenta calcular path relativo a `target_dir`.
          3. Se a imagem está fora de `target_dir`, falha (path
             relativo seria `../../...`, quebrando portabilidade).
        """
        new_faces: List[FaceInfo] = []
        failures: List[Tuple[str, str]] = []
        n_processed = 0

        for idx, face in enumerate(dataset.faces):
            n_processed += 1
            if progress_callback:
                progress_callback(idx, len(dataset), face.image_path or "")

            try:
                abs_path = self._resolve_absolute(dataset, face)
            except Exception as e:
                failures.append((face.image_path or "<empty>", f"resolve: {e}"))
                continue

            if not abs_path.exists():
                failures.append((face.image_path, "arquivo não encontrado"))
                continue

            # Verifica que a imagem está dentro de target_dir
            try:
                rel_path = abs_path.relative_to(target_dir)
            except ValueError:
                failures.append((
                    str(abs_path),
                    f"fora de target_dir ({target_dir}); use COPY_ORIGINALS"
                    f" pra mover ou aponte target_dir pra um ancestral comum",
                ))
                continue

            # PR L1c.1 — `as_posix()` garante separador `/` em qualquer SO.
            # No Windows, `str(rel_path)` produziria `imgs\a.jpg` que quebra
            # ao abrir o parquet em macOS/Linux.
            new_face = self._clone_face_with_path(face, rel_path.as_posix())
            new_faces.append(new_face)

        # path_root="." porque os paths são relativos a target_dir direto.
        new_dataset = FaceDataset(
            name=dataset.name,
            description=dataset.description,
            faces=new_faces,
            column_specs=dict(dataset.column_specs) if dataset.column_specs else {},
            path_mode=PathMode.RELATIVE,
            path_root=".",
        )
        # Cuidado: hash global do dataset não é mais válido (paths mudaram).
        new_dataset.hash = None

        return LocalizationResult(
            dataset=new_dataset,
            target_dir=target_dir,
            mode=LocalizationMode.REWRITE_RELATIVE,
            naming=NamingStrategy.PRESERVE,  # placeholder — não relevante
            n_processed=n_processed,
            n_copied=0,
            n_skipped=0,
            n_failed=len(failures),
            failures=tuple(failures),
        )

    # ── REBASE_PREFIX ───────────────────────────────────────────────────────

    def _rebase_prefix(
        self,
        dataset: FaceDataset,
        old_prefix: str,
        new_prefix: Path,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> LocalizationResult:
        """Substitui `old_prefix` por `new_prefix` em todos os paths.

        Caso de uso típico: dataset criado em outra máquina (Linux/Win)
        onde as imagens estavam em `/home/colega/datasets/celeb/...` ou
        `C:\\Users\\alguem\\imgs\\...`, e nesta máquina a mesma estrutura
        existe em outro caminho (ex: `/Users/eu/data/celeb/...`).

        Para cada face:
          1. Verifica que `image_path` começa com `old_prefix`.
          2. Substitui o prefixo por `new_prefix` (str manipulação,
             sem touch no disco).
          3. Verifica que o path resultante existe — registra como
             failure caso contrário.

        Não copia imagens. O dataset resultante usa `path_mode=ABSOLUTE`
        com os novos paths abertos no destino.
        """
        # Normaliza prefixo antigo (slash unix; sem trailing slash).
        old_norm = old_prefix.replace("\\", "/").rstrip("/")
        new_str = str(new_prefix).rstrip("/").rstrip("\\")

        new_faces: List[FaceInfo] = []
        failures: List[Tuple[str, str]] = []
        n_processed = 0

        for idx, face in enumerate(dataset.faces):
            n_processed += 1
            if progress_callback:
                progress_callback(idx, len(dataset), face.image_path or "")

            old_path = (face.image_path or "").replace("\\", "/")
            if not old_path:
                failures.append(("<empty>", "image_path vazio"))
                continue
            if not old_path.startswith(old_norm + "/") and old_path != old_norm:
                failures.append((
                    old_path,
                    f"não começa com prefixo '{old_norm}'",
                ))
                continue

            # Calcula sufixo (parte depois do prefixo) e cola no novo.
            suffix = old_path[len(old_norm):].lstrip("/")
            new_path = f"{new_str}/{suffix}".replace("\\", "/")

            # Sanity check — o arquivo precisa existir no destino.
            if not Path(new_path).exists():
                # Mensagem inclui a substituição inteira pra o usuário
                # ver onde o rebase apontou. Caso típico de 100% de
                # falha: new_prefix selecionado não tem os arquivos
                # — só mostrar o new_path é insuficiente pra diagnóstico.
                failures.append((
                    new_path,
                    f"arquivo não encontrado no destino "
                    f"(substituição: '{old_norm}' → '{new_str}'; "
                    f"verifique se os arquivos estão realmente no destino)",
                ))
                continue

            new_faces.append(self._clone_face_with_path(face, new_path))

        # Mantém path_mode=ABSOLUTE (paths já estão prontos pra uso direto
        # no destino). Salvar-como-novo via UI rebase + save segue padrão
        # do dataset original.
        new_dataset = FaceDataset(
            name=dataset.name,
            description=dataset.description,
            faces=new_faces,
            column_specs=dict(dataset.column_specs) if dataset.column_specs else {},
            path_mode=PathMode.ABSOLUTE,
            path_root=None,
        )
        new_dataset.hash = None

        return LocalizationResult(
            dataset=new_dataset,
            target_dir=Path(new_str),
            mode=LocalizationMode.REBASE_PREFIX,
            naming=NamingStrategy.PRESERVE,
            n_processed=n_processed,
            n_copied=0,
            n_skipped=0,
            n_failed=len(failures),
            failures=tuple(failures),
        )

    # ── COPY_ORIGINALS ──────────────────────────────────────────────────────

    def _copy_originals(
        self,
        dataset: FaceDataset,
        target_dir: Path,
        naming: NamingStrategy,
        skip_existing: bool,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> LocalizationResult:
        """Copia imagens pra `target_dir/images/` com naming dinâmico.

        Reentrante: `skip_existing=True` pula faces cujo destino já
        tem o mesmo hash que a fonte (idempotência).
        """
        images_dir = target_dir / _COPY_SUBDIR
        images_dir.mkdir(parents=True, exist_ok=True)

        new_faces: List[FaceInfo] = []
        failures: List[Tuple[str, str]] = []
        n_copied = 0
        n_skipped = 0
        n_processed = 0

        # Track de nomes já usados nessa rodada pra resolver colisões
        # PRESERVE/BY_IDENTITY (HASH é único por construção).
        used_names: set[str] = set()

        for idx, face in enumerate(dataset.faces):
            n_processed += 1
            if progress_callback:
                progress_callback(idx, len(dataset), face.image_path or "")

            try:
                src = self._resolve_absolute(dataset, face)
            except Exception as e:
                failures.append((face.image_path or "<empty>", f"resolve: {e}"))
                continue

            if not src.exists():
                failures.append((str(src), "arquivo não encontrado"))
                continue

            # Determina o nome de destino conforme naming
            try:
                rel_dest = self._build_dest_name(face, src, naming, used_names)
            except Exception as e:
                failures.append((str(src), f"naming: {e}"))
                continue

            dst = images_dir / rel_dest
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Idempotência: pula se já existe e hash bate
            if skip_existing and dst.exists():
                if self._files_have_same_hash(src, dst):
                    n_skipped += 1
                    used_names.add(rel_dest)
                    new_faces.append(self._clone_face_with_path(face, rel_dest))
                    continue
                # Existe mas com conteúdo diferente — desambigua adicionando
                # sufixo (idem PRESERVE collision logic).
                rel_dest = self._next_unique_name(rel_dest, used_names, images_dir)
                dst = images_dir / rel_dest
                dst.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(src, dst)
                n_copied += 1
                used_names.add(rel_dest)
                new_faces.append(self._clone_face_with_path(face, rel_dest))
            except Exception as e:
                failures.append((str(src), f"copy: {e}"))

        new_dataset = FaceDataset(
            name=dataset.name,
            description=dataset.description,
            faces=new_faces,
            column_specs=dict(dataset.column_specs) if dataset.column_specs else {},
            path_mode=PathMode.RELATIVE,
            path_root=_COPY_SUBDIR,
        )
        new_dataset.hash = None

        return LocalizationResult(
            dataset=new_dataset,
            target_dir=target_dir,
            mode=LocalizationMode.COPY_ORIGINALS,
            naming=naming,
            n_processed=n_processed,
            n_copied=n_copied,
            n_skipped=n_skipped,
            n_failed=len(failures),
            failures=tuple(failures),
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_absolute(self, dataset: FaceDataset, face: FaceInfo) -> Path:
        """Devolve path absoluto, respeitando o `path_mode` do dataset."""
        return resolve_image_path(
            face,
            parquet_dir=dataset._parquet_dir,
            path_mode=dataset.path_mode,
            path_root=dataset.path_root,
        )

    def _build_dest_name(
        self,
        face: FaceInfo,
        src: Path,
        naming: NamingStrategy,
        used_names: set[str],
    ) -> str:
        """Determina o nome relativo (dentro de `images_dir`) do
        arquivo de destino."""
        if naming == NamingStrategy.HASH:
            ext = src.suffix or ".jpg"
            digest = face.hash or self._hash_file(src)
            stem = digest[:16]
            return f"{stem}{ext}"

        if naming == NamingStrategy.BY_IDENTITY:
            ident = (face.identity or "_unknown").strip() or "_unknown"
            # Sanitiza identidade pra ser path-safe (sem `/` ou `\`).
            ident_safe = ident.replace("/", "_").replace("\\", "_")
            base = face.file_name or src.name
            candidate = f"{ident_safe}/{base}"
            return self._disambiguate(candidate, used_names)

        # PRESERVE
        base = face.file_name or src.name
        return self._disambiguate(base, used_names)

    @staticmethod
    def _disambiguate(name: str, used: set[str]) -> str:
        """Se `name` já está em `used`, gera `<stem>_1<ext>`,
        `<stem>_2<ext>` ... até achar um livre.

        PR L1c.1 — usa `PurePosixPath` em vez de `Path` pra que os
        componentes sempre sejam separados por `/`, independente do
        SO. `str(Path("a/b").parent)` em Windows retorna `"a"`, mas
        com paths de 2 níveis (`"a/b/c"`) retornaria `"a\\b"` —
        misturando separadores no resultado final.
        """
        if name not in used:
            return name
        p = PurePosixPath(name)
        parent = p.parent.as_posix() if str(p.parent) != "." else ""
        stem = p.stem
        ext = p.suffix
        i = 1
        while True:
            candidate_name = f"{stem}_{i}{ext}"
            candidate = f"{parent}/{candidate_name}" if parent else candidate_name
            if candidate not in used:
                return candidate
            i += 1

    def _next_unique_name(
        self, name: str, used: set[str], images_dir: Path,
    ) -> str:
        """Idem `_disambiguate` mas também checa o filesystem (caso
        rodadas anteriores tenham deixado arquivos)."""
        p = PurePosixPath(name)
        parent = p.parent.as_posix() if str(p.parent) != "." else ""
        stem = p.stem
        ext = p.suffix
        i = 1
        while True:
            candidate_name = f"{stem}_{i}{ext}"
            candidate = f"{parent}/{candidate_name}" if parent else candidate_name
            if candidate not in used and not (images_dir / candidate).exists():
                return candidate
            i += 1

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 do conteúdo. Usado quando NAMING=HASH e a face não
        tem `hash` populado."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _files_have_same_hash(self, a: Path, b: Path) -> bool:
        """Compara conteúdo via SHA-256. Usado pra idempotência."""
        try:
            return self._hash_file(a) == self._hash_file(b)
        except OSError:
            return False

    @staticmethod
    def _clone_face_with_path(face: FaceInfo, new_path: str) -> FaceInfo:
        """Clone shallow do FaceInfo com novo `image_path`. file_name
        é recalculado pelo `__post_init__` quando vazio; preservamos
        explicitamente o file_name original pra evitar mudar a UX.

        PR L1c.1 — defesa final: normaliza pra POSIX antes de atribuir.
        `copy.copy` não dispara `__post_init__`, então a normalização
        precisa ser explícita aqui.
        """
        clone = copy.copy(face)
        clone.image_path = new_path.replace("\\", "/")
        # Não mexemos no file_name nem no hash — ambos continuam válidos
        # (file_name preserva a legibilidade original; hash é do conteúdo,
        # cópia preserva bytes).
        return clone
