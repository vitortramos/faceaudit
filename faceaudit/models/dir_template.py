from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
import json

T = TypeVar("T", bound="DirTemplate")


# Built-in columns mapeadas direto pra atributos de FaceInfo.
# A chave é a canônica (igual ao atributo); o valor é o label PT-BR
# usado na UI (não persistido — só renderização).
BUILTIN_COLUMNS: Dict[str, str] = {
    "identity":  "Identidade",
    "gender":    "Gênero",
    "ethnicity": "Etnia",
    "origin":    "Origem",
}

# Token sentinela: marca uma parte do path que deve ser descartada.
# Mantido em minúsculas porque o adapter compara com `.lower().strip()`.
IGNORE_TOKEN = "ignorar"


@dataclass
class DirTemplate(JSONSerializableMixin):
    """
    Define as regras para extração de metadados a partir da estrutura de
    diretórios e nomenclatura de arquivos de um dataset.

    Cada nível de pasta e cada parte do nome de arquivo pode ser mapeado
    para uma coluna — built-in (identity/gender/ethnicity/origin, que
    setam direto o atributo correspondente em FaceInfo) ou custom
    (qualquer outro nome, que escreve em FaceInfo.extra_data[col_name]).

    Separadores são independentes para pasta e arquivo, permitindo que
    "BFW/m_001/img_03_frontal.jpg" use "_" só no filename, por exemplo.
    """
    name: str
    description: Optional[str] = None

    # Separadores independentes
    folder_separator: str = "_"
    use_folder_separator: bool = False
    filename_separator: str = "_"
    use_filename_separator: bool = False

    # Universo de colunas custom declaradas pelo usuário.
    # Built-ins (BUILTIN_COLUMNS) ficam implícitos no código —
    # não são serializados aqui.
    custom_columns: List[str] = field(default_factory=list)

    # Mapeamento por nível de pasta. Nível 0 é a primeira subpasta.
    folder_map: Dict[int, List[str]] = field(default_factory=dict)
    filename_map: Dict[int, str] = field(default_factory=dict)
    identity_from_parent: bool = False

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Garante a tipagem correta das chaves (de str para int) e migra
        templates antigos (separator/use_separator unificado) para o novo
        schema com pares independentes folder_/filename_."""
        if not data:
            return None

        data_copy = data.copy()

        # --- Migração de schema antigo (1 separador) → novo (2 pares) ---
        # Templates legados tinham `separator` + `use_separator` aplicados
        # tanto em folder quanto em filename. Replica para os dois novos
        # pares só se o novo schema não estiver presente.
        legacy_sep = data_copy.pop("separator", None)
        legacy_use = data_copy.pop("use_separator", None)
        if legacy_sep is not None:
            data_copy.setdefault("folder_separator", legacy_sep)
            data_copy.setdefault("filename_separator", legacy_sep)
        if legacy_use is not None:
            data_copy.setdefault("use_folder_separator", bool(legacy_use))
            data_copy.setdefault("use_filename_separator", bool(legacy_use))

        # Tipagem das chaves dos mapas (JSON sempre vira string)
        if "folder_map" in data_copy and isinstance(data_copy["folder_map"], dict):
            data_copy["folder_map"] = {
                int(k): v for k, v in data_copy["folder_map"].items()
            }

        if "filename_map" in data_copy and isinstance(data_copy["filename_map"], dict):
            data_copy["filename_map"] = {
                int(k): v for k, v in data_copy["filename_map"].items()
            }

        return cls(**data_copy)

    @classmethod
    def from_json(cls: Type[T], file_path: str) -> T:
        """Sobrescreve o Mixin para forçar a passagem pelo nosso from_dict seguro."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ───────────────────────── helpers de coluna ─────────────────────────

    def known_columns(self) -> List[str]:
        """Retorna a união (built-ins + custom) em ordem canônica.
        Usado pra popular combos de atributo na UI."""
        return list(BUILTIN_COLUMNS.keys()) + list(self.custom_columns)

    def label_for(self, column_key: str) -> str:
        """Retorna o label de exibição (PT-BR para built-ins, próprio nome
        para customs). Tolerante a None/desconhecido — retorna a chave."""
        if not column_key:
            return ""
        if column_key in BUILTIN_COLUMNS:
            return BUILTIN_COLUMNS[column_key]
        return column_key

    def is_builtin(self, column_key: str) -> bool:
        return column_key in BUILTIN_COLUMNS

    def column_usages(self, column_key: str) -> List[str]:
        """Lista, em PT-BR, todos os lugares onde uma coluna está mapeada.
        Usado pra bloquear remoção de coluna que ainda tem mapeamentos."""
        usages: List[str] = []
        for level, attrs in self.folder_map.items():
            for i, attr in enumerate(attrs):
                if attr == column_key:
                    usages.append(f"Pasta Nível {level} (item {i + 1})")
        for idx, attr in self.filename_map.items():
            if attr == column_key:
                usages.append(f"Nome do arquivo, Parte {idx}")
        return usages
