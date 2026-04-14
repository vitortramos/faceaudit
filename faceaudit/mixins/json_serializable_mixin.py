import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Type, TypeVar, Union, Optional, Any
from enum import Enum

# Tipo genérico para permitir que os métodos retornem a instância da classe que herda
T = TypeVar("T")

class JSONSerializableMixin:
    """
    Mixin que adiciona capacidades de serialização JSON recursiva aos DTOs.
    Permite salvar configurações em disco e reconstruí-las com tipagem forte.
    Faz uso da padronização arquitetural 'to_str()' presente nos Enums do projeto
    e varre estruturas aninhadas (dicts/lists) profundamente.
    """

    def to_dict(self) -> dict:
        """
        Converte o DTO (dataclass) para um dicionário, garantindo que Enums em 
        qualquer profundidade (mesmo dentro de dicts genéricos) chamem seu to_str().
        """
        # 1. Extrai o dicionário base (que deixa enums aninhados intactos)
        raw_dict = asdict(self)

        # 2. Varredura profunda para aplicar o padrão to_str() da arquitetura
        def _deep_sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _deep_sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_deep_sanitize(v) for v in obj]
            elif isinstance(obj, Enum):
                # Usa o contrato arquitetural do projeto
                if hasattr(obj, 'to_str') and callable(obj.to_str):
                    return obj.to_str()
                # Fallback de segurança 
                return obj.name
            return obj

        # 3. Retorna o dicionário completamente higienizado
        return _deep_sanitize(raw_dict)

    def to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Converte o DTO para uma string JSON, tratando Enums adequadamente.
        Se file_path for fornecido, também salva no disco.
        """
        
        # Agora delegamos a geração do dicionário para o to_dict() higienizado
        data = self.to_dict()
        json_str = json.dumps(data, indent=4, ensure_ascii=False)

        if file_path:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding='utf-8')
        
        return json_str

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o objeto a partir de um dicionário.
        Nota: Este método DEVE ser sobrescrito em DTOs que possuam 
        objetos aninhados ou Enums para garantir a conversão de string -> Objeto 
        (usando o respectivo from_str()).
        """
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], data_input: Union[str, Path]) -> T:
        """
        Reconstrói o objeto a partir de uma string JSON ou caminho de arquivo.
        """
        path = Path(str(data_input))
        if path.exists() and path.is_file():
            content = path.read_text(encoding='utf-8')
            data = json.loads(content)
        else:
            data = json.loads(data_input)
        
        return cls.from_dict(data)