"""Storage da nova arquitetura.

`ArtifactStorageProtocol` é o contrato; `LocalProjectArtifactStorage` e
`GlobalLibraryStorage` são as implementações concretas. O adaptador
legado `ArtifactStorageService` foi movido para `Código Velho/`.
"""
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol

__all__ = ["ArtifactStorageProtocol"]
