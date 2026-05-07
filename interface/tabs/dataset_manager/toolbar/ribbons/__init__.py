"""Ribbons da toolbar do Dataset Manager (PR Ribbon refactor).

3 ribbons (abas) que substituem a toolbar single-row anterior:
  - `ArquivoRibbon` — operações de I/O do dataset (Novo/Abrir/Salvar/...)
  - `DatasetRibbon` — operações sobre o conteúdo (Enrich/Merge/Filter/...)
  - `ImagensRibbon` — placeholder pro futuro tema Imagens

Cada ribbon expõe seus botões como atributos públicos (`self.btn_xyz`)
e a `ManagerToolbarView` cria atalhos (`view.btn_xyz`) que delegam aos
ribbons — preservando a API pública pra zero downstream change.
"""
