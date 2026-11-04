# 03 — Carregando e modificando caminhos

Este capítulo cobre dois fluxos complementares:

- **Carregar** um dataset `.parquet` já existente (criado por você
  anteriormente, recebido de um colega, ou copiado da biblioteca
  global).
- **Modificar Caminhos** — gerenciar como o dataset aponta pras
  imagens (`image_path`) em disco: desde uma correção leve da subpasta
  relativa (`path_root`) até migrações pesadas que reescrevem todos os
  caminhos ou copiam as imagens. Esse é o menu de soluções para o
  problema clássico do *“abri o parquet aqui, mas as miniaturas
  aparecem vazias”*.

> ℹ️ Em versões anteriores este botão se chamava **Localizar**. O nome
> mudou para **Modificar Caminhos** porque a janela hoje faz mais do
> que “localizar” imagens perdidas — ela também corrige o `path_root`
> de datasets relativos sem mover nenhum arquivo.

## Sumário do capítulo

1. [Abrindo um `.parquet`](#abrindo-um-parquet)
2. [Anatomia dos caminhos de imagem](#anatomia-dos-caminhos)
3. [Quando preciso modificar os caminhos?](#quando-preciso-modificar-os-caminhos)
4. [O diálogo *Modificar Caminhos*](#o-diálogo-modificar-caminhos)
5. [Operação — Corrigir a subpasta (`path_root`)](#operação--corrigir-a-subpasta-path_root)
6. [Operação — REBASE_PREFIX (substituir prefixo)](#operação--rebase_prefix)
7. [Operação — REWRITE_RELATIVE (caminhos relativos)](#operação--rewrite_relative)
8. [Operação — COPY_ORIGINALS (empacotar pra portabilidade)](#operação--copy_originals)
9. [Estratégia de nomeação (NamingStrategy)](#estratégia-de-nomeação)
10. [`skip_existing` — idempotência](#skip_existing--idempotência)
11. [Cenários práticos com antes/depois](#cenários-práticos)
12. [Erros comuns e diagnóstico](#erros-comuns)
13. [O que **nunca** muda](#o-que-nunca-muda)

---

## Abrindo um `.parquet`

A operação básica para trabalhar com um dataset existente é o botão
📂 **Abrir** da aba *Arquivo*.

O diálogo nativo do SO abre na pasta de datasets da biblioteca global
(`{paths.library}/datasets` do `config.json`). Você pode navegar
livremente; só são listados arquivos `.parquet`.

Selecionado o arquivo:

1. O FaceAudit dispara um *worker* assíncrono (`LoadDatasetWorker`)
   pra ler o parquet em thread separada — UI continua responsiva.
2. Ao concluir:
   - A tabela é populada com as linhas
   - O painel de detalhes fica em *"Selecione uma face"* até você
     clicar numa linha
   - A barra de status mostra nome, número de imagens e de identidades
     (ver [capítulo 01](01_visao_geral.md#a-barra-de-status))
   - O *Audit Log* registra a operação `OPEN` com timestamp e SHA-256
3. Em caso de erro (parquet corrompido, formato inválido, falta de
   colunas obrigatórias), um diálogo `QMessageBox` mostra a mensagem.

> 💡 Você também pode abrir um dataset **a partir do Gerenciador de
> Estratégias** — o botão *"📂 Abrir no Gerenciador de Datasets"* no
> painel de propriedades de um recurso do tipo `DATASET` troca de aba e
> carrega o parquet diretamente, sem passar pelo seletor de arquivo.

## Anatomia dos caminhos

Cada linha do dataset tem uma coluna `image_path` apontando para o
arquivo de imagem **no disco**. Existem dois **modos de caminho**:

| `path_mode` | Como o `image_path` é interpretado | Exemplo |
|---|---|---|
| `ABSOLUTE` | Caminho cru, exatamente como está | `/home/alice/imgs/foto_01.png` |
| `RELATIVE` | Resolvido em relação a `path_root` (atributo do dataset) | `path_root="images"`, `image_path="foto_01.png"` → `<pasta_parquet>/images/foto_01.png` |

Quando você abre uma imagem, o FaceAudit resolve o caminho
conforme o modo e tenta carregar o arquivo. Se não existe, a
miniatura fica vazia e o painel de detalhes mostra *"Imagem não
encontrada"*.

> ⚠️ Os modos são **propriedade do dataset**, não da máquina — um
> parquet `RELATIVE` é portável por construção, basta carregar o
> `.parquet` junto com a pasta de imagens.

O `path_root` é a **subpasta** (relativa à pasta do parquet) onde as
imagens moram. Se ele estiver errado — por exemplo, você copiou o
parquet e as imagens para uma estrutura de pastas diferente — as
miniaturas ficam em branco mesmo estando o arquivo bem ali no disco.
Corrigir esse ponteiro é a operação mais leve da janela (veja
[Corrigir a subpasta](#operação--corrigir-a-subpasta-path_root)).

## Quando preciso modificar os caminhos?

A operação *Modificar Caminhos* (botão 🗂 **Modificar Caminhos** na aba
*Dataset*) é necessária em qualquer situação onde **os caminhos no
parquet não batem com a localização das imagens no disco atual**. Os
cenários mais comuns:

| Situação | Sintoma | Operação recomendada |
|---|---|---|
| Dataset **relativo** com `path_root` errado (imagens já estão no lugar, só o ponteiro está errado) | Miniaturas em branco, mas os arquivos existem numa subpasta próxima | **Corrigir a subpasta (path_root)** |
| O parquet veio de outra máquina (colega, backup) | Miniaturas em branco; `image_path` aponta para `/home/colega/...` ou `C:\Users\colega\...` | **REBASE_PREFIX** |
| O parquet aponta para um caminho absoluto local que mudou de lugar | Você reorganizou pastas e mudou o local das imagens | **REBASE_PREFIX** ou **REWRITE_RELATIVE** |
| Você quer empacotar o dataset para arquivar ou transferir | Quer um zip autocontido com parquet + imagens | **COPY_ORIGINALS** |
| O parquet ficou com caminhos absolutos legados e você quer torná-lo portável | Funciona aqui, mas não em outra máquina | **REWRITE_RELATIVE** |

## O diálogo *Modificar Caminhos*

Ao clicar em 🗂 **Modificar Caminhos** abre-se o diálogo
*Modificar Caminhos — {nome do dataset}* (660×600):

![Diálogo Modificar Caminhos com combo de operação, sub-form de subpasta e destino](assets/31_localizar.png)

| Seção | O que controla |
|---|---|
| **Texto explicativo** (topo) | Resumo do propósito da operação |
| **Estado atual** | Read-only: `Relativo · Subpasta (path_root): …` ou `Absoluto` |
| **Operação** | Combobox com as operações disponíveis para o `path_mode` atual |
| **Corrigir subpasta (path_root)** | Visível apenas na operação *Corrigir subpasta*; campo `path_root` + **📁 Procurar…** + **✔ Validar** |
| **Prefixo antigo (a ser substituído)** | Visível apenas em REBASE_PREFIX; auto-preenchido com `detect_path_prefix()` |
| **Forma da imagem** | Visível apenas em COPY_ORIGINALS — **Copiar como:** *Foto original*, *Face recortada (com margem)* (+ **Margem do recorte:**) ou *Face alinhada (112×112)* |
| **Estratégia de nome** | Habilitada apenas em COPY_ORIGINALS |
| **Pasta destino** | Sugere `<parquet_dir>/<nome>__localized/` em COPY/REWRITE; em REBASE muda o rótulo para *Prefixo novo (pasta nesta máquina)* |
| **Pular arquivos que já existem com o mesmo conteúdo** | Habilitado apenas em COPY; idempotência (`skip_existing`) |

No combo, as operações aparecem com rótulos em português; este
capítulo usa os nomes internos:

| Rótulo no combo | Nome neste capítulo |
|---|---|
| *Corrigir subpasta (path_root) — sem mover arquivos* | Corrigir subpasta |
| *Reescrever caminhos (sem cópia)* | REWRITE_RELATIVE |
| *Copiar imagens originais* | COPY_ORIGINALS |
| *Atualizar prefixo de caminho (sem cópia)* | REBASE_PREFIX |

A escolha da operação **dinamicamente reconfigura** quais campos ficam
visíveis ou ativos, atualiza o texto de ajuda, e alterna o rótulo do
botão de confirmação entre **✔ Aplicar path_root** (correção leve) e
**🗂 Modificar caminhos** (migração pesada).

> 💡 O combo é **filtrado pelo `path_mode`**: em datasets **relativos** a
> operação *Corrigir subpasta* aparece primeiro e vem selecionada por
> padrão; em datasets **absolutos** ela nem é oferecida (não há
> `path_root` em modo absoluto) — o padrão vira COPY_ORIGINALS.

## Operação — Corrigir a subpasta (path_root)

**O que faz:** Atualiza **apenas** o ponteiro `path_root` — a subpasta
(relativa à pasta do parquet) onde o dataset procura as imagens.
**Nenhum arquivo é movido ou copiado**: só o metadado muda. Disponível
**exclusivamente** em datasets `RELATIVE`.

**Quando usar:** As imagens já estão no disco, mas o `path_root`
gravado no parquet aponta para a subpasta errada. Caso típico: você
copiou o `.parquet` e a pasta de imagens juntos, mas a estrutura de
subpastas ficou diferente da original (ex: as imagens estavam em
`crops/` e agora estão em `images/`).

### O sub-form *Corrigir subpasta*

| Controle | Função |
|---|---|
| Campo de texto | O novo `path_root` (subpasta relativa; deixe **vazio** para apontar à raiz da pasta do parquet) |
| **📁 Procurar…** | Abre um seletor de pasta ancorado na pasta do parquet e converte a escolha em caminho relativo automaticamente. Recusa pastas **fora** da pasta do parquet |
| **✔ Validar** | Resolve uma face de exemplo com o `path_root` candidato e confirma se o arquivo **existe** em disco — verde ✅ (existe) ou vermelho ⚠ (não existe). Não fecha o diálogo |

> ⚠️ **Validar confirma que os arquivos existem, não que são as
> imagens certas.** Um `path_root` que aponta para uma pasta *válida
> mas errada* (outra pasta de imagens quaisquer) passa na validação de
> existência. A conferência final é visual, na galeria.

### O que muda

| Item | Estado |
|---|---|
| `path_root` (metadado do dataset) | ✏️ Atualizado para a nova subpasta |
| `path_mode` | 🔒 Permanece `RELATIVE` |
| `image_path` de cada face | 🔒 Intacto (a *resolução* muda, não o valor gravado) |
| Arquivos no disco | 🔒 Intactos — nada é movido nem copiado |
| Hashes, embeddings, identidades | 🔒 Inalterados |

### Aplicando

Ao clicar **✔ Aplicar path_root**, a mudança é feita **em memória** e o
dataset fica marcado como *não salvo* (`★` na barra de status). As
miniaturas re-resolvem imediatamente. Use **Salvar** para gravar o novo
`path_root` no parquet.

> 💡 Se a validação falhar, o FaceAudit ainda oferece **aplicar mesmo
> assim** — mas confirme o resultado na galeria antes de salvar. Em
> dataset ainda não salvo em disco a validação fica indisponível e a
> subpasta é aplicada sem conferência.

## Operação — REBASE_PREFIX

**O que faz:** Substitui um **prefixo de caminho** em todos os
`image_path` por outro. Operação puramente em texto — **nada é
copiado**.

**Quando usar:** O caso mais comum em datasets absolutos. Você recebeu
um parquet com caminhos tipo `/home/colega/imgs/foto.jpg` e suas
imagens estão em `/Users/voce/dados/imgs/foto.jpg`. As pastas finais
batem exatamente — só o prefixo é diferente.

### Detecção automática do prefixo antigo

Quando você abre o diálogo, o sistema chama `detect_path_prefix()`
sobre o dataset:

1. Coleta todos os `image_path` não vazios
2. Normaliza separadores (`\` → `/`)
3. Valida que todos são absolutos
4. Acha o **maior prefixo comum** componente-a-componente
5. Retorna o prefixo (ou `None` se houver divergência na raiz, ou
   se algum caminho for relativo)

O campo **Prefixo Antigo** é pré-preenchido com esse valor. Você
pode editar livremente — útil quando os caminhos têm sub-prefixos
diferentes que você quer migrar separadamente.

### O que muda

| Item | Estado |
|---|---|
| `image_path` (no parquet) | ✏️ Atualizado: prefixo antigo trocado pelo novo |
| `path_mode` | 🔒 Permanece `ABSOLUTE` |
| Arquivos no disco | 🔒 Intactos — nada é movido ou copiado |
| Hashes, embeddings, identidades | 🔒 Inalterados |

### Validação

O serviço **verifica** que cada novo caminho realmente existe no
disco — qualquer caminho inexistente entra na lista de `failures`
do `LocalizationResult` e é reportado na UI (as cinco primeiras
falhas aparecem na mensagem final). Se o prefixo novo nem existe como
pasta, a operação é recusada antes de começar.

## Operação — REWRITE_RELATIVE

**O que faz:** Recalcula cada `image_path` como **relativo** à
*Pasta destino*. **Nada é copiado**.

**Quando usar:** As imagens já estão dentro da pasta destino e você
só quer converter o dataset de absoluto para relativo (tornando-o
portátil).

### O que muda

| Item | Estado |
|---|---|
| `image_path` | ✏️ Agora relativo (ex: `"celeb/alice.jpg"`) |
| `path_mode` | ✏️ Vira `RELATIVE` |
| `path_root` | ✏️ Aponta para a pasta destino |
| Arquivos no disco | 🔒 Intactos |

### Restrição importante

Se alguma imagem estiver **fora** da pasta destino, o caminho
relativo resultante teria `../../..` no início — quebra a
portabilidade. Nesse caso a face entra em `failures` e a UI exibe
um aviso sugerindo usar COPY_ORIGINALS ou mover o destino para um
ancestral comum.

## Operação — COPY_ORIGINALS

**O que faz:** **Copia** os arquivos de imagem para
`<pasta_destino>/<nome do dataset>/images/` e gera um novo dataset com
caminhos relativos. Cria uma cópia autocontida do dataset. Como cada
dataset ganha a **sua própria pasta** (`<pasta_destino>/<nome>/`), você
pode copiar **vários datasets para a MESMA pasta destino** sem misturar
as imagens. O parquet tem de ser gravado **nessa pasta** — o
**Salvar Como** seguinte já abre nela.

Em **Forma da imagem** você escolhe o que copiar: a *Foto original* (cópia
byte a byte), a *Face recortada (com margem)* ou a *Face alinhada
(112×112)*. As duas últimas detectam a face em cada imagem (exigem o
ForensicFace) e gravam pixel novo; imagens em que nenhuma face for
detectada ficam **de fora** do dataset resultante.

**Quando usar:** Empacotar o dataset para arquivamento em mídia
externa, transferência por USB/cloud, ou snapshot perícia
*pendrive paradigm*.

### O que muda

| Item | Estado |
|---|---|
| `image_path` | ✏️ Caminho relativo na nova estrutura |
| `path_mode` | ✏️ Vira `RELATIVE` |
| `path_root` | ✏️ `"images"` |
| Arquivos no disco | ✏️ **Copiados** para `<destino>/<nome>/images/` |
| `file_name` | 🔒 Preservado (a menos que colisão force sufixo) |
| Hashes | 🔒 Inalterados em *Foto original* (cópia byte-a-byte preserva conteúdo) |
| Tudo mais | 🔒 Inalterado |

> 💡 Após COPY_ORIGINALS e o **Salvar Como** dentro de
> `<pasta_destino>/<nome>/`, comprima essa pasta num zip e você tem um
> dataset **100 % portátil** — funciona em qualquer máquina com
> FaceAudit instalado.

## Quando faces falham

Uma face cuja imagem não foi encontrada ou não pôde ser copiada entra
na lista de falhas e, nas operações que copiam ou geram imagens, **sai
do resultado** — não há imagem a que apontar. Isso nunca é aplicado
em silêncio:

| Situação | O que acontece |
|---|---|
| Nenhuma falha | O resultado substitui o dataset aberto e o resumo é exibido |
| Algumas falhas | O FaceAudit lista as primeiras falhas e **pergunta** se aplica o resultado com as faces que sobraram. *Cancelar* mantém o dataset aberto como estava |
| **Nenhuma face sobrou** | O resultado é **recusado**: o dataset aberto continua como estava, nada é marcado para gravar e as falhas são exibidas. Um dataset sem faces seria gravado como um parquet vazio, que some dos seletores do Gerenciador de Estratégias sem explicação |

As imagens que chegaram a ser copiadas para a pasta destino continuam
lá; rodar de novo com *Pular arquivos que já existem com o mesmo conteúdo* aproveita a
cópia.

## Estratégia de nomeação

Disponível **somente em COPY_ORIGINALS** — controla como os arquivos
copiados são nomeados:

| Estratégia | Como nomeia | Quando usar |
|---|---|---|
| **PRESERVE** *(padrão)* | Mantém o nome original. Em colisão (dois arquivos diferentes com mesmo nome), adiciona sufixo `_1`, `_2`, …. | Datasets pequenos a médios; melhor pra legibilidade |
| **HASH** | Renomeia para `<sha256[:16]>.<ext>` (ex: `a3f7e2c9b1d4f0ea.jpg`) | Datasets grandes com muitas colisões potenciais; uniqueness garantido por construção |
| **BY_IDENTITY** | Cria subpasta por identidade: `<identity>/<nome_original>` | Datasets com identidades bem definidas, quando você quer browse manual; sufixo `_N` em colisões dentro da pasta |

> ⚠️ A estratégia `BY_IDENTITY` sanitiza caracteres problemáticos —
> `/` e `\` viram `_` — antes de criar o nome da subpasta. Identidades
> com caracteres especiais podem ficar ligeiramente diferentes.

## `skip_existing` — idempotência

Disponível **somente em COPY_ORIGINALS** — quando marcado, o
operação fica idempotente:

| Estado do destino | Skip ON | Skip OFF |
|---|---|---|
| Não existe | Copia normalmente | Copia normalmente |
| Existe + **hash bate** | ✅ Pula (incrementa `n_skipped`) | ❌ Sobrescreve mesmo conteúdo |
| Existe + **hash difere** | ❌ Trata como colisão (gera nome novo) | ⚠️ Sobrescreve o arquivo existente |

> 💡 **Sempre deixe ligado** em re-execuções — em datasets grandes
> economiza minutos ou horas (a verificação SHA-256 é muito mais
> rápida que a cópia).

## Cenários práticos

### Cenário 0 — Corrigindo a subpasta de um dataset relativo

**Antes** (parquet relativo copiado, mas as imagens ficaram noutra subpasta):

```
parquet   : /Users/me/caso/dataset.parquet
imagens   : /Users/me/caso/images/*.png   (estão AQUI)
path_mode : RELATIVE
path_root : "crops"                        ← ponteiro ERRADO → miniaturas vazias
```

**Ação:**

1. 🗂 **Modificar Caminhos**
2. Operação: **Corrigir subpasta (path_root)** *(já vem selecionada)*
3. Campo `path_root`: digite `images` ou use **📁 Procurar…**
4. **✔ Validar** → verde ✅ (o arquivo de exemplo existe)
5. **✔ Aplicar path_root** → miniaturas re-resolvem; `★` na barra de status
6. **Salvar**

**Depois:**

```
path_root : "images"   ← corrigido; nenhum arquivo movido
```

### Cenário A — Movendo um projeto de `/Users/alice` para `/Users/bob`

**Antes** (parquet recebido da Alice):

```
image_path = /Users/alice/data/celeb/alice_001.jpg
image_path = /Users/alice/data/celeb/bob_001.jpg
path_mode  = ABSOLUTE
```

**Ação:**

1. 🗂 **Modificar Caminhos**
2. Operação: **REBASE_PREFIX** (*Atualizar prefixo de caminho*)
3. Prefixo antigo: `/Users/alice/data/celeb` *(auto-detectado)*
4. Prefixo novo: `/Users/bob/projects/celeb` *(onde as imagens estão na sua máquina)*
5. **🗂 Modificar caminhos**

**Depois:**

```
image_path = /Users/bob/projects/celeb/alice_001.jpg
image_path = /Users/bob/projects/celeb/bob_001.jpg
path_mode  = ABSOLUTE  ← inalterado
```

Disco: nenhum arquivo tocado. Em REBASE_PREFIX o arquivo aberto
continua sendo o mesmo: **Salvar** sobrescreve o parquet original e
**Salvar Como** grava uma cópia.

### Cenário B — Parquet do Windows para macOS

**Antes** (colega te mandou de Windows):

```
image_path = C:\Users\colega\datasets\faces\foto_01.jpg
image_path = C:\Users\colega\datasets\faces\foto_02.jpg
```

**Ação:**

1. 🗂 **Modificar Caminhos**
2. Operação: **REBASE_PREFIX**
3. Prefixo antigo: `C:/Users/colega/datasets/faces` *(separadores normalizados)*
4. Prefixo novo: `/Users/voce/imagens/faces`
5. **🗂 Modificar caminhos**

### Cenário C — Empacotar dataset para arquivamento

**Antes:**

```
parquet  : /Users/me/projects/old_case/case_2024.parquet
images   : /Users/me/projects/old_case/imgs/*.jpg  (1500 arquivos)
image_path: /Users/me/projects/old_case/imgs/<filename>  (ABSOLUTE)
```

**Ação:**

1. 🗂 **Modificar Caminhos**
2. Operação: **COPY_ORIGINALS** (*Copiar imagens originais*)
3. Copiar como: **Foto original**
4. Estratégia: **PRESERVE**
5. Pasta destino: `/Volumes/HD_Externo/arquivo_2024/`
6. Pular arquivos que já existem: ☑️
7. **🗂 Modificar caminhos**
8. **Salvar Como** (já abre em `/Volumes/HD_Externo/arquivo_2024/case_2024/`)

**Depois:**

```
parquet  : /Users/me/projects/old_case/case_2024.parquet  ← original intacto
novo parquet:
  Salvar Como → /Volumes/HD_Externo/arquivo_2024/case_2024/case_2024.parquet

imagens copiadas:
  /Volumes/HD_Externo/arquivo_2024/case_2024/images/foto_01.jpg
  /Volumes/HD_Externo/arquivo_2024/case_2024/images/foto_02.jpg
  ...

image_path: foto_01.jpg, foto_02.jpg, ...  (RELATIVE)
path_root : "images"
```

Agora `/Volumes/HD_Externo/arquivo_2024/case_2024/` é uma pasta
**autocontida**: zipe, copie para outro disco, mande pro colega —
funciona em qualquer máquina com FaceAudit.

### Cenário D — Limpando caminhos absolutos legados

**Antes:**

```
parquet  : ~/projects/active/faces.parquet
images   : ~/projects/active/imgs/*  (já no lugar certo)
image_path: /home/user/projects/active/imgs/<file>  (ABSOLUTE)
```

**Ação:**

1. 🗂 **Modificar Caminhos**
2. Operação: **REWRITE_RELATIVE** (*Reescrever caminhos*)
3. Pasta destino: `~/projects/active/`
4. **🗂 Modificar caminhos**
5. **Salvar Como** em `~/projects/active/` (os caminhos são relativos a essa pasta)

**Depois:**

```
image_path: imgs/<file>     (RELATIVE)
path_root : "."
path_mode : RELATIVE
```

Agora você pode mover a pasta `active/` inteira pra qualquer lugar
e o dataset continua funcionando — sem cópia, sem renomear nada.

## Erros comuns

| Erro | Causa | Diagnóstico |
|---|---|---|
| Validei o `path_root` e deu verde, mas as imagens estão erradas | A validação só confere **existência**, não conteúdo | Confira na galeria; aponte o `path_root` para a subpasta correta |
| A operação *Corrigir subpasta* não aparece no combo | O dataset é `ABSOLUTE` (não tem `path_root`) | Use REWRITE_RELATIVE ou COPY_ORIGINALS pra convertê-lo em relativo primeiro |
| `REBASE_PREFIX: pasta destino '…' não existe` | Você digitou um caminho inválido como prefixo novo | Confirme com Finder/Explorer que a pasta realmente existe |
| Muitas linhas em `failures` no resumo final | Imagens não foram encontradas no destino | Verifique se o prefixo novo + estrutura interna bate com o local real das imagens. Tire algumas amostras do parquet e tente abrir manualmente |
| `fora de target_dir (…); use COPY_ORIGINALS…` (REWRITE_RELATIVE) | Alguma imagem está fora da pasta destino | Mude o destino para um ancestral comum, ou use COPY_ORIGINALS |
| `n_skipped` alto em COPY com skip ON | Re-execução sobre um destino que já tinha as cópias | Não é um problema; os arquivos idênticos foram pulados |
| Hashes diferem após COPY (*Foto original*) | **Não acontece** se a cópia foi byte-a-byte | Se acontecer, indica corrupção de disco; reporte como bug |

> ⚠️ **Face com falha sai do dataset resultante.** Em todas as
> operações de migração, a face cuja imagem não foi encontrada (ou que
> caiu fora do destino, ou que não começa com o prefixo antigo) **não
> entra** no dataset modificado — ela fica listada nas falhas da
> mensagem final. Confira o número de falhas antes de salvar: em
> REBASE_PREFIX, **Salvar** sobrescreve o parquet original com as faces
> que sobraram.

## O que **nunca** muda

Para tranquilidade forense — *Modificar Caminhos* é **garantidamente
não-destrutivo** em relação ao conteúdo biométrico:

| Atributo | Garantia |
|---|---|
| `hash` (SHA-256 da imagem) | Imutável em todas as operações |
| `identity` | Imutável |
| `embedding_v2`, `embedding_v4` | Imutáveis |
| `ff_info`, `ofiq_info`, `serfiq_info` | Imutáveis |
| `column_specs` (esquema do dataset) | Imutáveis |
| `name`, `description`, `is_aligned` | Imutáveis (exceto em COPY_ORIGINALS com *Face alinhada*, que produz um dataset alinhado) |
| Arquivos de imagem originais | Intactos em *Corrigir subpasta*, REBASE_PREFIX e REWRITE_RELATIVE; copiados (não movidos nem deletados) em COPY_ORIGINALS |

O que **pode** mudar é o número de faces: as que falham numa migração
saem do dataset resultante (ver o aviso em [Erros comuns](#erros-comuns)).

Toda operação *Modificar Caminhos* é **registrada no Audit Log** com a
operação **Modificação de caminhos das imagens**, o modo usado (ou
`fix_path_root` na correção leve) e a contagem de faces afetadas —
preservando a cadeia de custódia.

---

[← Capítulo 02b — Criando a partir de vídeo](02b_criando_dataset_de_video.md) · [Capítulo 04 — Navegando no dataset →](04_navegando_dataset.md)
