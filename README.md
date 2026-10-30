# Visualização da Folha de Presença

Dashboard (Streamlit) para comparar o ground truth de listas de presença com a inferência de um modelo, célula a célula (aluno × aula), com métricas por imagem e métricas gerais.

Modelos: Nemotron, Qwen, Ministral · Resoluções: R1344, R1792 · Prompts: Chain of Thought, GEPA, Zero-shot.

## Requisitos

- Python 3.10+
- Acesso à internet (os dados são lidos de pastas e planilhas públicas do Google Drive)

## Passo a passo

1. Entre na pasta do projeto:

   ```bash
   cd /home/jamillalobo/Documentos/research/dashboard-sgp
   ```

2. Crie o ambiente virtual (só na primeira vez):

   ```bash
   python3 -m venv .venv
   ```

3. Ative o ambiente:

   ```bash
   source .venv/bin/activate        # Linux/macOS
   # .venv\Scripts\activate         # Windows
   ```

4. Instale as dependências (só na primeira vez):

   ```bash
   pip install -r requirements.txt
   ```

   Depois crie o arquivo de links: `cp .env.example .env` e preencha (veja a seção Configuração).

5. Rode o dashboard escolhendo a fonte dos dados (o `--` antes de `--source` é obrigatório):

   ```bash
   streamlit run app.py -- --source drive    # lê imagens, GT e predições do Drive (padrão)
   streamlit run app.py -- --source upload   # você arrasta os arquivos na tela
   ```

   Sem `--source`, usa `drive`. O modo `upload` não precisa de internet nem do `.env`.

6. Abra no navegador: <http://localhost:8501> (abre sozinho na maioria dos casos).

Para parar, use `Ctrl+C` no terminal.

## Como usar

### Modo `drive`

1. Escolha a **Imagem** no seletor do topo (só aparecem imagens com GT e predição para o filtro).
2. Logo abaixo dele, escolha o **Modelo**, a **Resolução** e o **Prompt**.
3. Veja:
   - **Métricas gerais**: vêm da planilha consolidada (sem processamento, sem corte), com kappa agregado, presenças acertadas (TP), faltas acertadas (TN) e tabela por amostra.
   - **Métricas da imagem**: acurácia, kappa, precisão, recall, F1 e TP/FP/FN/TN.
   - **Três painéis**: imagem, labels (GT) e inferência. As cores seguem a legenda da grade (veja "Como cada célula é classificada").

### Modo `upload`

1. Arraste (ou clique para escolher) três arquivos: a **imagem** (`.jpg`/`.jpeg`/`.png`), o **ground truth** (`.json`, ou `.md` com JSON) e a **predição** (`.json`).
2. Os painéis e as métricas da imagem aparecem quando os três estiverem carregados. As métricas gerais da planilha só existem no modo `drive`.

Formato esperado dos JSONs: `{"students": [{"student_name": "...", "labels": [30 valores 0/1]}]}`. O GT também pode vir embrulhado numa chave de id, e cercas markdown (` ```json `) ao redor são ignoradas. A comparação é feita por posição da linha do aluno.

## Como cada célula é classificada

O GT precisa ter 30 valores por aluno (pode ter `null`); a predição pode vir com outro número de aulas ou de alunos. O pareamento é **por posição** (aluno i do GT com aluno i da predição). As mesmas regras valem para as métricas e para a grade, que usam a mesma função (`classify` em `data.py`).

| GT | Predição | Classe |
| --- | --- | --- |
| 1 | 1 | TP |
| 0 | 0 | TN |
| 0 | 1 | FP |
| 1 | 0 | FN |
| `null` | qualquer | fora da conta |
| 1 | `null` ou ausente | FN |
| 0 | `null` ou ausente | FP |
| ausente | 0 ou 1 | FP |
| ausente | `null` | fora da conta |

Predição ausente é resposta errada: com GT 0 ela vira FP, não TN.

| Caso | O que acontece |
| --- | --- |
| Menos aulas previstas (ex.: 28) | Aulas 29 e 30 sem predição: FN se o GT é 1, FP se é 0. Nenhum vira TN. |
| Mais aulas previstas (31, 32…) | Aulas além do GT: 0 ou 1 vira FP, `null` fica fora. |
| Mais alunos que o GT | Toda célula do aluno a mais vira FP, inclusive 0 e `null`. Nunca gera TP, TN ou FN. |
| Menos alunos que o GT | Aluno omitido: GT 1 vira FN e GT 0 vira FP. Uma omissão no meio da lista desloca os alunos seguintes, gerando uma cascata de FP e FN. |

Por isso precisão e kappa não medem só o que o modelo marcou: aulas ausentes e alunos omitidos com GT 0 entram como FP. Quando a estrutura difere do GT, o app mostra um aviso acima da predição.

### Legenda da grade

A legenda aparece sempre completa:

| Marca | Significado |
| --- | --- |
| Preto cheio | Presente (1) |
| Vazio | Ausente (0) |
| Laranja cheio | Falsa presença (FP) |
| Contorno laranja | Presença perdida (FN) |
| Laranja hachurado | `null`/ausente com GT 0 (FP) |
| Contorno laranja tracejado | `null`/ausente com GT 1 (FN) |
| Contorno azul | Aula ou aluno a mais (FP) (colunas além de 30 têm o número em azul) |
| Cinza tracejado | Fora da conta (`null`) |

Se o modelo devolveu JSON inválido para a imagem, o app mostra um aviso no lugar das métricas.

## Cache (modo `drive`)

Os arquivos baixados ficam em `.cache/`, então a primeira abertura é mais lenta. Se os dados do Drive mudarem, clique em **Recarregar do Drive** ao lado dos filtros (ou apague a pasta `.cache/`).

## Estrutura

| Arquivo | Função |
| --- | --- |
| `app.py` | Interface Streamlit |
| `data.py` | Índices de imagens/GT/predições, parsing dos JSONs e métricas |
| `drive.py` | Acesso ao Drive/planilhas e cache; lê os links do `.env` |

## Configuração

Só para o modo `drive`: os links do Drive ficam no arquivo `.env` (que não vai para o git). Copie o modelo e preencha:

```bash
cp .env.example .env
```

| Variável | O que é |
| --- | --- |
| `GT_FOLDERS` | Pastas com subpastas `images` e `labels`, separadas por vírgula (uma por lote) |
| `PREDICTIONS_FOLDER` | Pasta com as predições dos modelos |
| `METRICS_SHEET` | Planilha consolidada de métricas |

Aceita o link completo ou só o ID. Para incluir novas imagens com GT, acrescente a pasta em `GT_FOLDERS`. **Reinicie o streamlit** depois de editar o `.env`.

As pastas e planilhas precisam estar com acesso "qualquer pessoa com o link".

## Problemas comuns

- **"Não consegui ler o Drive"**: verifique a internet e se as pastas continuam públicas.
- **Imagem não aparece no seletor**: ela não tem GT nas pastas de `GT_FOLDERS` ou não tem predição para o modelo/prompt/resolução escolhidos.
- **Porta ocupada**: rode com `streamlit run app.py --server.port 8502`.
