# AT2 — Redes Neurais MLP e Deploy de API em Produção

Trabalho prático da disciplina de Aprendizagem de Máquina. O projeto cobre o ciclo completo de um modelo de classificação: exploração de dados, treinamento de uma Rede Neural Multicamadas (MLP), encapsulamento em Pipeline Scikit-Learn e disponibilização da inferência via API REST com FastAPI.

## Sumário

- [Visão geral](#visão-geral)
- [Dataset](#dataset)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Como reproduzir](#como-reproduzir)
- [Notebook de análise e treinamento](#notebook-de-análise-e-treinamento)
- [API REST](#api-rest)
- [Testes de integração HTTP](#testes-de-integração-http)
- [Resultados](#resultados)
- [Decisões técnicas](#decisões-técnicas)

## Visão geral

O objetivo de negócio é prever se um cliente de um banco português vai aderir a um depósito a prazo (`deposit = yes/no`) com base em dados demográficos e no histórico de campanhas telefônicas. A predição permite priorizar o time de telemarketing e reduzir o custo por conversão.

O fluxo implementado:

1. **Notebook** (`notebooks/analise_treinamento.ipynb`): EDA, construção do Pipeline com `ColumnTransformer` + `MLPClassifier`, avaliação com MCC e matriz de confusão, e serialização do pipeline completo em `models/pipeline_rede_neural.joblib`.
2. **API** (`src/app.py`): carrega o pipeline no startup via `lifespan` e expõe endpoints de diagnóstico e predição, com validação estrita de payload pelo Pydantic (`src/schemas.py`).
3. **Testes HTTP** (`requisicoes.http`): cenários válidos e inválidos para comprovar o funcionamento e a rejeição de entradas malformadas (HTTP 422).

## Dataset

- **Fonte:** Bank Marketing Dataset (UCI Machine Learning Repository / Kaggle).
- **Arquivo local:** `data/dataset.csv` (versão `bank.csv`, 17 colunas).
- **Volume:** 11.162 registros, sem valores ausentes.
- **Variável alvo:** `deposit` (binária: `yes` / `no`), praticamente balanceada (5.873 `no` vs. 5.289 `yes`).
- **Features:** 7 numéricas (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) e 9 categóricas (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`).

### Dicionário de dados (principais colunas)

| Coluna | Tipo | Significado |
| --- | --- | --- |
| `age` | int | Idade do cliente |
| `job` | categórico | Profissão |
| `marital` | categórico | Estado civil |
| `education` | categórico | Escolaridade |
| `default` | categórico | Possui crédito em default? |
| `balance` | int | Saldo médio anual (euros) |
| `housing` | categórico | Possui financiamento habitacional? |
| `loan` | categórico | Possui empréstimo pessoal? |
| `contact` | categórico | Tipo de contato da campanha |
| `day` | int | Dia do último contato |
| `month` | categórico | Mês do último contato |
| `duration` | int | Duração da última ligação (segundos) |
| `campaign` | int | Número de contatos nesta campanha |
| `pdays` | int | Dias desde o último contato (-1 = nunca) |
| `previous` | int | Número de contatos anteriores à campanha |
| `poutcome` | categórico | Resultado da campanha anterior |
| **`deposit`** | **binário** | **Target: aderiu ao depósito?** |

## Estrutura do repositório

```text
AT2_MLP_Erlon_Viana/
├── data/
│   └── dataset.csv                    # Base original (bank.csv)
├── models/
│   └── pipeline_rede_neural.joblib    # Pipeline completo serializado
├── notebooks/
│   └── analise_treinamento.ipynb      # EDA, treinamento e métricas
├── src/
│   ├── __init__.py
│   ├── schemas.py                     # Contratos Pydantic (entrada/saída)
│   └── app.py                         # FastAPI, lifespan e endpoints
├── requisicoes.http                   # Testes de integração HTTP
├── pyproject.toml                     # Dependências gerenciadas via UV
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## Pré-requisitos

- **Python 3.11+**
- **UV** (gerenciador de dependências e ambiente virtual). Instalação no Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Após instalar, feche e reabra o terminal para que o comando `uv` seja reconhecido.

## Como reproduzir

### 1. Clonar e entrar no projeto

```bash
git clone <URL_DO_REPOSITORIO>
cd AT2_MLP_Erlon_Viana
```

### 2. Sincronizar o ambiente

```bash
uv sync
```

Isso cria o `.venv` e instala todas as dependências declaradas no `pyproject.toml` (incluindo `fastapi[standard]`, necessário para o comando `fastapi dev`).

### 3. (Opcional) Registrar o kernel do Jupyter

```bash
uv run python -m ipykernel install --user --name=at2-mlp --display-name "Python (AT2 MLP)"
```

No VS Code, selecione o kernel **Python (AT2 MLP)** ao abrir o notebook.

### 4. Executar o notebook

Abra `notebooks/analise_treinamento.ipynb` e rode todas as células em ordem. Ao final, o arquivo `models/pipeline_rede_neural.joblib` será gerado.

### 5. Subir a API

```bash
uv run fastapi dev src/app.py
```

A API estará disponível em `http://127.0.0.1:8000`, com documentação interativa (Swagger UI) em `http://127.0.0.1:8000/docs`.

## Notebook de análise e treinamento

O notebook está organizado linearmente nas seguintes seções:

1. **Contexto do problema e dicionário de dados:** explicação do problema de negócio, origem do dataset e tabela descritiva das colunas.
2. **Análise exploratória resumida:**
   - `df.shape` → `(11162, 17)`;
   - verificação de valores ausentes (nenhum);
   - contagem da variável alvo (checagem de desbalanceamento);
   - histograma de `age` (distribuição de um atributo crítico).
3. **Construção do Pipeline Scikit-Learn:**
   - `train_test_split` com `test_size=0.2`, `random_state=77` e `stratify=y`;
   - `ColumnTransformer` com `StandardScaler` nas numéricas e `OneHotEncoder(handle_unknown="ignore")` nas categóricas;
   - `MLPClassifier` com arquitetura `(128, 64, 32)` (3 camadas ocultas), `learning_rate_init=0.001`, `max_iter=500`, `early_stopping=True`, `n_iter_no_change=15` e `random_state=77`.
4. **Avaliação do modelo:**
   - `matthews_corrcoef` no conjunto de teste;
   - `confusion_matrix` + `ConfusionMatrixDisplay`.
5. **Exportação do modelo:**
   - `joblib.dump(pipeline, "models/pipeline_rede_neural.joblib")`.

### Por que salvar o Pipeline completo e não apenas a rede?

Ao serializar o objeto `Pipeline`, o arquivo `.joblib` armazena também as médias e os desvios do `StandardScaler` e as categorias aprendidas pelo `OneHotEncoder`. Assim, a API recebe dados brutos e aplica as mesmas transformações do treino, sem risco de inconsistência de pré-processamento.

## API REST

### Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `GET` | `/diagnostico/modelo` | Metadados e status do serviço |
| `POST` | `/predict` | Inferência a partir de um payload validado |

### Ciclo de vida (`lifespan`)

O pipeline é carregado uma única vez no startup da aplicação, via `@asynccontextmanager`, e armazenado em `app.state.modelo`. As rotas apenas consultam esse estado — não há `joblib.load` dentro das funções de rota, evitando latência e consumo desnecessário de memória.

### Contratos Pydantic (`src/schemas.py`)

- **`PredicaoInput`:** mapeia as 16 features de entrada com tipagem estrita (`int`, `str`) e restrições `ge`/`le` nos campos numéricos (ex.: `age` entre 18 e 100, `day` entre 1 e 31).
- **`PredicaoOutput`:** retorna `categoria_prevista` (rótulo textual) e `score_confianca` (float entre 0.0 e 1.0, calculado por `predict_proba`).

### Exemplo de requisição

```http
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "age": 45,
  "job": "admin.",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 1500,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "may",
  "duration": 800,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

Resposta esperada (HTTP 200):

```json
{
  "categoria_prevista": "yes",
  "score_confianca": 0.87
}
```

## Testes de integração HTTP

O arquivo `requisicoes.http` na raiz do projeto contém os cenários de teste, todos com o cabeçalho `User-Agent: Tester-Audit-Agent`:

- `GET /diagnostico/modelo` — verifica o status e os metadados do serviço.
- `POST /predict` com Perfil A — payload válido simulando um cliente com perfil propenso à adesão.
- `POST /predict` com Perfil B — payload válido simulando um perfil distinto, para comprovar predições diferentes.
- `POST /predict` com payload inválido — por exemplo, `age` como texto ou valor fora dos limites, para comprovar a rejeição pelo Pydantic com HTTP 422 (Unprocessable Entity).

Para executar, use a extensão REST Client (VS Code) ou Thunder Client.

## Resultados

| Métrica | Valor |
| --- | --- |
| Matthews Correlation Coefficient (MCC) | 0,7096 |
| Arquitetura da MLP | `(128, 64, 32)` |
| `random_state` | 77 |
| Divisão treino/teste | 80% / 20% (estratificada) |

O MCC de 0,7096 indica boa capacidade de discriminação entre as classes, sem overfitting grave — valor considerado forte para o Bank Marketing.

## Decisões técnicas

- **`early_stopping=True`:** com 11 mil registros e três camadas ocultas, o early stopping usa uma fração do treino como validação e interrompe o treinamento quando a performance não melhora, reduzindo overfitting e tempo de treino.
- **`handle_unknown="ignore"` no `OneHotEncoder`:** evita erro caso a API receba uma categoria não vista no treino (ex.: uma profissão nova); a categoria é codificada como vetor de zeros.
- **`stratify=y` no `train_test_split`:** preserva a proporção das classes no treino e no teste, importante para métricas confiáveis.
- **Pipeline completo serializado:** garante que a API aplique exatamente o mesmo pré-processamento do treino.
- **`fastapi[standard]`:** necessário para o comando `fastapi dev`; sem o extra `standard`, o CLI não é instalado.
