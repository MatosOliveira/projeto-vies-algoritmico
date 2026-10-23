# Análise de Viés Algorítmico em Modelos de Classificação Aplicados a Dados Educacionais e Socioeconômicos Brasileiros

## 🎯 Visão Geral

Este projeto investiga **viés algorítmico** em modelos supervisionados treinados com os **microdados do ENEM 2024 e 2025**, usando apenas informações socioeconômicas e demográficas dos participantes. O objetivo é avaliar:

- o desempenho dos modelos;
- a equidade (*fairness*) entre grupos sensíveis (sexo, raça/cor, região, renda e UF);
- o impacto de técnicas de mitigação de viés (pré, durante e pós-processamento);
- os *trade-offs* entre acurácia e equidade.

A variável alvo é construída a partir da **questão Q023** do questionário socioeconômico (tipo de escola em que o participante cursou o ensino médio). O alvo vale **1 para escola privada** e **0 para os demais casos**, com três variantes definidas em `config.TARGETS`:

| Variante    | Classe 0 (pública/demais)             | Classe 1 (privada)     | Excluídos |
|-------------|---------------------------------------|------------------------|-----------|
| `principal` | A (só pública), B e C (trajetória mista) | D (sem bolsa), E (bolsa integral) | — |
| `estrito`   | A                                     | D, E                   | B, C      |
| `sem_bolsa` | A                                     | D                      | B, C, E   |

Participantes com Q023 = F (não frequentou o ensino médio) são removidos na preparação dos dados.

---

## 📂 Estrutura do Projeto

```
projeto-vies-algoritmico/
├── data/
│   ├── PARTICIPANTES_2024.csv          # microdados brutos do INEP (não versionados)
│   ├── RESULTADOS_2024.csv
│   ├── PARTICIPANTES_2025.csv
│   ├── RESULTADOS_2025.csv
│   └── processed/                      # gerado por data_preparation.py
│       ├── enem_amostra_{ano}.parquet      # amostra tratada (entrada do pipeline)
│       ├── contexto_municipal_{ano}.csv    # médias de notas por município (auditoria)
│       └── _tmp/                           # blocos temporários (apagados ao final)
├── src/
│   ├── config.py                # caminhos, parâmetros e definição de variáveis
│   ├── data_preparation.py      # etapa 1: limpeza, amostragem e contexto municipal
│   ├── main.py                  # etapa 2: treino, mitigação e avaliação
│   ├── preprocessing.py         # leitura da amostra, X/y, partição e codificação
│   ├── train.py                 # modelos baseline
│   ├── mitigation.py            # técnicas de mitigação
│   └── evaluate.py              # métricas de desempenho e fairness
├── results/
│   ├── tables/                  # relatórios da preparação dos dados (por ano)
│   │   ├── cleaning_log_{ano}.csv
│   │   ├── amostra_alocacao_estratos_{ano}.csv
│   │   └── amostra_resumo_grupos_{ano}.csv
│   ├── runs/                    # uma pasta por execução: {ano}_{cenário}_{alvo}/
│   │   └── 2024_A_principal/
│   │       ├── desempenho.csv
│   │       ├── fairness_grupos.csv
│   │       ├── fairness_resumo.csv
│   │       ├── diagnostico_q023.csv
│   │       ├── q023_por_grupo.csv
│   │       └── meta.json
│   └── figures/                 # reservado para gráficos
├── requirements.txt
└── README.md
```

As pastas `data/processed/` e `results/` (e subpastas) são criadas automaticamente por `config.py`.

---

## 🧱 Descrição dos Scripts

### 🔧 `config.py`
Parâmetros centrais do projeto:

- caminhos das pastas (`DATA_DIR`, `PROCESSED_DIR`, `RUNS_DIR`, `TABLES_DIR`, `FIGURES_DIR`);
- parâmetros do experimento: semente (`SEED = 42`), fração de teste (30%) e fração do treino reservada à calibração dos limiares (25%);
- suporte mínimo para reportar métricas de grupo (`MIN_N_GRUPO`, `MIN_POS_GRUPO`, `MIN_NEG_GRUPO`) e número de repetições *bootstrap* (`N_BOOT = 200`);
- variantes do alvo (`TARGETS`) e rótulos das letras de Q023;
- atributos avaliados (`EVAL_ATTRS`: sexo, raça, região, renda e UF) e mitigados por padrão (`MITIGATION_ATTRS`: sexo, raça, região e renda);
- preditores categóricos e numéricos e os cenários de features (`SCENARIOS`):

| Cenário      | Features |
|--------------|----------|
| `A`          | questionário socioeconômico e dados demográficos (principal) |
| `A_sem_sens` | `A` sem sexo, raça/cor, UF e renda (Q007) como features |
| `B`          | `A` + contexto municipal de notas (análise de sensibilidade) |

---

### 🧹 `data_preparation.py` — etapa 1
Lê os CSVs do INEP **em blocos** (sem carregar o arquivo completo em memória) e gera a amostra usada pelo pipeline:

1. **Limpeza dos participantes:** exclui treineiros, registros com atributo sensível inválido (sem imputação), Q023 ausente/inválida e Q023 = F. Cada exclusão é contada no funil de limpeza.
2. **Variáveis derivadas:** `REGIAO`, `RACA_GRUPO` (0 e 6 → `nao_informada`), `RENDA_GRUPO` (faixas de Q007 em salários mínimos) e harmonização de Q022 entre 2024 e 2025.
3. **Amostragem estratificada** (~200 mil registros por padrão) por ano × sexo × raça × região × renda, com piso por estrato e **peso amostral** (`PESO_AMOSTRAL = 1/q`) para que as métricas representem a população.
4. **Contexto municipal:** médias de notas por município de prova (com *fallback* para a UF quando há menos de 30 presentes), calculadas a partir do arquivo de resultados.

> Os arquivos de participantes e de resultados **não possuem chave comum** (nota do dicionário do INEP). Por isso as notas entram apenas como contexto agregado do município (cenário `B`), nunca como dado individual.

Principais opções:

| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `--years` | `2024 2025` | anos a processar |
| `--tag` | — | sufixo nos arquivos de saída (ex.: `--tag 2024` → `enem_amostra_2024.parquet`) |
| `--target-n` | `200000` | tamanho aproximado da amostra |
| `--min-por-estrato` | `100` | piso por estrato |
| `--min-presentes-mun` | `30` | mínimo de presentes para usar a média do município |
| `--skip-contexto` | — | não lê os arquivos de resultados (cenário `B` fica indisponível) |
| `--keep-treineiros` | — | mantém os treineiros |
| `--chunksize`, `--seed`, `--data-dir`, `--keep-tmp` | | leitura em blocos, semente, pasta dos dados e blocos temporários |

---

### 🗂️ `preprocessing.py`
- `load_year`: lê `enem_amostra_{ano}.parquet` (ou `enem_amostra.parquet`, filtrando o ano);
- `build_dataset`: monta X, y, atributos sensíveis, pesos e as letras de Q023 conforme o cenário e o alvo;
- `split_indices`: divide em **treino / calibração / teste** (≈ 52,5% / 17,5% / 30%), estratificado por alvo × renda × raça × sexo;
- `make_preprocessor`: imputação (moda/mediana) + *one-hot* + padronização, **ajustados apenas no treino**.

### 🤖 `train.py`
Modelos baseline, treinados com os pesos amostrais normalizados:

- **Regressão Logística**;
- **Árvore de Decisão** (`max_depth = 10`).

### 🛠️ `mitigation.py`
Técnicas aplicadas à regressão logística, uma vez para cada atributo sensível mitigado:

| Etapa | Técnica | Identificador em `technique` |
|-------|---------|------------------------------|
| Pré-processamento | Reweighing (Kamiran & Calders) | `reweighing` |
| Durante o treino | ExponentiatedGradient (Fairlearn) com Demographic Parity / Equalized Odds | `reductions_dp` / `reductions_eo` |
| Pós-processamento | ThresholdOptimizer (Fairlearn) com Equalized Odds / Demographic Parity, ajustado na partição de calibração | `threshold_eo` / `threshold_dp` |

Limitações registradas: nas *reductions* as restrições do Fairlearn são estimadas sem os pesos amostrais, e o ThresholdOptimizer não aceita pesos ao ajustar os limiares.

### 📈 `evaluate.py`
Usa uma única predição por modelo e os pesos amostrais do conjunto de teste:

- **Desempenho:** acurácia, precisão, recall, F1, acurácia balanceada, AUC, taxa de seleção e prevalência (com e sem pesos);
- **Fairness por grupo:** taxa base, taxa de seleção, TPR, FPR e acurácia;
- **Resumo de fairness:** `dp_diff`, `dp_ratio` (impacto desigual), `tpr_diff`, `fpr_diff`, `eo_diff` e `base_rate_diff`, com **IC 95% por bootstrap**. Grupos com suporte insuficiente e o grupo `nao_informada` de raça ficam fora do resumo;
- **Diagnóstico por Q023:** proporção classificada como privada em cada letra de Q023, no total e por faixa de renda.

### 🚀 `main.py` — etapa 2
Para cada combinação de ano × cenário × alvo:

1. carrega a amostra e monta o conjunto de dados;
2. divide em treino, calibração e teste e ajusta o pré-processador no treino;
3. treina os baselines;
4. aplica as técnicas de mitigação para cada atributo;
5. avalia todos os modelos em **todos** os atributos sensíveis;
6. salva os resultados em `results/runs/{ano}_{cenário}_{alvo}/`.

Principais opções:

| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `--year` | `2024` | um ou mais anos |
| `--scenario` | `A` | `A`, `A_sem_sens` e/ou `B` |
| `--target` | `principal` | `principal`, `estrito` e/ou `sem_bolsa` |
| `--attrs` | sexo, raça, região, renda | atributos a mitigar |
| `--include-uf` | — | mitiga também por UF |
| `--only-baseline` | — | treina e avalia apenas os baselines |
| `--skip-reductions` | — | pula o ExponentiatedGradient (execução rápida) |
| `--reductions-eo` | — | roda as *reductions* também com Equalized Odds |
| `--reductions-n` | `0` (todo o treino) | subamostra do treino usada só nas *reductions* |
| `--reductions-max-iter` | `50` | iterações do ExponentiatedGradient |
| `--no-weights` | — | ignora `PESO_AMOSTRAL` |
| `--n-boot` | `200` | repetições bootstrap (0 desativa os IC) |
| `--seed` | `42` | semente |

---

## 🔄 Fluxo do Projeto

```
 data/PARTICIPANTES_{ano}.csv      data/RESULTADOS_{ano}.csv
              │                                │
              ▼                                ▼
 ┌─────────────────────────────────────────────────────────┐
 │ data_preparation.py                                     │
 │  limpeza em blocos → amostra estratificada com pesos    │
 │  → contexto municipal de notas (agregado)               │
 └───────────────┬─────────────────────────────┬───────────┘
                 ▼                             ▼
 data/processed/enem_amostra_{ano}.parquet   results/tables/*_{ano}.csv
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────┐
 │ main.py                                                 │
 │  preprocessing → treino / calibração / teste            │
 │  train         → regressão logística, árvore de decisão │
 │  mitigation    → reweighing, reductions, threshold      │
 │  evaluate      → desempenho, fairness (IC 95%), Q023    │
 └───────────────────────────┬─────────────────────────────┘
                             ▼
             results/runs/{ano}_{cenário}_{alvo}/
```

---

## 🧪 Como Obter os Resultados

Os comandos abaixo são executados a partir da **raiz do projeto**. Também é possível entrar em `src/` e rodar `python data_preparation.py ...` e `python main.py ...`.

### 1. Instale as dependências

```
python -m pip install -r requirements.txt
```

Bibliotecas: pandas, numpy, scikit-learn, scipy, fairlearn, pyarrow (leitura e gravação de Parquet), joblib, matplotlib e seaborn.

### 2. Coloque os microdados em `data/`

Baixe os microdados do ENEM no portal do INEP e copie os CSVs de participantes e de resultados para `data/`. O script identifica o ano e o tipo pelo **nome do arquivo**, que precisa conter `PARTICIPANTES` ou `RESULTADOS` e o ano (ex.: `PARTICIPANTES_2024.csv`, `RESULTADOS_2024.csv`). Deve haver apenas um arquivo de cada tipo por ano.

### 3. Prepare as amostras (etapa 1)

Gere uma amostra por ano, usando `--tag` para que um ano não sobrescreva o outro:

```
python src/data_preparation.py --years 2024 --tag 2024
python src/data_preparation.py --years 2025 --tag 2025
```

Saídas: `data/processed/enem_amostra_{ano}.parquet`, `data/processed/contexto_municipal_{ano}.csv` e as tabelas `cleaning_log`, `amostra_alocacao_estratos` e `amostra_resumo_grupos` em `results/tables/`.

Para um teste rápido, use uma amostra menor (ex.: `--target-n 50000`). Sem `--tag`, os anos são processados juntos e gravados em `enem_amostra.parquet`, que o `main.py` também consegue ler.

### 4. Rode o pipeline de modelagem (etapa 2)

Configuração usada nos resultados atuais (`results/runs/2024_A_*`):

```
python src/main.py --year 2024 --scenario A --target principal estrito sem_bolsa --reductions-n 60000
```

Outros exemplos:

```
# os dois anos e os três cenários (alvo principal)
python src/main.py --year 2024 2025 --scenario A A_sem_sens B --reductions-n 60000

# execução rápida, sem ExponentiatedGradient
python src/main.py --year 2024 --skip-reductions

# apenas os baselines
python src/main.py --year 2024 --only-baseline
```

O ExponentiatedGradient é a etapa mais demorada. `--reductions-n` limita o tamanho do treino usado só nessa técnica; as demais continuam usando o treino completo.

### 5. Consulte os resultados

Cada execução grava em `results/runs/{ano}_{cenário}_{alvo}/`:

| Arquivo | Conteúdo |
|---------|----------|
| `desempenho.csv` | métricas globais (ponderadas) de cada modelo/técnica |
| `fairness_grupos.csv` | taxa base, taxa de seleção, TPR, FPR e acurácia por grupo e atributo |
| `fairness_resumo.csv` | `dp_diff`, `dp_ratio`, `tpr_diff`, `fpr_diff`, `eo_diff` com IC 95% |
| `diagnostico_q023.csv` | % classificado como privada por letra de Q023 (total e por renda) |
| `q023_por_grupo.csv` | composição das letras de Q023 em cada grupo (descritivo) |
| `meta.json` | parâmetros usados e tamanhos das partições |

Em todos os CSVs, as colunas `model`, `technique`, `mitigated_attr` e `constraint` identificam o experimento, e `eval_attr` indica o atributo sensível avaliado.

---

## 📚 Bibliotecas Utilizadas

| Biblioteca | Uso |
|------------|-----|
| **pandas / numpy** | manipulação de dados e cálculos numéricos |
| **pyarrow** | arquivos Parquet da amostra tratada |
| **scikit-learn** | pré-processamento, modelos e métricas de desempenho |
| **scipy** | dependência numérica do scikit-learn e do Fairlearn |
| **Fairlearn** | ExponentiatedGradient (*reductions*) e ThresholdOptimizer |
| **joblib** | serialização de objetos |
| **matplotlib / seaborn** | geração de gráficos |
