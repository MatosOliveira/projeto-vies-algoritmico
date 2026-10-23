"""
config.py - parâmetros centrais do projeto (pipeline de análise de viés no ENEM).

Mantém os nomes BASE_DIR, DATA_DIR e TABLES_DIR usados por data_preparation.py.
"""
import os

# ---------------------------------------------------------------- diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, RUNS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- experimento
SEED = 42
TEST_SIZE = 0.30        # fração da amostra reservada ao teste
CALIB_SIZE = 0.25       # fração do TREINO reservada à calibração dos limiares (pós-proc.)

# Critérios mínimos de suporte para reportar métricas de grupo
MIN_N_GRUPO = 100       # tamanho mínimo (teste) para taxa de seleção
MIN_POS_GRUPO = 30      # positivos mínimos (teste) para TPR
MIN_NEG_GRUPO = 30      # negativos mínimos (teste) para FPR
N_BOOT = 200            # repetições bootstrap para intervalos de confiança

# ---------------------------------------------------------------- variáveis
# Alvo construído a partir das letras de Q023 (tipo de escola do ensino médio):
#   A = só pública | B, C = mista (parte pública, parte privada) |
#   D = só privada sem bolsa | E = só privada com bolsa integral | F = excluída na preparação
# Letras ausentes do dicionário de uma variante são EXCLUÍDAS daquela análise.
TARGETS = {
    "principal": {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1},   # privada (D, E) x demais
    "estrito":   {"A": 0, "D": 1, "E": 1},                   # sem trajetória mista
    "sem_bolsa": {"A": 0, "D": 1},                           # só pública x privada paga
}
Q023_LABELS = {"A": "publica", "B": "mista_sem_bolsa", "C": "mista_bolsa_integral",
               "D": "privada_sem_bolsa", "E": "privada_bolsa_integral"}

# Atributos usados na AVALIAÇÃO de fairness (colunas já derivadas em data_preparation.py)
EVAL_ATTRS = ["TP_SEXO", "RACA_GRUPO", "REGIAO", "RENDA_GRUPO", "SG_UF_PROVA"]
# Atributos para os quais se aplica MITIGAÇÃO por padrão (UF fica opcional: --include-uf)
MITIGATION_ATTRS = ["TP_SEXO", "RACA_GRUPO", "REGIAO", "RENDA_GRUPO"]
# Grupos que aparecem nas tabelas mas ficam fora do resumo (ex.: gap máx-mín)
EXCLUDE_FROM_SUMMARY = {"RACA_GRUPO": ["nao_informada"]}

# Preditores (sem notas individuais: não há chave entre participantes e resultados)
CAT_FEATURES = (
    ["TP_SEXO", "TP_COR_RACA", "SG_UF_PROVA", "TP_ST_CONCLUSAO"]
    + ["Q001", "Q002", "Q003", "Q004", "Q006", "Q007"]
    + [f"Q{i:03d}" for i in range(8, 23)]
)
NUM_FEATURES = ["TP_FAIXA_ETARIA", "Q005"]
# Colunas que representam atributos sensíveis dentro das features (cenário "A_sem_sens")
SENSITIVE_FEATURES = ["TP_SEXO", "TP_COR_RACA", "SG_UF_PROVA", "Q007"]
# Contexto municipal agregado (cenário B)
CTX_FEATURES = [
    "CTX_NU_NOTA_CN", "CTX_NU_NOTA_CH", "CTX_NU_NOTA_LC", "CTX_NU_NOTA_MT",
    "CTX_NU_NOTA_REDACAO", "CTX_NOTA_MEDIA",
]

SCENARIOS = {
    "A": "questionário e demográficos (principal)",
    "A_sem_sens": "A sem sexo, raça/cor, UF e renda como features",
    "B": "A + contexto municipal de notas (sensibilidade)",
}
