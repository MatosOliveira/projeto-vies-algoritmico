"""preprocessing.py - leitura da amostra, montagem de X/y, partição e codificação."""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C


def load_year(year):
    """Lê enem_amostra_{ano}.parquet (ou enem_amostra.parquet filtrando NU_ANO)."""
    p = os.path.join(C.PROCESSED_DIR, f"enem_amostra_{year}.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p)
    else:
        p0 = os.path.join(C.PROCESSED_DIR, "enem_amostra.parquet")
        if not os.path.exists(p0):
            raise FileNotFoundError(f"Não encontrei {p} nem {p0}. Rode data_preparation.py.")
        df = pd.read_parquet(p0)
    df = df[df["NU_ANO"] == year].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Nenhum registro de {year} no arquivo lido.")
    return df


def _as_obj(s):
    """Série categórica/string -> object com NaN real (compatível com o imputer)."""
    return s.astype(object).where(s.notna(), np.nan)


def build_dataset(df, scenario, target):
    """
    Devolve X (DataFrame), y, sens (DataFrame str), w (pesos amostrais), cat, num, q023.
    O alvo é montado a partir das letras de Q023 conforme C.TARGETS[target]; letras fora
    do mapeamento da variante são excluídas.
    """
    mapping = C.TARGETS[target]
    letters = df["Q023"].astype(str).str.strip()
    df = df[letters.isin(list(mapping))].reset_index(drop=True)
    q023 = df["Q023"].astype(str).str.strip().reset_index(drop=True)
    y = q023.map(mapping).astype(int)

    cat, num = list(C.CAT_FEATURES), list(C.NUM_FEATURES)
    if scenario == "A_sem_sens":
        cat = [c for c in cat if c not in C.SENSITIVE_FEATURES]
    elif scenario == "B":
        missing = [c for c in C.CTX_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"Cenário B exige colunas de contexto ausentes: {missing}. "
                           "Rode data_preparation.py com os arquivos de RESULTADOS.")
        num += C.CTX_FEATURES
    elif scenario != "A":
        raise ValueError(f"Cenário desconhecido: {scenario}")

    X = df[cat + num].copy()
    for c in cat:
        X[c] = _as_obj(X[c])
    sens = pd.DataFrame({a: _as_obj(df[a]).astype(str) for a in C.EVAL_ATTRS})
    w = df["PESO_AMOSTRAL"].astype(float).reset_index(drop=True)
    return X.reset_index(drop=True), y, sens, w, cat, num, q023


def split_indices(y, sens, seed=C.SEED):
    """Treino / calibração / teste, estratificados por alvo × renda × raça × sexo."""
    key = (y.astype(str) + "|" + sens["RENDA_GRUPO"] + "|" + sens["RACA_GRUPO"]
           + "|" + sens["TP_SEXO"])
    counts = key.map(key.value_counts())
    key = key.where(counts >= 10, "raro")
    if key.value_counts().min() < 2:          # proteção: cai para estratificar só pelo alvo
        key = y.astype(str)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=key, random_state=seed)
    fit, cal = train_test_split(tr, test_size=C.CALIB_SIZE, stratify=key.iloc[tr],
                                random_state=seed)
    return fit, cal, te


def make_preprocessor(cat, num):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:                          # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    transformers = [("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                      ("ohe", ohe)]), cat)]
    if num:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                              ("sc", StandardScaler())]), num))
    return ColumnTransformer(transformers, sparse_threshold=0.0)
