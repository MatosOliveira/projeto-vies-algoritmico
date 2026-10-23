"""
mitigation.py - técnicas de mitigação (pré, durante e pós-processamento).

  * reweighing  (pré)   : pesos de Kamiran & Calders sobre (atributo sensível, alvo).
  * reductions  (durante): ExponentiatedGradient com DemographicParity ou EqualizedOdds.
  * threshold   (pós)   : ThresholdOptimizer com estimador já treinado (prefit) e limiares
                          ajustados numa partição de CALIBRAÇÃO separada do treino.

Pesos amostrais (PESO_AMOSTRAL):
  * reweighing e baselines usam os pesos de forma explícita.
  * reductions: os pesos entram no estimador base; as restrições do Fairlearn são estimadas
    sem pesos (limitação a registrar no TCC).
  * threshold: o Fairlearn não aceita pesos ao ajustar os limiares (limitação a registrar).
"""
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

import config as C
from train import fit_logistic, normalize


# ------------------------------------------------------------------ pré-processamento
def reweighing_weights(y, a, w):
    """Peso final = w * P(a)P(y) / P(a,y), com frequências ponderadas por w."""
    df = pd.DataFrame({"y": np.asarray(y), "a": np.asarray(a), "w": np.asarray(w, float)})
    tot = df["w"].sum()
    p_a = df.groupby("a")["w"].sum() / tot
    p_y = df.groupby("y")["w"].sum() / tot
    p_ay = df.groupby(["a", "y"])["w"].sum() / tot
    key = pd.MultiIndex.from_arrays([df["a"], df["y"]])
    factor = (p_a.reindex(df["a"]).to_numpy() * p_y.reindex(df["y"]).to_numpy()
              / p_ay.reindex(key).to_numpy())
    return df["w"].to_numpy() * factor


def fit_reweighing(X, y, a, w):
    return fit_logistic(X, y, reweighing_weights(y, a, w))


# ------------------------------------------------------------------ durante o treino
class WeightedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Regressão logística que combina os pesos do Fairlearn com os pesos amostrais."""

    def __init__(self, base_weight=None, max_iter=1000):
        self.base_weight = base_weight
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.base_weight is not None:
            sw = sw * np.asarray(self.base_weight, float)
        if sw.sum() <= 0:
            sw = np.ones(len(y))
        self.lr_ = LogisticRegression(max_iter=self.max_iter)
        self.lr_.fit(X, y, sample_weight=sw / sw.mean())
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X):
        return self.lr_.predict(X)

    def predict_proba(self, X):
        return self.lr_.predict_proba(X)


def fit_reductions(X, y, a, w, constraint="demographic_parity", max_iter=50, eps=0.01):
    cons = DemographicParity() if constraint == "demographic_parity" else EqualizedOdds()
    est = WeightedLogisticRegression(base_weight=np.asarray(w, float))
    eg = ExponentiatedGradient(est, cons, max_iter=max_iter, eps=eps)
    eg.fit(X, np.asarray(y), sensitive_features=np.asarray(a))
    return eg


def predict_reductions(eg, X, seed=C.SEED):
    return np.asarray(eg.predict(np.asarray(X, dtype=np.float64),
                                 random_state=seed)).astype(int)


# ------------------------------------------------------------------ pós-processamento
def fit_threshold(base_model, X_cal, y_cal, a_cal, constraint="equalized_odds"):
    opt = ThresholdOptimizer(estimator=base_model, constraints=constraint,
                             objective="accuracy_score", prefit=True,
                             predict_method="predict_proba")
    opt.fit(np.asarray(X_cal, dtype=np.float64), np.asarray(y_cal),
            sensitive_features=np.asarray(a_cal))
    return opt


def predict_threshold(opt, X, a, seed=C.SEED):
    X = np.asarray(X, dtype=np.float64)   # evita conflito float32/float64 no pandas recente
    return np.asarray(opt.predict(X, sensitive_features=np.asarray(a),
                                  random_state=seed)).astype(int)
