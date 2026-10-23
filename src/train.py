"""train.py - modelos baseline (regressão logística e árvore de decisão)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import config as C


def normalize(w):
    """Pesos com média 1 (mantém a regularização comparável entre execuções)."""
    w = np.asarray(w, dtype=float)
    return w / w.mean()


def fit_logistic(X, y, w=None):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y, sample_weight=None if w is None else normalize(w))
    return model


def fit_tree(X, y, w=None):
    model = DecisionTreeClassifier(max_depth=10, random_state=C.SEED)
    model.fit(X, y, sample_weight=None if w is None else normalize(w))
    return model
