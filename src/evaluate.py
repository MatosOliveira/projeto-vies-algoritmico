"""
evaluate.py - desempenho global e métricas de fairness por grupo (ponderadas e com IC).

Todas as métricas usam UMA única predição por modelo (mesmo y_pred para desempenho e
fairness) e os pesos amostrais do teste. Grupos com pouco suporte ficam fora do resumo.
"""
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

import config as C


# ------------------------------------------------------------------ desempenho global
def performance(y, yp, w, score=None):
    out = {
        "accuracy": accuracy_score(y, yp, sample_weight=w),
        "precision": precision_score(y, yp, sample_weight=w, zero_division=0),
        "recall": recall_score(y, yp, sample_weight=w, zero_division=0),
        "f1": f1_score(y, yp, sample_weight=w, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, yp, sample_weight=w),
        "accuracy_sem_peso": accuracy_score(y, yp),
        "recall_sem_peso": recall_score(y, yp, zero_division=0),
        "f1_sem_peso": f1_score(y, yp, zero_division=0),
        "taxa_selecao": float(np.average(yp, weights=w)),
        "prevalencia": float(np.average(y, weights=w)),
    }
    out["auc"] = roc_auc_score(y, score, sample_weight=w) if score is not None else np.nan
    return out


# ------------------------------------------------------------------ métricas por grupo
def _div(a, b):
    out = np.full(len(a), np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _stats(c, G, y, yp, w, m=None):
    mm = np.ones(len(y)) if m is None else m
    ww = w * mm
    bc = lambda v: np.bincount(c, weights=v, minlength=G)
    n, npos = bc(mm), bc(y * mm)
    W, Wpos = bc(ww), bc(ww * y)
    Wneg = W - Wpos
    tp, fp = bc(ww * y * yp), bc(ww * (1 - y) * yp)
    nneg = n - npos
    tpr = _div(tp, Wpos)
    tpr[npos < C.MIN_POS_GRUPO] = np.nan
    fpr = _div(fp, Wneg)
    fpr[nneg < C.MIN_NEG_GRUPO] = np.nan
    return {"n": n, "n_pos": npos, "n_neg": nneg, "pop": W,
            "base_rate": _div(Wpos, W), "sel_rate": _div(bc(ww * yp), W),
            "tpr": tpr, "fpr": fpr, "acc": _div(bc(ww * (y == yp)), W)}


def _range(v):
    v = v[~np.isnan(v)]
    return float(v.max() - v.min()) if len(v) >= 2 else np.nan


def _ratio(v):
    v = v[~np.isnan(v)]
    return float(v.min() / v.max()) if len(v) >= 2 and v.max() > 0 else np.nan


def _summary(st, include):
    sel = np.where(include & (st["n"] >= C.MIN_N_GRUPO), st["sel_rate"], np.nan)
    tpr = np.where(include, st["tpr"], np.nan)
    fpr = np.where(include, st["fpr"], np.nan)
    tpr_d, fpr_d = _range(tpr), _range(fpr)
    eo = np.nanmax([tpr_d, fpr_d]) if not (np.isnan(tpr_d) and np.isnan(fpr_d)) else np.nan
    return {"dp_diff": _range(sel), "dp_ratio": _ratio(sel), "tpr_diff": tpr_d,
            "fpr_diff": fpr_d, "eo_diff": float(eo), "base_rate_diff": _range(
                np.where(include & (st["n"] >= C.MIN_N_GRUPO), st["base_rate"], np.nan))}


def fairness(y, yp, w, groups, attr, n_boot=C.N_BOOT, seed=C.SEED):
    """Devolve (tabela por grupo, dicionário-resumo com IC 95% por bootstrap)."""
    y, yp, w = np.asarray(y, float), np.asarray(yp, float), np.asarray(w, float)
    codes, levels = pd.factorize(pd.Series(groups), sort=True)
    G = len(levels)
    excluded = set(C.EXCLUDE_FROM_SUMMARY.get(attr, []))
    include = np.array([lv not in excluded for lv in levels])

    st = _stats(codes, G, y, yp, w)
    table = pd.DataFrame({
        "eval_attr": attr, "group": levels, "n_teste": st["n"].astype(int),
        "n_pos": st["n_pos"].astype(int), "n_neg": st["n_neg"].astype(int),
        "pop_estimada": st["pop"], "base_rate": st["base_rate"],
        "sel_rate": st["sel_rate"], "tpr": st["tpr"], "fpr": st["fpr"],
        "accuracy": st["acc"], "in_summary": include & (st["n"] >= C.MIN_N_GRUPO),
    })
    summ = _summary(st, include)

    if n_boot and n_boot > 0:
        rng = np.random.default_rng([seed, zlib.crc32(attr.encode())])
        boots = {k: [] for k in summ}
        for _ in range(n_boot):
            m = rng.multinomial(len(y), np.full(len(y), 1.0 / len(y))).astype(float)
            bs = _summary(_stats(codes, G, y, yp, w, m), include)
            for k, v in bs.items():
                boots[k].append(v)
        for k, vals in boots.items():
            vals = np.array(vals, float)
            vals = vals[~np.isnan(vals)]
            summ[k + "_lo"] = float(np.percentile(vals, 2.5)) if len(vals) > 20 else np.nan
            summ[k + "_hi"] = float(np.percentile(vals, 97.5)) if len(vals) > 20 else np.nan
    summ["n_grupos_resumo"] = int((include & (st["n"] >= C.MIN_N_GRUPO)).sum())
    summ["n_grupos_tpr"] = int((~np.isnan(np.where(include, st["tpr"], np.nan))).sum())
    return table, summ


# ------------------------------------------------------------------ diagnóstico por Q023
def q023_diagnostic(yp, w, letters, renda):
    """
    Proporção (ponderada) classificada como 'privada' por letra de Q023, no total e por
    faixa de renda. Para D e E equivale ao TPR dentro da letra; para A, B e C, à taxa de
    falsos positivos dentro da letra (no alvo principal).
    """
    df = pd.DataFrame({"letra": np.asarray(letters), "renda": np.asarray(renda),
                       "yp": np.asarray(yp, float), "w": np.asarray(w, float)})
    rows = []
    for renda_grupo, d in [("todas", df)] + list(df.groupby("renda")):
        for letra, g in d.groupby("letra"):
            rows.append({
                "letra": letra, "trajetoria": C.Q023_LABELS.get(letra, letra),
                "renda_grupo": renda_grupo, "n_teste": len(g),
                "pop_estimada": float(g["w"].sum()),
                "prop_prevista_privada": float(np.average(g["yp"], weights=g["w"])),
                "confiavel": len(g) >= C.MIN_POS_GRUPO,
            })
    return pd.DataFrame(rows)


def q023_distribution(letters, w, sens):
    """Composição (ponderada) das letras de Q023 dentro de cada grupo sensível."""
    rows = []
    base = pd.DataFrame({"letra": np.asarray(letters), "w": np.asarray(w, float)})
    for attr in ["TOTAL"] + list(sens.columns):
        groups = pd.Series("total", index=base.index) if attr == "TOTAL" else sens[attr]
        tab = base.assign(grupo=np.asarray(groups)).pivot_table(
            index="grupo", columns="letra", values="w", aggfunc="sum", fill_value=0.0)
        n = base.assign(grupo=np.asarray(groups)).groupby("grupo").size()
        share = tab.div(tab.sum(axis=1), axis=0)
        for grupo, r in share.iterrows():
            rows.append({"atributo": attr, "grupo": grupo, "n_amostra": int(n[grupo]),
                         **{f"pct_{c}": float(r[c]) for c in share.columns}})
    return pd.DataFrame(rows)
