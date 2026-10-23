"""
main.py - pipeline de análise de viés: baselines, mitigação e avaliação por ano.

Exemplos (a partir de src/):
    python main.py --year 2024                              # cenário A, alvo principal
    python main.py --year 2024 2025 --scenario A A_sem_sens B
    python main.py --year 2024 --target estrito --skip-reductions   # rápido
    python main.py --year 2024 --target principal estrito sem_bolsa
    python main.py --year 2024 --only-baseline

Saídas em results/runs/{ano}_{cenário}_{alvo}/:
    desempenho.csv        métricas globais (ponderadas) de cada experimento
    fairness_grupos.csv   taxa de seleção, TPR, FPR etc. por grupo e atributo
    fairness_resumo.csv   dp_diff, dp_ratio (DI), tpr/fpr/eo_diff com IC 95% (bootstrap)
    diagnostico_q023.csv  % classificado como privada por letra de Q023 (total e por renda)
    q023_por_grupo.csv    composição das letras de Q023 em cada grupo (descritivo)
    meta.json             parâmetros e tamanhos de cada partição
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config as C
import evaluate as E
import mitigation as M
import preprocessing as P
import train as T


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(year, scenario, target, args):
    run_id = f"{year}_{scenario}_{target}"
    out_dir = os.path.join(C.RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    log(f"=== {run_id} ===")

    df = P.load_year(year)
    X, y, sens, w, cat, num, q023 = P.build_dataset(df, scenario, target)
    fit, cal, te = P.split_indices(y, sens, args.seed)
    log(f"registros: {len(y):,} | treino {len(fit):,} | calibração {len(cal):,} "
        f"| teste {len(te):,} | positivos {int(y.sum()):,}")

    pre = P.make_preprocessor(cat, num)
    Xf = pre.fit_transform(X.iloc[fit]).astype(np.float64)       # ajustado só no treino
    Xc = pre.transform(X.iloc[cal]).astype(np.float64)
    Xt = pre.transform(X.iloc[te]).astype(np.float64)
    yf, yc, yt = y.values[fit], y.values[cal], y.values[te]
    wf = np.ones(len(fit)) if args.no_weights else w.values[fit]
    wt = np.ones(len(te)) if args.no_weights else w.values[te]
    log(f"features após codificação: {Xf.shape[1]}")

    experiments = []   # dicts com a predição de teste já calculada (uma única vez)

    def add(model, technique, attr, constraint, yp, score=None):
        experiments.append({"model": model, "technique": technique, "mitigated_attr": attr,
                            "constraint": constraint, "yp": np.asarray(yp).astype(int),
                            "score": score})

    # --- baselines
    lr = T.fit_logistic(Xf, yf, wf)
    tree = T.fit_tree(Xf, yf, wf)
    add("logistic", "baseline", "-", "-", lr.predict(Xt), lr.predict_proba(Xt)[:, 1])
    add("tree", "baseline", "-", "-", tree.predict(Xt), tree.predict_proba(Xt)[:, 1])
    log("baselines treinados")

    # --- mitigação (somente regressão logística)
    if not args.only_baseline:
        for attr in args.attrs:
            af, ac, at = sens[attr].values[fit], sens[attr].values[cal], sens[attr].values[te]

            t0 = time.time()
            rw = M.fit_reweighing(Xf, yf, af, wf)
            add("logistic", "reweighing", attr, "pre", rw.predict(Xt), rw.predict_proba(Xt)[:, 1])
            log(f"{attr}: reweighing ({time.time() - t0:.0f}s)")

            for cons, tag in [("equalized_odds", "threshold_eo"),
                              ("demographic_parity", "threshold_dp")]:
                opt = M.fit_threshold(lr, Xc, yc, ac, cons)
                add("logistic", tag, attr, cons, M.predict_threshold(opt, Xt, at, args.seed))
            log(f"{attr}: threshold (EO e DP)")

            if not args.skip_reductions:
                idx = np.arange(len(fit))
                if args.reductions_n and args.reductions_n < len(idx):
                    idx = np.random.default_rng(args.seed).choice(idx, args.reductions_n,
                                                                   replace=False)
                cons_list = [("demographic_parity", "reductions_dp")]
                if args.reductions_eo:
                    cons_list.append(("equalized_odds", "reductions_eo"))
                for cons, tag in cons_list:
                    t0 = time.time()
                    eg = M.fit_reductions(Xf[idx], yf[idx], af[idx], wf[idx], cons,
                                          args.reductions_max_iter)
                    add("logistic", tag, attr, cons, M.predict_reductions(eg, Xt, args.seed))
                    log(f"{attr}: {tag} ({time.time() - t0:.0f}s)")

    # --- composição de Q023 por grupo (amostra inteira, ponderada) - parte descritiva
    E.q023_distribution(q023.values, w.values, sens).to_csv(
        os.path.join(out_dir, "q023_por_grupo.csv"), index=False)

    # --- avaliação (cada modelo em TODOS os atributos)
    perf_rows, group_rows, summ_rows, diag_rows = [], [], [], []
    for exp in experiments:
        key = {"run_id": run_id, "year": year, "scenario": scenario, "target": target,
               "model": exp["model"], "technique": exp["technique"],
               "mitigated_attr": exp["mitigated_attr"], "constraint": exp["constraint"]}
        perf_rows.append({**key, **E.performance(yt, exp["yp"], wt, exp["score"])})
        for attr in C.EVAL_ATTRS:
            table, summ = E.fairness(yt, exp["yp"], wt, sens[attr].values[te], attr,
                                     n_boot=args.n_boot, seed=args.seed)
            group_rows.append(table.assign(**key))
            summ_rows.append({**key, "eval_attr": attr, **summ})
        diag = E.q023_diagnostic(exp["yp"], wt, q023.values[te], sens["RENDA_GRUPO"].values[te])
        diag_rows.append(diag.assign(**key))
    log("avaliação concluída")

    pd.DataFrame(perf_rows).to_csv(os.path.join(out_dir, "desempenho.csv"), index=False)
    pd.concat(group_rows, ignore_index=True).to_csv(
        os.path.join(out_dir, "fairness_grupos.csv"), index=False)
    pd.DataFrame(summ_rows).to_csv(os.path.join(out_dir, "fairness_resumo.csv"), index=False)
    pd.concat(diag_rows, ignore_index=True).to_csv(
        os.path.join(out_dir, "diagnostico_q023.csv"), index=False)
    meta = {"run_id": run_id, "scenario": C.SCENARIOS[scenario], "n_total": int(len(y)),
            "n_treino": int(len(fit)), "n_calibracao": int(len(cal)), "n_teste": int(len(te)),
            "n_features_codificadas": int(Xf.shape[1]), "pesos_amostrais": not args.no_weights,
            "args": {k: v for k, v in vars(args).items()}}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    log(f"salvo em {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--year", nargs="+", type=int, default=[2024])
    p.add_argument("--scenario", nargs="+", default=["A"], choices=list(C.SCENARIOS))
    p.add_argument("--target", nargs="+", default=["principal"], choices=list(C.TARGETS))
    p.add_argument("--attrs", nargs="+", default=C.MITIGATION_ATTRS,
                   help="atributos a mitigar (padrão: sexo, raça, região, renda)")
    p.add_argument("--include-uf", action="store_true", help="mitigar também por UF")
    p.add_argument("--only-baseline", action="store_true")
    p.add_argument("--skip-reductions", action="store_true", help="pula o ExponentiatedGradient")
    p.add_argument("--reductions-eo", action="store_true", help="reductions também com EO")
    p.add_argument("--reductions-max-iter", type=int, default=50)
    p.add_argument("--reductions-n", type=int, default=0,
                   help="subamostra do treino só para reductions (0 = usa tudo)")
    p.add_argument("--no-weights", action="store_true", help="ignora PESO_AMOSTRAL")
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--seed", type=int, default=C.SEED)
    args = p.parse_args()
    if args.include_uf and "SG_UF_PROVA" not in args.attrs:
        args.attrs = list(args.attrs) + ["SG_UF_PROVA"]
    return args


if __name__ == "__main__":
    a = parse_args()
    for yr in a.year:
        for sc in a.scenario:
            for tg in a.target:
                run_one(yr, sc, tg, a)
    log("Pipeline concluído.")
