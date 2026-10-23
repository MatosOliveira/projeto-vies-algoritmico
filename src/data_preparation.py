"""
data_preparation.py
===================
Etapa de limpeza, amostragem estratificada e enriquecimento contextual dos
microdados do ENEM (2024 e 2025), SEM carregar os arquivos completos em memória.

Uso (a partir da pasta src/):
    python data_preparation.py                       # 2024 e 2025, ~200 mil registros
    python data_preparation.py --years 2024          # só 2024
    python data_preparation.py --target-n 50000      # amostra menor (teste rápido)
    python data_preparation.py --years 2025 --tag 2025   # não sobrescreve as saídas de 2024

Entradas (em DATA_DIR; o ano e o tipo são detectados pelo nome do arquivo):
    *PARTICIPANTES*2024*.csv , *RESULTADOS*2024*.csv   (idem 2025)

Saídas:
    data/processed/enem_amostra.parquet          amostra tratada (entrada do pipeline)
    data/processed/contexto_municipal.csv        médias de notas por município e UF (auditoria)
    results/tables/cleaning_log.csv              funil de limpeza (registros removidos por etapa)
    results/tables/amostra_alocacao_estratos.csv população, amostra e peso por estrato
    results/tables/amostra_resumo_grupos.csv     n e prevalência do alvo por grupo sensível

Decisões metodológicas (documentar no Cap. 6 do TCC):
    * Os arquivos de participantes e de resultados NÃO possuem chave comum (NU_SEQUENCIAL não
      se relaciona com NU_INSCRICAO, conforme nota 1 do dicionário do INEP). Por isso as notas
      entram apenas como CONTEXTO AGREGADO por município (CO_MUNICIPIO_PROVA), nunca por pessoa.
      A média municipal descreve o local, não o indivíduo (falácia ecológica).
    * Municípios com menos de MIN_PRESENTES_MUN presentes usam a média da UF (fallback).
    * Faltantes (qualquer TP_PRESENCA_* != 1) ficam fora do cálculo das médias.
    * Atributos sensíveis NÃO são imputados: registros inválidos são removidos e contados no log.
      Raça/cor 0 ("não declarado") e 6 ("não dispõe") formam o grupo "nao_informada".
    * Compatibilidade 2024 x 2025 (dicionários conferidos): Q007 segue em múltiplos do salário
      mínimo (R$ 1.412 em 2024; R$ 1.518 em 2025), então as faixas por SM são comparáveis;
      TP_COR_RACA não tem mais a categoria 6 em 2025; Q022 é harmonizada (ver HARMONIZACAO).
    * Imputação das demais variáveis do questionário NÃO é feita aqui: deve ocorrer no pipeline,
      ajustada apenas no treino (evita vazamento).
    * Não se usa a participação percentual de escola privada do município como contexto, pois
      seria uma versão agregada do próprio alvo (Q023).
"""
import argparse
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from config import DATA_DIR, TABLES_DIR

# ----------------------------------------------------------------------------------------
# Parâmetros (podem ser alterados por linha de comando)
# ----------------------------------------------------------------------------------------
DEFAULT_YEARS = [2024, 2025]
DEFAULT_TARGET_N = 200_000
DEFAULT_MIN_POR_ESTRATO = 100        # piso por estrato (estratos menores entram inteiros)
DEFAULT_MIN_PRESENTES_MUN = 30       # mínimo de presentes para usar a média do município
DEFAULT_CHUNKSIZE = 300_000
DEFAULT_SEED = 42

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TMP_DIR = os.path.join(PROCESSED_DIR, "_tmp")

# ----------------------------------------------------------------------------------------
# Colunas e mapeamentos
# ----------------------------------------------------------------------------------------
QUESTIONARIO = [f"Q{i:03d}" for i in range(1, 24)]
PART_COLS = (
    ["NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ST_CONCLUSAO",
     "IN_TREINEIRO", "CO_MUNICIPIO_PROVA", "SG_UF_PROVA"] + QUESTIONARIO
)
RES_COLS = (
    ["CO_MUNICIPIO_PROVA", "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC",
     "TP_PRESENCA_MT", "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT",
     "TP_STATUS_REDACAO", "NU_NOTA_REDACAO"]
)
NOTAS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"]

RACA_GRUPO = {"1": "branca", "2": "preta", "3": "parda", "4": "amarela", "5": "indigena",
              "0": "nao_informada", "6": "nao_informada"}
REGIAO = {1: "N", 2: "NE", 3: "SE", 4: "S", 5: "CO"}   # 1º dígito do código do município

# Faixas de renda em múltiplos do salário mínimo (as letras de Q007 seguem múltiplos do SM;
# conferir se o questionário de 2025 mantém a mesma estrutura de faixas).
RENDA_GRUPO = {
    "A": "ate_1SM", "B": "ate_1SM",
    "C": "1_a_2SM", "D": "1_a_2SM",
    "E": "2_a_3SM", "F": "2_a_3SM",
    "G": "3_a_6SM", "H": "3_a_6SM", "I": "3_a_6SM",
    "J": "mais_6SM", "K": "mais_6SM", "L": "mais_6SM", "M": "mais_6SM",
    "N": "mais_6SM", "O": "mais_6SM", "P": "mais_6SM", "Q": "mais_6SM",
}

# Harmonização 2024 x 2025 (conferida nos dicionários do INEP):
#   Q022: em 2024 as categorias D ("três ou mais") e E ("quatro ou mais") se sobrepõem; em 2025
#   D = "três" e E = "quatro ou mais". Para comparar os anos, E é fundida em D ("três ou mais").
HARMONIZACAO = {"Q022": {"E": "D"}}

# Q023 (tipo de escola do ensino médio)
#   A -> publica | B, C -> mista (parte pública, parte privada) | D, E -> privada
TRAJETORIA = {"A": "publica", "B": "mista", "C": "mista", "D": "privada", "E": "privada"}
ALVO_PRINCIPAL = {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1}       # privada=1; pública+mista=0
ALVO_ESTRITO = {"A": 0, "D": 1, "E": 1}                        # mista (B, C) fica ausente

STRATA_COLS = ["NU_ANO", "TP_SEXO", "RACA_GRUPO", "REGIAO", "RENDA_GRUPO"]
NUMERIC_FEATURES = ["TP_FAIXA_ETARIA", "Q005"]

FUNNEL_ORDER = [
    "lidos",
    "treineiros_excluidos",
    "atributo_sensivel_invalido",
    "q023_invalida_ou_ausente",
    "q023_nao_frequentou_em",
]


# ----------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------
def find_file(data_dir, kind, year):
    """Localiza o CSV do tipo ('PARTICIPANTES' ou 'RESULTADOS') e ano pelo nome do arquivo."""
    matches = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".csv") and kind.lower() in f.lower() and str(year) in f
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Mais de um arquivo para {kind} {year}: {matches}")
    return os.path.join(data_dir, matches[0]) if matches else None


def read_chunks(path, wanted_cols, chunksize, as_str):
    """Lê o CSV em blocos, apenas com as colunas pedidas que existem no arquivo."""
    header = pd.read_csv(path, sep=";", encoding="latin1", nrows=0, index_col=False).columns
    missing = [c for c in wanted_cols if c not in header]
    if missing:
        raise KeyError(f"{os.path.basename(path)}: colunas ausentes {missing}. "
                       "Conferir o dicionário do ano.")
    kwargs = dict(sep=";", encoding="latin1", usecols=wanted_cols, chunksize=chunksize,
                  index_col=False)
    if as_str:
        kwargs["dtype"] = str
    return pd.read_csv(path, **kwargs)


def clean_str(series):
    """Remove espaços e transforma vazio em NaN (série de strings)."""
    s = series.astype("object").where(series.notna(), None)
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    return s.replace("", np.nan)


# ----------------------------------------------------------------------------------------
# 1) Participantes: limpeza em blocos
# ----------------------------------------------------------------------------------------
def clean_participantes_chunk(chunk, year, exclude_treineiros, funnel):
    df = chunk.copy()
    for c in df.columns:
        df[c] = clean_str(df[c])
    funnel[(year, "lidos")] += len(df)

    # 1) Treineiros
    if exclude_treineiros:
        keep = df["IN_TREINEIRO"].ne("1")
        funnel[(year, "treineiros_excluidos")] += int((~keep).sum())
        df = df[keep]

    # 2) Atributos sensíveis válidos (sem imputação)
    mun = pd.to_numeric(df["CO_MUNICIPIO_PROVA"], errors="coerce")
    valid = (
        df["TP_SEXO"].isin(["M", "F"])
        & df["TP_COR_RACA"].isin(list(RACA_GRUPO))
        & df["SG_UF_PROVA"].str.fullmatch(r"[A-Z]{2}", na=False)
        & df["Q007"].isin(list(RENDA_GRUPO))
        & mun.between(1_000_000, 5_999_999)
    )
    funnel[(year, "atributo_sensivel_invalido")] += int((~valid).sum())
    df, mun = df[valid], mun[valid]

    # 3) Q023 válida
    q023_ok = df["Q023"].isin(list("ABCDEF"))
    funnel[(year, "q023_invalida_ou_ausente")] += int((~q023_ok).sum())
    df, mun = df[q023_ok], mun[q023_ok]

    # 4) Q023 = F (não frequentou o ensino médio): sem informação para o alvo
    freq = df["Q023"].ne("F")
    funnel[(year, "q023_nao_frequentou_em")] += int((~freq).sum())
    df, mun = df[freq], mun[freq]

    # Colunas derivadas
    out = df.copy()
    for col, mapping in HARMONIZACAO.items():
        out[col] = out[col].replace(mapping)
    out["NU_ANO"] = np.int16(year)
    out["CO_MUNICIPIO_PROVA"] = mun.astype("int64")
    out["REGIAO"] = (out["CO_MUNICIPIO_PROVA"] // 1_000_000).map(REGIAO)
    out["RACA_GRUPO"] = out["TP_COR_RACA"].map(RACA_GRUPO)
    out["RENDA_GRUPO"] = out["Q007"].map(RENDA_GRUPO)
    out["ESCOLA_TRAJETORIA"] = out["Q023"].map(TRAJETORIA)
    out["ALVO_PRIVADA"] = out["Q023"].map(ALVO_PRINCIPAL).astype("int8")
    out["ALVO_PRIVADA_ESTRITO"] = out["Q023"].map(ALVO_ESTRITO).astype("Int8")
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    out["NU_INSCRICAO"] = pd.to_numeric(out["NU_INSCRICAO"], errors="coerce").astype("Int64")
    return out


# ----------------------------------------------------------------------------------------
# 2) Alocação da amostra estratificada
# ----------------------------------------------------------------------------------------
def allocate(strata_counts, target_n, min_por_estrato):
    """
    Define a taxa de sorteio q_h de cada estrato: n_h = min(N_h, max(piso, p * N_h)),
    com p escolhido por bisseção para que sum(n_h) ~ target_n.
    """
    alloc = strata_counts.rename("N_h").reset_index()
    N = alloc["N_h"].to_numpy(dtype=float)
    floor = np.minimum(N, float(min_por_estrato))

    def total(p):
        return np.minimum(N, np.maximum(floor, p * N)).sum()

    if N.sum() <= target_n:
        p = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if total(mid) > target_n:
                hi = mid
            else:
                lo = mid
        p = lo
    n_h = np.minimum(N, np.maximum(floor, p * N))
    if p == 0.0 or floor.sum() > target_n:
        print(f"  [aviso] a soma dos pisos por estrato ({floor.sum():,.0f}) excede a meta "
              f"({target_n:,}). A amostra ficará acima da meta: reduza --min-por-estrato "
              "ou aumente --target-n.")
    alloc["n_alvo"] = np.round(n_h).astype(int)
    alloc["q"] = np.clip(n_h / N, 0, 1)
    alloc["peso"] = 1.0 / alloc["q"]
    print(f"  Fração proporcional p = {p:.5f}; amostra esperada ~ {n_h.sum():,.0f} "
          f"de {N.sum():,.0f} registros válidos ({len(alloc)} estratos).")
    return alloc


# ----------------------------------------------------------------------------------------
# 3) Contexto municipal a partir dos Resultados (sem ligação individual)
# ----------------------------------------------------------------------------------------
def build_context(years, data_dir, chunksize, min_presentes):
    parts_mun, parts_uf = [], []
    for year in years:
        path = find_file(data_dir, "RESULTADOS", year)
        if path is None:
            print(f"  [aviso] arquivo de RESULTADOS {year} não encontrado: sem contexto de {year}.")
            continue
        print(f"  Lendo {os.path.basename(path)} (apenas colunas necessárias)...")
        for chunk in read_chunks(path, RES_COLS, chunksize, as_str=False):
            for c in ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
                      "TP_STATUS_REDACAO", "CO_MUNICIPIO_PROVA"] + NOTAS + ["NU_NOTA_REDACAO"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            # presentes nas quatro provas objetivas (exclui faltantes e eliminados)
            present = (
                (chunk["TP_PRESENCA_CN"] == 1) & (chunk["TP_PRESENCA_CH"] == 1)
                & (chunk["TP_PRESENCA_LC"] == 1) & (chunk["TP_PRESENCA_MT"] == 1)
                & chunk[NOTAS].notna().all(axis=1) & chunk["CO_MUNICIPIO_PROVA"].notna()
            )
            d = chunk[present].copy()
            d["NU_ANO"] = year
            d["CO_MUNICIPIO_PROVA"] = d["CO_MUNICIPIO_PROVA"].astype("int64")
            d["CO_UF"] = d["CO_MUNICIPIO_PROVA"] // 100_000
            red_ok = (d["TP_STATUS_REDACAO"] == 1) & d["NU_NOTA_REDACAO"].notna()
            d["RED"] = d["NU_NOTA_REDACAO"].where(red_ok)
            d["RED_N"] = red_ok.astype(int)
            d["RED"] = d["RED"].fillna(0.0)
            d["N"] = 1
            sums = ["N", "RED", "RED_N"] + NOTAS
            parts_mun.append(d.groupby(["NU_ANO", "CO_MUNICIPIO_PROVA"])[sums].sum())
            parts_uf.append(d.groupby(["NU_ANO", "CO_UF"])[sums].sum())

    if not parts_mun:
        return None, None

    def finalize(parts):
        agg = pd.concat(parts).groupby(level=[0, 1]).sum()
        out = pd.DataFrame(index=agg.index)
        out["CTX_N_PRESENTES"] = agg["N"].astype(int)
        for c in NOTAS:
            out["CTX_" + c] = agg[c] / agg["N"]
        out["CTX_NU_NOTA_REDACAO"] = np.where(agg["RED_N"] > 0, agg["RED"] / agg["RED_N"], np.nan)
        cols = ["CTX_" + c for c in NOTAS] + ["CTX_NU_NOTA_REDACAO"]
        out["CTX_NOTA_MEDIA"] = out[cols].mean(axis=1)
        return out.reset_index()

    return finalize(parts_mun), finalize(parts_uf)


def attach_context(sample, ctx_mun, ctx_uf, min_presentes):
    """Anexa o contexto: município se n >= mínimo; caso contrário, média da UF."""
    ctx_cols = [c for c in ctx_mun.columns if c.startswith("CTX_") and c != "CTX_N_PRESENTES"]
    mun = ctx_mun.copy()
    mun.loc[mun["CTX_N_PRESENTES"] < min_presentes, ctx_cols + ["CTX_N_PRESENTES"]] = np.nan
    mun = mun.rename(columns={c: c + "__mun" for c in ctx_cols + ["CTX_N_PRESENTES"]})
    uf = ctx_uf.rename(columns={c: c + "__uf" for c in ctx_cols + ["CTX_N_PRESENTES"]})

    sample = sample.copy()
    sample["CO_UF"] = sample["CO_MUNICIPIO_PROVA"] // 100_000
    sample = sample.merge(mun, on=["NU_ANO", "CO_MUNICIPIO_PROVA"], how="left")
    sample = sample.merge(uf, on=["NU_ANO", "CO_UF"], how="left")

    use_mun = sample["CTX_N_PRESENTES__mun"].notna()
    for c in ctx_cols + ["CTX_N_PRESENTES"]:
        sample[c] = sample[c + "__mun"].where(use_mun, sample[c + "__uf"])
    sample["CTX_FONTE"] = np.where(use_mun, "municipio",
                                   np.where(sample["CTX_NOTA_MEDIA"].notna(), "uf", "ausente"))
    return sample.drop(columns=[c for c in sample.columns if c.endswith(("__mun", "__uf"))]
                       + ["CO_UF"])


# ----------------------------------------------------------------------------------------
# Relatórios
# ----------------------------------------------------------------------------------------
def funnel_table(funnel, years):
    rows = []
    for year in years:
        remaining = funnel.get((year, "lidos"), 0)
        rows.append({"ano": year, "etapa": "lidos", "removidos": 0, "restantes": remaining})
        for step in FUNNEL_ORDER[1:]:
            removed = funnel.get((year, step), 0)
            remaining -= removed
            rows.append({"ano": year, "etapa": step, "removidos": removed,
                         "restantes": remaining})
    return pd.DataFrame(rows)


def group_summary(sample):
    rows = []
    for col in ["TP_SEXO", "RACA_GRUPO", "REGIAO", "SG_UF_PROVA", "RENDA_GRUPO", "NU_ANO"]:
        for level, g in sample.groupby(col, observed=True):
            w = g["PESO_AMOSTRAL"]
            rows.append({
                "atributo": col, "grupo": level,
                "n_amostra": len(g), "n_positivos_amostra": int(g["ALVO_PRIVADA"].sum()),
                "prevalencia_amostra": g["ALVO_PRIVADA"].mean(),
                "n_populacao_estimado": float(w.sum()),
                "prevalencia_ponderada": float(np.average(g["ALVO_PRIVADA"], weights=w)),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------------------------
def run(args):
    data_dir = args.data_dir
    sfx = f"_{args.tag}" if args.tag else ""   # sufixo opcional para não sobrescrever saídas
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    funnel = defaultdict(int)
    strata_counts = None
    part_files = []
    dup_check = []
    years_found = []

    # --- Passo 1: ler participantes em blocos, limpar e guardar blocos compactos ---------
    for year in args.years:
        path = find_file(data_dir, "PARTICIPANTES", year)
        if path is None:
            print(f"[aviso] arquivo de PARTICIPANTES {year} não encontrado em {data_dir}; "
                  "ano ignorado.")
            continue
        years_found.append(year)
        print(f"[1/4] Limpando {os.path.basename(path)} em blocos de {args.chunksize:,}...")
        for i, chunk in enumerate(read_chunks(path, PART_COLS, args.chunksize, as_str=True)):
            clean = clean_participantes_chunk(chunk, year, not args.keep_treineiros, funnel)
            if clean.empty:
                continue
            counts = clean.groupby(STRATA_COLS, observed=True).size()
            strata_counts = counts if strata_counts is None else strata_counts.add(
                counts, fill_value=0)
            dup_check.append(clean[["NU_ANO", "NU_INSCRICAO"]])
            f = os.path.join(TMP_DIR, f"part_{year}_{i:04d}.parquet")
            clean.to_parquet(f, index=False)
            part_files.append(f)
            print(f"      bloco {i}: {len(chunk):,} lidos -> {len(clean):,} mantidos")

    if not part_files:
        raise SystemExit("Nenhum registro válido encontrado. Verifique DATA_DIR e os nomes dos arquivos.")

    # Duplicidade de NU_INSCRICAO dentro do ano (verificação sobre toda a base limpa)
    ids = pd.concat(dup_check)
    n_dup = int(ids.duplicated().sum())
    print(f"[2/4] Verificação de duplicidade de NU_INSCRICAO por ano: {n_dup:,} duplicados.")
    del ids, dup_check

    # --- Passo 2: alocação ----------------------------------------------------------------
    strata_counts = strata_counts.astype(int)
    alloc = allocate(strata_counts, args.target_n, args.min_por_estrato)

    # --- Passo 3: sorteio (Poisson estratificado, com peso = 1/q) -------------------------
    print("[3/4] Sorteando a amostra estratificada...")
    sampled = []
    for idx, f in enumerate(part_files):
        df = pd.read_parquet(f)
        df = df.merge(alloc[STRATA_COLS + ["q", "peso"]], on=STRATA_COLS, how="left")
        rng = np.random.default_rng([args.seed, idx])
        keep = rng.random(len(df)) < df["q"].to_numpy()
        sampled.append(df[keep].drop(columns=["q"]))
    sample = pd.concat(sampled, ignore_index=True).rename(columns={"peso": "PESO_AMOSTRAL"})

    # remove duplicados de inscrição eventualmente sorteados e descarta o identificador
    before = len(sample)
    sample = sample.drop_duplicates(subset=["NU_ANO", "NU_INSCRICAO"]).drop(columns="NU_INSCRICAO")
    print(f"      amostra: {before:,} registros ({before - len(sample)} duplicados removidos).")

    # --- Passo 4: contexto municipal (resultados) -----------------------------------------
    ctx_mun = ctx_uf = None
    if not args.skip_contexto:
        print("[4/4] Construindo contexto municipal a partir dos resultados...")
        ctx_mun, ctx_uf = build_context(years_found, data_dir, args.chunksize,
                                        args.min_presentes_mun)
    if ctx_mun is not None:
        ctx_mun.to_csv(os.path.join(PROCESSED_DIR, f"contexto_municipal{sfx}.csv"), index=False)
        sample = attach_context(sample, ctx_mun, ctx_uf, args.min_presentes_mun)
        print("      fonte do contexto na amostra:\n"
              + sample["CTX_FONTE"].value_counts().to_string())
    else:
        print("[4/4] Sem contexto municipal (cenário B indisponível).")

    # tipos compactos
    cat_cols = [c for c in sample.columns
                if pd.api.types.is_object_dtype(sample[c])
                or pd.api.types.is_string_dtype(sample[c])]
    for c in cat_cols:
        sample[c] = sample[c].astype("category")

    # --- Saídas ---------------------------------------------------------------------------
    sample.to_parquet(os.path.join(PROCESSED_DIR, f"enem_amostra{sfx}.parquet"), index=False)

    funnel_df = funnel_table(funnel, years_found)
    funnel_df.to_csv(os.path.join(TABLES_DIR, f"cleaning_log{sfx}.csv"), index=False)

    realized = sample.groupby(STRATA_COLS, observed=True).size().rename("n_amostra").reset_index()
    alloc.merge(realized, on=STRATA_COLS, how="left").fillna({"n_amostra": 0}).to_csv(
        os.path.join(TABLES_DIR, f"amostra_alocacao_estratos{sfx}.csv"), index=False)
    group_summary(sample).to_csv(os.path.join(TABLES_DIR, f"amostra_resumo_grupos{sfx}.csv"),
                                 index=False)

    if not args.keep_tmp:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("\nFunil de limpeza:")
    print(funnel_df.to_string(index=False))
    print(f"\nAmostra final: {len(sample):,} registros, {sample.shape[1]} colunas "
          f"-> {os.path.join(PROCESSED_DIR, 'enem_amostra.parquet')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    p.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
    p.add_argument("--min-por-estrato", type=int, default=DEFAULT_MIN_POR_ESTRATO)
    p.add_argument("--min-presentes-mun", type=int, default=DEFAULT_MIN_PRESENTES_MUN)
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--keep-treineiros", action="store_true",
                   help="não excluir treineiros (padrão: excluir)")
    p.add_argument("--skip-contexto", action="store_true",
                   help="não ler os arquivos de resultados")
    p.add_argument("--tag", default="",
                   help="sufixo nos nomes dos arquivos de saída (ex.: --tag 2025 gera "
                        "enem_amostra_2025.parquet) para não sobrescrever execuções anteriores")
    p.add_argument("--keep-tmp", action="store_true", help="manter blocos temporários")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
