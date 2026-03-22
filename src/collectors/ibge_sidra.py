# src/collectors/ibge_sidra.py
# ==============================================================
# Coleta dados de abate de bovinos via API SIDRA (IBGE).
# Tabela 1092 — Abate de animais nos estabelecimentos sob
# inspeção federal, estadual ou municipal.
#
# Variáveis coletadas:
#   284      → Animais abatidos (cabeças)
#   1000284  → Peso total das carcaças (toneladas)
#
# Filtros aplicados:
#   c12716/allxt  → todas as categorias de bovinos
#   c18/56        → bovinos
#   c12529/111738 → vacas
#
# Documentação SIDRA: https://apisidra.ibge.gov.br/
# ==============================================================

import requests
import pandas as pd
from config.settings import STATE, DATE_RANGE, SIDRA_URL


def _build_url() -> str:
    return SIDRA_URL.format(state_code=STATE["ibge_code"])


def _parse_sidra_response(data: list[dict]) -> pd.DataFrame:
    """
    Converte a resposta JSON do SIDRA em DataFrame.
    A primeira linha do JSON é o cabeçalho de metadados.
    """
    if not data or len(data) < 2:
        raise ValueError("Resposta SIDRA vazia ou inválida.")

    headers = data[0]
    records = data[1:]

    df = pd.DataFrame(records)
    df = df.rename(columns={
        "D3C": "periodo",      # Trimestre (ex: "201001")
        "V":   "valor",
        "D2N": "variavel",
    })

    # Mantém apenas colunas relevantes
    df = df[["periodo", "variavel", "valor"]].copy()

    # Filtra valores não numéricos (SIDRA retorna "-" para ausência)
    df = df[df["valor"].str.match(r"^[\d,.-]+$", na=False)]
    df["valor"] = df["valor"].str.replace(",", ".").astype(float)

    # Converte período trimestral (YYYYTT) para data
    # SIDRA usa formato: 201001 = 1º trimestre 2010
    def trimestre_to_date(s: str) -> pd.Timestamp:
        year = int(s[:4])
        quarter = int(s[4:])
        month = (quarter - 1) * 3 + 1
        return pd.Timestamp(year=year, month=month, day=1)

    df["data"] = df["periodo"].apply(trimestre_to_date)

    # Pivot: uma coluna por variável
    df_pivot = df.pivot_table(
        index="data",
        columns="variavel",
        values="valor",
        aggfunc="first",
    )

    # Renomeia colunas para nomes padronizados
    rename = {}
    for col in df_pivot.columns:
        if "cabeças" in col.lower() or "abatidos" in col.lower():
            rename[col] = "abate_cabecas"
        elif "peso" in col.lower() or "carcaça" in col.lower():
            rename[col] = "abate_peso_ton"

    df_pivot = df_pivot.rename(columns=rename)
    df_pivot.index.name = "data"

    return df_pivot


def _expand_to_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Expande dados trimestrais para frequência diária via forward fill.
    Justificativa: abate trimestral é condição estrutural do período —
    não varia dia a dia. Forward fill é metodologicamente adequado.
    """
    daily_index = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(daily_index, method="ffill")
    df.index.name = "data"
    return df


def load_sidra() -> pd.DataFrame:
    """
    Coleta e retorna dados de abate de bovinos do SIDRA em frequência diária.

    Retorna
    -------
    pd.DataFrame com colunas:
        abate_cabecas  : número de animais abatidos no trimestre
        abate_peso_ton : peso total das carcaças em toneladas
    """
    url = _build_url()

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Falha ao conectar com SIDRA: {e}\n"
            f"URL tentada: {url}"
        )

    data = response.json()
    df = _parse_sidra_response(data)

    # Filtra pelo período configurado
    start = DATE_RANGE["start"]
    end   = DATE_RANGE["end"]
    df = df.loc[start:end]

    df = _expand_to_daily(df, start, end)

    return df


if __name__ == "__main__":
    df = load_sidra()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}")