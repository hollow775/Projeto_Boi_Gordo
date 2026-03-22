# src/collectors/cepea.py
# ==============================================================
# Leitura das planilhas manuais exportadas do CEPEA.
# O CEPEA não possui API pública — a coleta é via download manual
# em https://www.cepea.esalq.usp.br/br/indicador/
#
# Planilhas esperadas (exportar como .xlsx do site):
#   - Boi gordo (R$/arroba)
#   - Bezerro  (R$/cabeça)
#   - Milho    (R$/saca 60kg)
#
# Estrutura esperada das planilhas CEPEA:
#   - Linha 0-2: cabeçalho institucional (ignorado)
#   - Coluna 0: Data  (dd/mm/aaaa)
#   - Coluna 1: À Vista (valor que usaremos)
# ==============================================================

import pandas as pd
from pathlib import Path
from config.settings import STATE, DATE_RANGE


# Mapeamento: nome do produto → nome da coluna no DataFrame final
PRODUCT_MAP = {
    "boi_gordo": "preco_boi_gordo",
    "bezerro":   "preco_bezerro",
    "milho":     "preco_milho",
}


def _read_cepea_sheet(filepath: str | Path, column_name: str) -> pd.Series:
    """
    Lê uma planilha CEPEA e retorna uma Series com índice DatetimeIndex diário.

    Parâmetros
    ----------
    filepath    : caminho para o arquivo .xlsx
    column_name : nome da coluna resultante

    Retorna
    -------
    pd.Series com índice de datas e nome = column_name
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Planilha não encontrada: {filepath}\n"
            "Baixe o arquivo em https://www.cepea.esalq.usp.br/br/indicador/ "
            "e salve no caminho configurado em config/settings.py"
        )

    df = pd.read_excel(
        filepath,
        header=2,           # linha 3 é o cabeçalho real nas planilhas CEPEA
        usecols=[0, 1],     # data e valor à vista
        decimal=",",
        thousands=".",
    )

    df.columns = ["data", "valor"]

    # Remove linhas sem data ou valor
    df = df.dropna(subset=["data", "valor"])

    # Converte data
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["data"])

    # Garante valor numérico (CEPEA às vezes exporta string com vírgula)
    df["valor"] = (
        df["valor"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    series = df.set_index("data")["valor"].rename(column_name)
    series.index = pd.DatetimeIndex(series.index)
    series = series.sort_index()

    return series


def _reindex_to_daily(series: pd.Series, start: str, end: str) -> pd.Series:
    """
    Expande a série para todos os dias corridos do período e
    interpola linearmente os dias sem cotação (fins de semana/feriados).
    """
    full_index = pd.date_range(start=start, end=end, freq="D")
    series = series.reindex(full_index)
    series = series.interpolate(method="linear")
    return series


def load_cepea(
    boi_file:     str | Path | None = None,
    bezerro_file: str | Path | None = None,
    milho_file:   str | Path | None = None,
) -> pd.DataFrame:
    """
    Carrega e combina as três planilhas CEPEA.

    Se o caminho não for fornecido, usa os padrões definidos em settings.py.
    Retorna DataFrame diário com colunas:
        preco_boi_gordo, preco_bezerro, preco_milho

    Parâmetros
    ----------
    boi_file     : caminho para planilha do boi gordo
    bezerro_file : caminho para planilha do bezerro
    milho_file   : caminho para planilha do milho
    """
    base = Path(STATE["cepea_file"]).parent

    paths = {
        "boi_gordo": boi_file     or base / "cepea_boi_gordo.xlsx",
        "bezerro":   bezerro_file or base / "cepea_bezerro.xlsx",
        "milho":     milho_file   or base / "cepea_milho.xlsx",
    }

    start = DATE_RANGE["start"]
    end   = DATE_RANGE["end"]

    series_list = []
    for product, filepath in paths.items():
        col_name = PRODUCT_MAP[product]
        s = _read_cepea_sheet(filepath, col_name)
        s = _reindex_to_daily(s, start, end)
        series_list.append(s)

    df = pd.concat(series_list, axis=1)
    df.index.name = "data"

    return df


if __name__ == "__main__":
    df = load_cepea()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}")