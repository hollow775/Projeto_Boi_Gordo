# src/collectors/cepea.py
# ==============================================================
# Leitura das planilhas manuais exportadas do CEPEA.
# Suporta tanto .xls quanto .xlsx — detecta automaticamente.
#
# Planilhas esperadas (exportar do site CEPEA):
#   - Boi gordo (R$/arroba)
#   - Bezerro  (R$/cabeca)
#   - Milho    (R$/saca 60kg)
#
# Estrutura real das planilhas CEPEA:
#   Linha 1: "INDICADOR DO BOI GORDO CEPEA/ESALQ"
#   Linha 2: (vazia)
#   Linha 3: "Fonte: Cepea"
#   Linha 4: "Data" | "A vista R$"   <- cabecalho real (header=3)
#   Linha 5: 23/07/1997 | 26.67      <- dados
# ==============================================================

import pandas as pd
from pathlib import Path
from config.settings import STATE, DATE_RANGE


PRODUCT_MAP = {
    "boi_gordo": "preco_boi_gordo",
    "bezerro":   "preco_bezerro",
    "milho":     "preco_milho",
}


def _resolve_path(filepath: Path) -> Path:
    """
    Retorna o caminho existente, tentando .xls e .xlsx automaticamente.
    """
    if filepath.exists():
        return filepath

    alt_suffix = ".xls" if filepath.suffix == ".xlsx" else ".xlsx"
    alt = filepath.with_suffix(alt_suffix)
    if alt.exists():
        return alt

    raise FileNotFoundError(
        f"Planilha nao encontrada: {filepath}\n"
        "Baixe o arquivo em https://www.cepea.esalq.usp.br/br/indicador/ "
        "e salve no caminho configurado em config/settings.py"
    )


def _read_cepea_sheet(filepath: str | Path, column_name: str) -> pd.Series:
    """
    Le uma planilha CEPEA e retorna uma Series com indice DatetimeIndex diario.

    Parametros
    ----------
    filepath    : caminho para o arquivo .xls ou .xlsx
    column_name : nome da coluna resultante

    Retorna
    -------
    pd.Series com indice de datas e nome = column_name
    """
    filepath = _resolve_path(Path(filepath))

    # Seleciona engine conforme extensao
    engine = "xlrd" if filepath.suffix == ".xls" else "openpyxl"

    df = pd.read_excel(
        filepath,
        header=3,           # linha 4 e o cabecalho real nas planilhas CEPEA
        usecols=[0, 1],     # data e valor a vista
        engine=engine,
    )

    df.columns = ["data", "valor"]

    # Remove linhas sem data ou valor
    df = df.dropna(subset=["data", "valor"])

    # Converte data
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["data"])

    # Converte valor para float
    # CEPEA exporta com ponto como decimal (ex: 26.67)
    # Nao usar thousands="." — interpreta ponto como milhar e corrompe os dados
    df["valor"] = (
        df["valor"]
        .astype(str)
        .str.strip()
        .str.replace(r"[^\d,\.]", "", regex=True)  # remove caracteres nao numericos
        .str.replace(",", ".", regex=False)           # normaliza separador decimal
        .astype(float)
    )

    series = df.set_index("data")["valor"].rename(column_name)
    series.index = pd.DatetimeIndex(series.index)
    series = series.sort_index()

    return series


def _reindex_to_daily(series: pd.Series, start: str, end: str) -> pd.Series:
    """
    Expande a serie para todos os dias corridos do periodo e
    interpola linearmente os dias sem cotacao (fins de semana/feriados).
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
    Carrega e combina as tres planilhas CEPEA.

    Aceita .xls ou .xlsx — detecta automaticamente.
    Se o caminho nao for fornecido, usa os padroes definidos em settings.py.

    Retorna DataFrame diario com colunas:
        preco_boi_gordo, preco_bezerro, preco_milho
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


def save_cepea_xlsx(output_path: str | None = None) -> None:
    """
    Salva a serie combinada do CEPEA em Excel para inspecao visual.
    Util para verificar se os valores estao sendo lidos corretamente.
    Inclui os dados do indice de inflação (Banco Central).
    """
    from config.settings import DATA_PROCESSED
    from src.collectors.base_deflacionaria import load_inflation_deflator
    
    df = load_cepea()

    try:
        df_deflator = load_inflation_deflator()
        df = df.join(df_deflator, how="left")
    except Exception as e:
        print(f"[cepea] Aviso: não foi possível anexar o índice de inflação: {e}")

    path = output_path or str(DATA_PROCESSED / "cepea_combinado_inspecao.xlsx")

    df_export = df.copy()
    df_export.index.name = "Data"
    
    col_mapping = {
        "preco_boi_gordo": "Preco Boi Gordo (R$/arroba)",
        "preco_bezerro":   "Preco Bezerro (R$/cabeca)",
        "preco_milho":     "Preco Milho (R$/saca 60kg)",
    }
    if "ipca_index" in df_export.columns:
        col_mapping["ipca_index"] = "Indice IPCA"
        
    df_export = df_export.rename(columns=col_mapping)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="CEPEA Combinado")
        df_export.describe().round(2).to_excel(writer, sheet_name="Estatisticas")

    print(f"[cepea] Arquivo de inspecao salvo em: {path}")