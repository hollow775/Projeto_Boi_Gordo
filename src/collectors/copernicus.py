# src/collectors/copernicus.py
# ==============================================================
# Coleta dados de precipitação via Copernicus Climate Data Store.
# Dataset: ERA5-Land Monthly Means
# Variável: total_precipitation (metros/mês)
#
# Pré-requisitos:
#   1. Cadastro em https://cds.climate.copernicus.eu/
#   2. Instalar biblioteca: pip install cdsapi
#   3. Criar arquivo ~/.cdsapirc com:
#        url: https://cds.climate.copernicus.eu/api/v2
#        key: <seu-uid>:<sua-api-key>
#
# Documentação ERA5-Land:
#   https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means
# ==============================================================

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import STATE, DATE_RANGE, CDS_DATASET, CDS_VARIABLE


def _build_year_month_lists(start: str, end: str) -> tuple[list, list]:
    dates = pd.date_range(start=start, end=end, freq="MS")
    years  = sorted(set(str(d.year) for d in dates))
    months = [str(m).zfill(2) for m in range(1, 13)]
    return years, months


def download_era5(force: bool = False) -> Path:
    """
    Baixa o arquivo ERA5-Land para a bounding box do estado configurado.
    O download é feito apenas uma vez (arquivo em cache em data/raw/).

    Parâmetros
    ----------
    force : se True, baixa novamente mesmo que o arquivo já exista.

    Retorna
    -------
    Path para o arquivo .nc baixado
    """
    output_path = Path(STATE["era5_file"])

    if output_path.exists() and not force:
        print(f"ERA5 já baixado: {output_path}. Use force=True para re-baixar.")
        return output_path

    bbox = STATE["copernicus_bbox"]  # [lon_min, lat_min, lon_max, lat_max]
    # CDS espera: [lat_max, lon_min, lat_min, lon_max] (Norte/Oeste/Sul/Leste)
    area = [bbox[3], bbox[0], bbox[1], bbox[2]]

    years, months = _build_year_month_lists(DATE_RANGE["start"], DATE_RANGE["end"])

    client = cdsapi.Client()
    client.retrieve(
        CDS_DATASET,
        {
            "variable":     CDS_VARIABLE,
            "product_type": "monthly_averaged_reanalysis",
            "year":         years,
            "month":        months,
            "time":         "00:00",
            "area":         area,
            "format":       "netcdf",
        },
        str(output_path),
    )

    print(f"ERA5 salvo em: {output_path}")
    return output_path


def _parse_era5(filepath: Path) -> pd.Series:
    """
    Lê o arquivo NetCDF e calcula a precipitação média mensal
    sobre toda a área do estado (média espacial).

    A variável ERA5 tp está em metros/hora acumulados.
    Multiplicamos por dias do mês * 24 para converter em mm/mês.
    """
    ds = xr.open_dataset(filepath)

    # ERA5-Land: variável 'tp' = total precipitation (m)
    tp = ds["tp"]

    # Média espacial sobre o bounding box
    tp_mean = tp.mean(dim=["latitude", "longitude"])

    df = tp_mean.to_dataframe(name="precipitacao_mm")
    df.index = pd.to_datetime(df.index)
    df.index.name = "data"

    # Converte metros para milímetros
    df["precipitacao_mm"] = df["precipitacao_mm"] * 1000

    ds.close()
    return df["precipitacao_mm"]


def _expand_to_daily(series: pd.Series, start: str, end: str) -> pd.Series:
    """
    Expande precipitação mensal para diária via forward fill.
    Precipitação mensal é condição estrutural do período.
    """
    daily_index = pd.date_range(start=start, end=end, freq="D")
    series = series.reindex(daily_index, method="ffill")
    series.index.name = "data"
    return series


def load_copernicus(force_download: bool = False) -> pd.DataFrame:
    """
    Baixa (se necessário) e retorna precipitação diária em mm/mês.

    Retorna
    -------
    pd.DataFrame com coluna:
        precipitacao_mm : precipitação acumulada do mês (mm)
    """
    filepath = download_era5(force=force_download)
    series = _parse_era5(filepath)

    start = DATE_RANGE["start"]
    end   = DATE_RANGE["end"]

    series = series.loc[start:end]
    series = _expand_to_daily(series, start, end)

    return series.to_frame()


if __name__ == "__main__":
    df = load_copernicus()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Precipitação média: {df['precipitacao_mm'].mean():.2f} mm/mês")