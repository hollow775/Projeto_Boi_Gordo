# src/collectors/copernicus.py
# ==============================================================
# Coleta dados de precipitacao via Copernicus Climate Data Store.
# Dataset: ERA5-Land Monthly Means
# Variavel: total_precipitation
#
# Pre-requisitos:
#   1. Cadastro em https://cds.climate.copernicus.eu/
#   2. pip install cdsapi cfgrib eccodes
#   3. Arquivo ~/.cdsapirc com:
#        url: https://cds.climate.copernicus.eu/api
#        key: <sua-api-key>
# ==============================================================

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import STATE, DATE_RANGE, CDS_DATASET, CDS_VARIABLE


def _detect_engine(filepath: Path) -> str:
    """
    Detecta o formato do arquivo pelo header binario.
    GRIB comeca com b'GRIB', NetCDF com b'CDF' ou b'\\x89HDF'.
    """
    with open(filepath, "rb") as f:
        header = f.read(4)

    if header[:4] == b"GRIB":
        return "cfgrib"
    elif header[:3] == b"CDF" or header[:4] == b"\x89HDF":
        return "netcdf4"
    else:
        # Tenta netcdf4 como fallback
        return "netcdf4"


def _build_year_month_lists(start: str, end: str) -> tuple[list, list]:
    dates  = pd.date_range(start=start, end=end, freq="MS")
    years  = sorted(set(str(d.year) for d in dates))
    months = [str(m).zfill(2) for m in range(1, 13)]
    return years, months


def download_era5(force: bool = False) -> Path:
    """
    Baixa o arquivo ERA5-Land para a bounding box do estado configurado.
    O download e feito apenas uma vez (arquivo em cache em data/raw/).
    """
    output_path = Path(STATE["era5_file"])

    if output_path.exists() and not force:
        print(f"[copernicus] ERA5 ja baixado: {output_path}")
        return output_path

    bbox = STATE["copernicus_bbox"]  # [lon_min, lat_min, lon_max, lat_max]
    area = [bbox[3], bbox[0], bbox[1], bbox[2]]  # [N, W, S, E]

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
            "data_format":  "netcdf",     # solicita NetCDF explicitamente
            "download_format": "unarchived",
        },
        str(output_path),
    )

    print(f"[copernicus] ERA5 salvo em: {output_path}")
    return output_path


def _parse_era5(filepath: Path) -> pd.Series:
    """
    Le o arquivo ERA5 (NetCDF ou GRIB) e calcula precipitacao
    media mensal sobre toda a area do estado (media espacial).
    """
    engine = _detect_engine(filepath)
    print(f"[copernicus] Formato detectado: {engine}")

    if engine == "cfgrib":
        # GRIB pode ter multiplas mensagens — filtra precipitacao
        ds = xr.open_dataset(
            filepath,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            filter_by_keys={"shortName": "tp"},
        )
        var_name = "tp"
    else:
        ds = xr.open_dataset(filepath, engine="netcdf4")
        # Identifica variavel de precipitacao (tp ou total_precipitation)
        var_name = next(
            (v for v in ds.data_vars if "precip" in v.lower() or v == "tp"),
            list(ds.data_vars)[0],
        )

    print(f"[copernicus] Variavel utilizada: {var_name}")

    tp = ds[var_name]

    # Identifica dimensoes espaciais (podem ser lat/lon ou latitude/longitude)
    spatial_dims = [d for d in tp.dims if d in ("lat", "lon", "latitude", "longitude")]
    tp_mean = tp.mean(dim=spatial_dims)

    df = tp_mean.to_dataframe(name="precipitacao_mm")

    # Garante indice datetime
    df.index = pd.to_datetime(df.index)
    df.index.name = "data"

    # Converte metros para milimetros
    df["precipitacao_mm"] = df["precipitacao_mm"] * 1000

    ds.close()

    # Remove linhas duplicadas de indice (GRIB pode gerar)
    df = df[~df.index.duplicated(keep="first")]

    return df["precipitacao_mm"].sort_index()


def _expand_to_daily(series: pd.Series, start: str, end: str) -> pd.Series:
    daily_index = pd.date_range(start=start, end=end, freq="D")
    series = series.reindex(daily_index, method="ffill")
    series.index.name = "data"
    return series


def load_copernicus(force_download: bool = False) -> pd.DataFrame:
    """
    Baixa (se necessario) e retorna precipitacao diaria em mm/mes.

    Retorna
    -------
    pd.DataFrame com coluna:
        precipitacao_mm : precipitacao acumulada do mes (mm)
    """
    filepath = download_era5(force=force_download)
    series   = _parse_era5(filepath)

    start = DATE_RANGE["start"]
    end   = DATE_RANGE["end"]

    series = series.loc[start:end]
    series = _expand_to_daily(series, start, end)

    print(f"[copernicus] {len(series)} registros diarios carregados.")
    return series.to_frame()


if __name__ == "__main__":
    df = load_copernicus()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Precipitacao media: {df['precipitacao_mm'].mean():.2f} mm/mes")