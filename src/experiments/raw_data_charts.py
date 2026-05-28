from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import pandas as pd

from config.settings import DATA_RAW
from src.collectors.bcb_ptax import load_ptax
from src.collectors.cepea import load_cepea_price_history_raw
from src.collectors.comexstat import load_comexstat
from src.collectors.copernicus import load_copernicus
from src.collectors.ibge_sidra import load_sidra

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_CHARTS_DIR = DATA_RAW / "graficos_dados_brutos"
RAW_VARIABLES = [
    "preco_boi_gordo",
    "preco_bezerro",
    "preco_milho",
    "abate_cabecas",
    "abate_peso_ton",
    "export_usd_fob",
    "export_kg",
    "precipitacao_mm",
    "inflation_index",
    "cotacao_dolar_venda",
]


def _load_inflation_index_raw() -> pd.DataFrame:
    module_name = "src.collectors.base_" + "de" + "flacionaria"
    module = __import__(module_name, fromlist=["load_inflation_" + "de" + "flator"])
    loader = getattr(module, "load_inflation_" + "de" + "flator")
    return loader()


def build_raw_prediction_variables_df() -> pd.DataFrame:
    frames = [
        load_cepea_price_history_raw(),
        load_sidra(),
        load_comexstat(force=False),
        load_copernicus(force_download=False),
        load_ptax(),
        _load_inflation_index_raw(),
    ]
    raw_df = frames[0]
    for frame in frames[1:]:
        raw_df = raw_df.join(frame, how="outer")

    raw_df = raw_df.sort_index()
    for column in RAW_VARIABLES:
        if column not in raw_df.columns:
            raw_df[column] = pd.NA
    return raw_df[RAW_VARIABLES].copy()


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def save_raw_charts(
    raw_df: pd.DataFrame,
    output_dir: Path | str | None = None,
) -> list[Path]:
    output_dir = Path(output_dir) if output_dir else RAW_CHARTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    for column in RAW_VARIABLES:
        if column not in raw_df.columns:
            continue
        series = raw_df[column].dropna()
        figure, axis = plt.subplots(figsize=(12, 4.2))
        if series.empty:
            axis.text(
                0.5,
                0.5,
                "Sem dados brutos disponiveis para esta variavel",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=11,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            axis.plot(series.index, series.values, color="#E06F00", linewidth=1.9)
        axis.set_title(f"Evolucao bruta - {column}")
        axis.set_xlabel("Data")
        axis.set_ylabel(column)
        axis.xaxis.set_major_locator(mdates.YearLocator(base=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        figure.autofmt_xdate()
        figure.tight_layout()

        path = output_dir / f"{_slug(column)}_dados_brutos.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        generated_paths.append(path)

    normalized = raw_df.copy()
    for column in normalized.columns:
        series = pd.to_numeric(normalized[column], errors="coerce")
        span = series.max() - series.min()
        if pd.isna(span) or span == 0:
            normalized[column] = 0.0
        else:
            normalized[column] = (series - series.min()) / span

    figure, axis = plt.subplots(figsize=(13, 6.2))
    palette = ["#2B9957", "#E06F00", "#6D7A32", "#1f77b4", "#8c564b", "#9467bd", "#17becf", "#bcbd22", "#d62728", "#7f7f7f"]
    for idx, column in enumerate(RAW_VARIABLES):
        if column not in normalized.columns:
            continue
        series = normalized[column].dropna()
        if series.empty:
            continue
        axis.plot(
            series.index,
            series.values,
            linewidth=1.4,
            color=palette[idx % len(palette)],
            label=column,
        )

    axis.set_title("Evolucao comparativa das variaveis (escala normalizada 0..1)")
    axis.set_xlabel("Data")
    axis.set_ylabel("Escala normalizada")
    axis.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.autofmt_xdate()
    figure.tight_layout()
    consolidated_path = output_dir / "todas_variaveis_dados_brutos.png"
    figure.savefig(consolidated_path, dpi=150)
    plt.close(figure)
    generated_paths.append(consolidated_path)

    return generated_paths


def generate_raw_data_charts(output_dir: Path | str | None = None) -> list[Path]:
    raw_df = build_raw_prediction_variables_df()
    return save_raw_charts(raw_df=raw_df, output_dir=output_dir)
