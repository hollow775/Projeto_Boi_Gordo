from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.collectors.cepea import load_cepea_price_history_raw
from src.experiments.split_2024_holdout_2025 import (
    HOLDOUT_END,
    SERIES_LABELS,
    TRAIN_END,
    get_experiment_paths,
    load_or_build_feature_datasets,
    predict_manual_curve,
)


GREEN = "#2B9957"
ORANGE = "#E06F00"
HISTORY_WINDOWS = {
    "Última semana": pd.DateOffset(weeks=1),
    "Último mês": pd.DateOffset(months=1),
    "Últimos 6 meses": pd.DateOffset(months=6),
    "Último ano": pd.DateOffset(years=1),
    "Últimos 5 anos": pd.DateOffset(years=5),
    "Desde 2010": pd.Timestamp("2010-01-01"),
}


def _format_example_table(example_df: pd.DataFrame) -> pd.DataFrame:
    if example_df.empty:
        return example_df
    renamed = example_df.rename(columns=SERIES_LABELS).copy()
    return renamed.T.rename(columns={renamed.index[0]: "Exemplo (último dia de treino)"})


def _read_example_values(processed_dir):
    example_path = processed_dir / "exemplo_ultimo_dia_treino.csv"
    if not example_path.exists():
        return pd.DataFrame()
    return pd.read_csv(example_path, index_col="data", parse_dates=True)


def _load_boi_history_raw_cepea() -> pd.DataFrame:
    raw_cepea_df = load_cepea_price_history_raw()
    return raw_cepea_df.loc[:, ["preco_boi_gordo"]].dropna().copy()


def _render_history_chart(boi_history_df: pd.DataFrame) -> None:
    selected_window = st.radio(
        "Período histórico",
        options=list(HISTORY_WINDOWS.keys()),
        horizontal=True,
    )
    end_date = boi_history_df.index.max()
    selected_window_value = HISTORY_WINDOWS[selected_window]
    if isinstance(selected_window_value, pd.DateOffset):
        start_date = end_date - selected_window_value
    else:
        start_date = max(pd.Timestamp(selected_window_value), boi_history_df.index.min())
    history_df = boi_history_df.loc[boi_history_df.index >= start_date].copy()
    if history_df.empty:
        st.warning("Sem dados para o período selecionado.")
        return

    y_min = float(history_df["preco_boi_gordo"].min())
    y_max = float(history_df["preco_boi_gordo"].max())
    y_top = y_max if y_max > y_min else y_min + 1.0

    chart_df = history_df.reset_index().rename(columns={"index": "data"})
    st.subheader("Histórico real do boi gordo")
    hover = alt.selection_point(
        nearest=True,
        on="mouseover",
        fields=["data"],
        empty=False,
    )

    base = (
        alt.Chart(chart_df)
        .encode(
            x=alt.X("data:T", title="Data"),
            y=alt.Y(
                "preco_boi_gordo:Q",
                title="R$/arroba",
                scale=alt.Scale(domain=[y_min, y_top]),
            ),
        )
    )
    line = base.mark_line(color=ORANGE, strokeWidth=3)
    points = (
        base.mark_circle(color=GREEN, size=75)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                alt.Tooltip(
                    "preco_boi_gordo:Q",
                    title="Valor (R$/arroba)",
                    format=".2f",
                ),
            ],
        )
        .add_params(hover)
    )
    st.altair_chart((line + points).properties(height=330), use_container_width=True)
    valor_final = float(history_df["preco_boi_gordo"].iloc[-1])
    variacao = valor_final - y_min
    st.caption(
        f"Valor inicial (mínimo da série exibida): R$ {y_min:.2f} | "
        f"Valor final: R$ {valor_final:.2f} | Alta no período: R$ {variacao:.2f}"
    )


def _manual_input_form(clean_full_df: pd.DataFrame):
    example_columns = [column for column in clean_full_df.columns if column in SERIES_LABELS]
    last_history_date = clean_full_df.index.max().date()

    with st.form("forecast_form"):
        top_left, top_right = st.columns(2)
        with top_left:
            forecast_base_date = st.date_input(
                "Data-base do cenário",
                value=last_history_date,
                min_value=TRAIN_END.date(),
                max_value=(HOLDOUT_END + pd.Timedelta(days=365)).date(),
            )
        with top_right:
            model_type = st.selectbox(
                "Curva exibida",
                options=[
                    ("media_modelos", "Média dos modelos"),
                    ("xgboost", "XGBoost"),
                    ("random_forest", "Random Forest"),
                ],
                format_func=lambda option: option[1],
            )[0]

        st.markdown("##### Variáveis do cenário")
        values: dict[str, str] = {}
        input_cols = st.columns(2, gap="small")
        for idx, column in enumerate(example_columns):
            with input_cols[idx % 2]:
                values[column] = st.text_input(SERIES_LABELS[column], value="")
        submitted = st.form_submit_button("Gerar previsão 1..15 dias")
    return submitted, forecast_base_date, model_type, values


def _parse_manual_values(values: dict[str, str]) -> dict[str, float]:
    parsed = {}
    missing = []
    invalid = []
    for key, value in values.items():
        cleaned = value.strip().replace(",", ".")
        if not cleaned:
            missing.append(SERIES_LABELS[key])
            continue
        try:
            parsed[key] = float(cleaned)
        except ValueError:
            invalid.append(SERIES_LABELS[key])

    if missing or invalid:
        messages = []
        if missing:
            messages.append(f"Campos obrigatorios vazios: {', '.join(missing)}.")
        if invalid:
            messages.append(f"Campos invalidos: {', '.join(invalid)}.")
        raise ValueError(" ".join(messages))
    return parsed


def _render_forecast(curve_df: pd.DataFrame, anchors_df: pd.DataFrame) -> None:
    st.subheader("Curva prevista do dia 1 ao dia 15")

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        curve_df["data_previsao"],
        curve_df["valor_previsto"],
        color=GREEN,
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    axis.set_ylabel("R$/arroba")
    axis.set_xlabel("Data prevista")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.autofmt_xdate(rotation=25)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    st.caption(
        "Regra composta aplicada: dia 1 usa h1, dias 2..7 usam h7 e dias 8..15 usam h15."
    )
    st.dataframe(
        curve_df.assign(
            data_previsao=curve_df["data_previsao"].dt.date,
            valor_previsto=curve_df["valor_previsto"].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Âncoras dos modelos usados na curva")
    st.dataframe(
        anchors_df.round(2).rename(
            columns={
                "horizonte_modelo": "Horizonte do modelo",
                "previsao_xgboost": "XGBoost",
                "previsao_random_forest": "Random Forest",
                "media_modelos": "Média",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Boi Gordo - fluxo 2024/2025", layout="wide")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #f2f8f1 0%, #fff8ef 65%);
                color: #1f2937;
            }}
            .stApp, .stMarkdown, .stCaption, label, p, span, div {{
                color: #1f2937;
            }}
            h1, h2, h3, h4 {{
                color: {GREEN};
            }}
            [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {{
                background: rgba(255,255,255,0.78);
                border: 1px solid #e9efe4;
                border-radius: 12px;
            }}
            .stButton button, .stFormSubmitButton button {{
                background-color: {ORANGE};
                color: white;
                border: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Previsão do boi gordo - treino até 2024 e holdout 2025")
    st.caption(
        "Interface simples para explorar cenários manuais com modelos treinados até 2024-12-31."
    )

    paths = get_experiment_paths()
    if not (
        paths.cache_train_path.exists()
        and paths.cache_full_path.exists()
        and paths.cache_clean_path.exists()
    ):
        st.error(
            "Artefatos do fluxo 2024/2025 ainda não foram gerados. "
            "Execute `python main_split_2024_holdout_2025.py --full` primeiro."
        )
        return

    _, _, clean_full_df = load_or_build_feature_datasets(use_cache=True, paths=paths)
    boi_history_df = _load_boi_history_raw_cepea()
    example_df = _read_example_values(paths.processed_dir)

    left, right = st.columns([1.3, 1.1], gap="large")
    with left:
        with st.container(border=True):
            _render_history_chart(boi_history_df)
    with right:
        with st.container(border=True):
            st.subheader("Preenchimento rápido")
            st.caption("Exemplo do último dia de treino (2024-12-31) para facilitar o preenchimento manual.")
            st.dataframe(_format_example_table(example_df), use_container_width=True)
            submitted, forecast_base_date, model_type, values = _manual_input_form(clean_full_df)
    if not submitted:
        return

    try:
        parsed_values = _parse_manual_values(values)
        curve_df, anchors_df = predict_manual_curve(
            clean_history_df=clean_full_df,
            manual_inputs=parsed_values,
            forecast_base_date=pd.Timestamp(forecast_base_date),
            model_type=model_type,
            paths=paths,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    _render_forecast(curve_df, anchors_df)


if __name__ == "__main__":
    main()
