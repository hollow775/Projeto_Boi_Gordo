from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import DATA_OUTPUTS, LAST_RUN_PATH

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


def _read_last_run_timestamp() -> str:
    if LAST_RUN_PATH.exists():
        data = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
        return data.get("timestamp", "desconhecido")
    return "pipeline nunca executado"


def _load_predictions() -> pd.DataFrame:
    path = DATA_OUTPUTS / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_price_history() -> pd.DataFrame:
    path = DATA_OUTPUTS / "price_history.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df


def _render_history_chart(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        st.warning("Sem dados de histórico disponíveis.")
        return

    selected_window = st.radio(
        "Período histórico",
        options=list(HISTORY_WINDOWS.keys()),
        horizontal=True,
    )
    end_date = history_df.index.max()
    window_value = HISTORY_WINDOWS[selected_window]
    if isinstance(window_value, pd.DateOffset):
        start_date = end_date - window_value
    else:
        start_date = max(pd.Timestamp(window_value), history_df.index.min())

    filtered = history_df.loc[history_df.index >= start_date].copy()
    if filtered.empty:
        st.warning("Sem dados para o período selecionado.")
        return

    y_min = float(filtered["real_price_deflated"].min())
    y_max = float(filtered["real_price_deflated"].max())

    chart_df = filtered.reset_index().rename(columns={"date": "data"})
    st.subheader("Histórico real do boi gordo (deflacionado)")

    hover = alt.selection_point(nearest=True, on="mouseover", fields=["data"], empty=False)
    base = alt.Chart(chart_df).encode(
        x=alt.X("data:T", title="Data"),
        y=alt.Y("real_price_deflated:Q", title="R$/arroba", scale=alt.Scale(domain=[y_min, y_max])),
    )
    line = base.mark_line(color=ORANGE, strokeWidth=3)
    points = (
        base.mark_circle(color=GREEN, size=75)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                alt.Tooltip("real_price_deflated:Q", title="R$/arroba", format=".2f"),
            ],
        )
        .add_params(hover)
    )
    st.altair_chart((line + points).properties(height=330), use_container_width=True)


def _render_predictions(predictions_df: pd.DataFrame) -> None:
    if predictions_df.empty:
        st.warning("Sem previsões disponíveis. Execute o pipeline primeiro.")
        return

    st.subheader("Previsões por horizonte")

    model_filter = st.selectbox(
        "Modelo",
        options=["xgboost", "random_forest"],
        format_func=lambda x: "XGBoost" if x == "xgboost" else "Random Forest",
    )
    filtered = predictions_df[predictions_df["model"] == model_filter].copy()
    filtered = filtered.sort_values("horizon")

    st.dataframe(
        filtered[["date", "horizon", "predicted_value"]].rename(
            columns={
                "date": "Data prevista",
                "horizon": "Horizonte (dias)",
                "predicted_value": "Previsão (R$/arroba)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Boi Gordo - Pipeline Produção", layout="wide")
    st.markdown(
        f"""
        <style>
            .stApp {{ background: linear-gradient(180deg, #f2f8f1 0%, #fff8ef 65%); color: #1f2937; }}
            h1, h2, h3, h4 {{ color: {GREEN}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Previsão do boi gordo — Pipeline de Produção")

    last_run = _read_last_run_timestamp()
    st.caption(f"Última execução do pipeline: **{last_run}**")

    predictions_df = _load_predictions()
    history_df = _load_price_history()

    left, right = st.columns([1.3, 1.1], gap="large")
    with left:
        with st.container(border=True):
            _render_history_chart(history_df)
    with right:
        with st.container(border=True):
            _render_predictions(predictions_df)


if __name__ == "__main__":
    main()
