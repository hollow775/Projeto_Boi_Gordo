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


def _parse_manual_values(raw_values: dict[str, str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    missing_fields: list[str] = []
    invalid_fields: list[str] = []

    for field, raw_value in raw_values.items():
        text_value = str(raw_value).strip()
        if not text_value:
            missing_fields.append(field)
            continue
        try:
            parsed[field] = float(text_value.replace(",", "."))
        except ValueError:
            invalid_fields.append(field)

    if missing_fields or invalid_fields:
        messages: list[str] = []
        if missing_fields:
            messages.append(f"Campos obrigatorios vazios: {', '.join(missing_fields)}")
        if invalid_fields:
            messages.append(f"Campos invalidos: {', '.join(invalid_fields)}")
        raise ValueError("; ".join(messages))

    return parsed


def _read_last_run_timestamp() -> str:
    for path in (LAST_RUN_PATH.parent / "production_last_run.json", LAST_RUN_PATH):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("timestamp", data.get("trained_at_utc", "desconhecido"))
    return "pipeline nunca executado"


def _output_file(filename: str):
    production_path = DATA_OUTPUTS / "production" / filename
    if production_path.exists():
        return production_path
    return DATA_OUTPUTS / filename


def _load_predictions() -> pd.DataFrame:
    path = _output_file("predictions.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_price_history() -> pd.DataFrame:
    path = _output_file("price_history.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df


def _empty_change_signals() -> dict[str, float | str | None]:
    return {
        "latest_date": None,
        "latest_value": None,
        "delta_1d": None,
        "delta_7d": None,
        "pct_7d": None,
    }


def _compute_history_change_signals(history_df: pd.DataFrame) -> dict[str, float | str | None]:
    if history_df.empty or "real_price_deflated" not in history_df.columns:
        return _empty_change_signals()

    series = history_df["real_price_deflated"].dropna().sort_index()
    if series.empty:
        return _empty_change_signals()

    latest_value = float(series.iloc[-1])
    delta_1d = float(latest_value - series.iloc[-2]) if len(series) >= 2 else None
    delta_7d = float(latest_value - series.iloc[-8]) if len(series) >= 8 else None
    pct_7d = float(delta_7d / series.iloc[-8] * 100) if delta_7d is not None and series.iloc[-8] else None
    return {
        "latest_date": series.index[-1].strftime("%Y-%m-%d"),
        "latest_value": latest_value,
        "delta_1d": delta_1d,
        "delta_7d": delta_7d,
        "pct_7d": pct_7d,
    }


def _render_change_signals(history_df: pd.DataFrame) -> None:
    signals = _compute_history_change_signals(history_df)
    st.subheader("Sinais de mudança recentes")
    if signals["latest_value"] is None:
        st.info("Sem histórico de produção suficiente para calcular mudanças.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Último preço real",
        f"R$ {signals['latest_value']:.2f}",
        help=f"Data mais recente: {signals['latest_date']}",
    )
    col2.metric(
        "Variação diária",
        "n/d" if signals["delta_1d"] is None else f"R$ {signals['delta_1d']:.2f}",
    )
    col3.metric(
        "Variação 7 dias",
        "n/d"
        if signals["delta_7d"] is None or signals["pct_7d"] is None
        else f"R$ {signals['delta_7d']:.2f} ({signals['pct_7d']:.2f}%)",
    )


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
            _render_change_signals(history_df)
        with st.container(border=True):
            _render_history_chart(history_df)
    with right:
        with st.container(border=True):
            _render_predictions(predictions_df)


if __name__ == "__main__":
    main()
