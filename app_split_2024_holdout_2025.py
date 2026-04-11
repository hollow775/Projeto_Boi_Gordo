from __future__ import annotations

import pandas as pd
import streamlit as st

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


def _render_history_chart(clean_full_df: pd.DataFrame) -> None:
    history_df = clean_full_df.loc[:, ["preco_boi_gordo"]].tail(120).copy()
    st.subheader("Histórico real recente do boi gordo")
    st.line_chart(history_df.rename(columns={"preco_boi_gordo": "Preço real"}))


def _manual_input_form(clean_full_df: pd.DataFrame):
    example_columns = [column for column in clean_full_df.columns if column in SERIES_LABELS]
    last_history_date = clean_full_df.index.max().date()

    with st.form("forecast_form"):
        st.markdown("### Preencha manualmente as variáveis do cenário")
        forecast_base_date = st.date_input(
            "Data-base do cenário",
            value=last_history_date,
            min_value=TRAIN_END.date(),
            max_value=(HOLDOUT_END + pd.Timedelta(days=365)).date(),
        )
        values = {}
        for column in example_columns:
            values[column] = st.text_input(SERIES_LABELS[column], value="")

        model_type = st.selectbox(
            "Curva exibida",
            options=[
                ("media_modelos", "Média dos modelos"),
                ("xgboost", "XGBoost"),
                ("random_forest", "Random Forest"),
            ],
            format_func=lambda option: option[1],
        )[0]

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
            messages.append(f"Campos obrigatórios vazios: {', '.join(missing)}.")
        if invalid:
            messages.append(f"Campos inválidos: {', '.join(invalid)}.")
        raise ValueError(" ".join(messages))
    return parsed


def _render_forecast(curve_df: pd.DataFrame, anchors_df: pd.DataFrame) -> None:
    st.subheader("Curva prevista do dia 1 ao dia 15")
    chart_df = curve_df.set_index("data_previsao")[["valor_previsto"]].rename(
        columns={"valor_previsto": "Preço previsto"}
    )
    st.line_chart(chart_df)

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
    st.set_page_config(page_title="Boi Gordo — fluxo 2024/2025", layout="wide")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #fffaf3 0%, #ffffff 30%);
            }}
            h1, h2, h3 {{
                color: {GREEN};
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
    st.title("Previsão do boi gordo — treino até 2024 e holdout 2025")
    st.caption(
        "Interface simples para explorar cenários manuais com os modelos treinados até 2024-12-31."
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
    example_df = _read_example_values(paths.processed_dir)

    left, right = st.columns([1.4, 1.0])
    with left:
        _render_history_chart(clean_full_df)
    with right:
        st.subheader("Exemplo visual de preenchimento")
        st.caption("Referência do último dia de treino: 2024-12-31. Os campos abaixo continuam manuais.")
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
