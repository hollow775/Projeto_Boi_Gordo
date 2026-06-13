from __future__ import annotations

import pandas as pd

from src.production.policy import ProductionPolicy, get_production_policy


def export_production_outputs(
    features_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    policy: ProductionPolicy | None = None,
) -> None:
    """Export production predictions/history for the Streamlit website."""
    from src.export.export_outputs import export_all

    policy = policy or get_production_policy()
    policy.ensure_directories()
    export_all(
        features_df,
        clean_df,
        models_dir=policy.models_dir,
        data_outputs=policy.outputs_dir,
    )
