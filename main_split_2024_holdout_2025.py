import argparse

from src.experiments.split_2024_holdout_2025 import (
    evaluate_holdout,
    get_experiment_paths,
    load_or_build_feature_datasets,
    run_full_experiment,
    save_ui_reference_artifacts,
    train_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fluxo isolado: treino até 2024-12-31 e holdout em 2025.",
    )
    parser.add_argument("--train", action="store_true", help="Treina os modelos do fluxo 2024/2025.")
    parser.add_argument("--evaluate", action="store_true", help="Avalia o holdout de 2025.")
    parser.add_argument("--full", action="store_true", help="Executa treino + avaliação do fluxo 2024/2025.")
    parser.add_argument("--no-cache", action="store_true", help="Ignora caches locais do fluxo 2024/2025.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_cache = not args.no_cache
    paths = get_experiment_paths()

    if args.full:
        result = run_full_experiment(use_cache=use_cache, paths=paths)
        print(result["holdout_metrics_df"].to_string(index=False))
        print(f"\n[split_2024_holdout_2025] Artefatos salvos em: {paths.processed_dir}")
        return 0

    train_features_df, full_features_df, clean_full_df = load_or_build_feature_datasets(
        use_cache=use_cache,
        paths=paths,
    )
    save_ui_reference_artifacts(clean_full_df, paths=paths)

    if args.train:
        train_experiment(train_features_df, paths=paths)
        print(f"[split_2024_holdout_2025] Modelos salvos em: {paths.models_dir}")
        return 0

    if args.evaluate:
        metrics_df, _ = evaluate_holdout(full_features_df, paths=paths)
        print(metrics_df.to_string(index=False))
        print(f"\n[split_2024_holdout_2025] Métricas salvas em: {paths.holdout_metrics_path}")
        return 0

    print(
        "Nenhum modo especificado.\n"
        "Use: python main_split_2024_holdout_2025.py --train | --evaluate | --full\n"
        "     adicione --no-cache para reconstruir os datasets."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
