from __future__ import annotations

import argparse

from src.experiments.raw_data_charts import RAW_CHARTS_DIR, generate_raw_data_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera graficos de dados brutos (sem etapa de limpeza) "
            "para as variaveis usadas na previsao."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(RAW_CHARTS_DIR),
        help="Pasta de saida para os PNGs. Default: data/raw/graficos_dados_brutos",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = generate_raw_data_charts(output_dir=args.output_dir)
    print(f"[raw_charts] Arquivos gerados: {len(paths)}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

