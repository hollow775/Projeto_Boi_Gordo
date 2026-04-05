# src/collectors/export_dados.py
# ==============================================================
# Script auxiliar para exportar dados das fontes para planilhas,
# garantindo uma visualização fácil pelo usuário antes do treino.
# ==============================================================

import pandas as pd
from pathlib import Path
import sys

# Garante que o pacote raiz seja encontrado
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.collectors.comexstat import load_comexstat
from src.collectors.copernicus import load_copernicus
from src.collectors.ibge_sidra import load_sidra
from src.collectors.bcb_ptax import load_ptax

def main():
    # Caminho onde as planilhas serão salvas (no diretório deste próprio script)
    output_dir = Path(__file__).parent
    
    print("Coletando ComexStat...")
    try:
        df_comex = load_comexstat()
        file_comex = output_dir / "comexstat_inspecao.xlsx"
        df_comex.to_excel(file_comex)
        print(f"Salvo: {file_comex.name}")
    except Exception as e:
        print(f"Erro ComexStat: {e}")

    print("\nColetando SIDRA...")
    try:
        df_sidra = load_sidra()
        file_sidra = output_dir / "sidra_inspecao.xlsx"
        df_sidra.to_excel(file_sidra)
        print(f"Salvo: {file_sidra.name}")
    except Exception as e:
        print(f"Erro SIDRA: {e}")

    # Copernicus exige chaves e é um raster - o load_copernicus faz agregação se já tem baixado
    print("\nColetando Copernicus (ERA5)...")
    try:
        df_copernicus = load_copernicus()
        file_copernicus = output_dir / "copernicus_inspecao.xlsx"
        df_copernicus.to_excel(file_copernicus)
        print(f"Salvo: {file_copernicus.name}")
    except Exception as e:
        print(f"Erro Copernicus: {e}")

    print("\nColetando Dólar (PTAX BCB)...")
    try:
        df_ptax = load_ptax()
        file_ptax = output_dir / "ptax_inspecao.xlsx"
        df_ptax.to_excel(file_ptax)
        print(f"Salvo: {file_ptax.name}")
    except Exception as e:
        print(f"Erro PTAX: {e}")

if __name__ == "__main__":
    main()
