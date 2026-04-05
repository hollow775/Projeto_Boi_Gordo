# src/collectors/bcb_ptax.py
# ==============================================================
# Coleta a cotação diária do Dólar Americano (USD) através da
# API Olinda (PTAX) do Banco Central do Brasil.
# ==============================================================

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
from config.settings import DATE_RANGE, BCB_PTAX_URL

def _expand_to_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Expande dados filtrando apenas os dias úteis cotados e propagando
    a última cotação para os finais de semana/feriados.
    """
    daily_index = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(daily_index, method="ffill").bfill()
    df.index.name = "data"
    return df

def load_ptax() -> pd.DataFrame:
    """
    Coleta e retorna a cotação PTAX (Venda) do Dólar em frequência diária.
    
    Retorna
    -------
    pd.DataFrame com a coluna:
        cotacao_dolar_venda : Valor do dólar venda em reais
    """
    
    # Converter YYYY-MM-DD para MM-DD-YYYY para a API Olinda
    start_dt = pd.to_datetime(DATE_RANGE["start"])
    end_dt = pd.to_datetime(DATE_RANGE["end"])
    
    start_str = start_dt.strftime("%m-%d-%Y")
    end_str = end_dt.strftime("%m-%d-%Y")
    
    url = BCB_PTAX_URL.format(start=start_str, end=end_str)
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Falha ao conectar na API PTAX: {e}\nURL: {url}")
        
    data = response.json().get("value", [])
    if not data:
        raise ValueError("Resposta PTAX vazia ou inválida.")
        
    df = pd.DataFrame(data)
    
    # Extrai parte da data (vem no formato YYYY-MM-DD HH:MM:SS) e converte
    df["data"] = pd.to_datetime(df["dataHoraCotacao"]).dt.normalize()
    
    # Mantém apenas a cotação de venda e o dia
    df = df[["data", "cotacaoVenda"]].rename(columns={"cotacaoVenda": "cotacao_dolar_venda"})
    
    # Podem haver múltiplos boletins num mesmo dia raramente (fechamentos intermediarios),
    # mas o Olinda traz a PTAX final do dia. Por segurança pegamos o último do dia.
    df = df.groupby("data").last()
    
    start = DATE_RANGE["start"]
    end = DATE_RANGE["end"]
    df = df.loc[start:end]
    
    df = _expand_to_daily(df, start, end)
    
    return df

if __name__ == "__main__":
    df = load_ptax()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
