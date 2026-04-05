# config/settings.py
# ==============================================================
# ÚNICO arquivo a modificar para trocar o estado de análise.
# ==============================================================

from pathlib import Path

# ── Diretórios base ────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models_saved"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Estado alvo ────────────────────────────────────────────────
STATE = {
    "name": "São Paulo",
    "ibge_code": "35",          # Código IBGE do estado no SIDRA
    "cepea_file": str(DATA_RAW / "cepea_sp.xlsx"),   # planilha exportada do CEPEA
    # Bounding box [lon_min, lat_min, lon_max, lat_max] para Copernicus ERA5
    "copernicus_bbox": [-53.1, -25.3, -44.0, -19.8],
    # Nome do arquivo ERA5 que será baixado
    "era5_file": str(DATA_RAW / "era5_sp.nc"),
}

# ── Período de análise ─────────────────────────────────────────
DATE_RANGE = {
    "start": "2010-01-01",
    "end": "2025-12-31",
}

# ── Horizontes de previsão (dias corridos) ─────────────────────
HORIZONS = [1, 15, 30, 60]

# ── Deflação ───────────────────────────────────────────────────
# Código da série de inflação no Banco Central (SGS: 433 para IPCA, 190 para IGP-DI)
INFLATION_SGS_CODE = 190
# Mês de referência para deflação (formato YYYY-MM)
DEFLATION_BASE = "2025-12"

# ── Copernicus CDS ─────────────────────────────────────────────
# Requer cadastro em https://cds.climate.copernicus.eu/
# e arquivo ~/.cdsapirc com url e key
CDS_DATASET = "reanalysis-era5-land-monthly-means"
CDS_VARIABLE = "total_precipitation"

# ── ComexStat MDIC ─────────────────────────────────────────────
# Capítulo NCM 02 = carnes. Códigos específicos de carne bovina.
COMEX_NCM_CODES = ["0201", "0202"]   # in natura; processada
COMEX_API_URL = "https://api-comexstat.mdic.gov.br/general"

# ── SIDRA IBGE ─────────────────────────────────────────────────
# Tabela 1092 — Abate de animais (nova URL via demanda do usuário)
SIDRA_URL = (
    "https://apisidra.ibge.gov.br/values/t/1092/n1/all/v/285/p/last%2064/c12716/allxt/c18/56/c12529/118225"
)

# ── Banco Central ──────────────────────────────────────────────
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
BCB_PTAX_URL = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@dataInicial='{start}'&@dataFinalCotacao='{end}'&$format=json"