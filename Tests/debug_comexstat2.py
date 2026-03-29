# debug_comexstat2.py — fase 2: descobre o formato correto dos filtros
import requests, json

BASE = "https://api-comexstat.mdic.gov.br"
PERIOD = {"from": "2022-01", "to": "2022-03"}

def test(label, payload):
    r = requests.post(f"{BASE}/general", json=payload, timeout=60)
    data = r.json().get("data", {}).get("list", [])
    print(f"[{label}] status={r.status_code} registros={len(data)}")
    if data:
        print(f"   primeiro registro: {data[0]}")

# Teste A: chapter como string "02"
test("A-chapter-str", {
    "flow": "export", "monthDetail": True, "period": PERIOD,
    "filters": [{"filter": "chapter", "values": ["02"]}],
    "details": ["chapter"], "metrics": ["metricFOB", "metricKG"]
})

# Teste B: heading (SH4) como inteiro 201 e 202
test("B-heading-int", {
    "flow": "export", "monthDetail": True, "period": PERIOD,
    "filters": [{"filter": "heading", "values": [201, 202]}],
    "details": ["heading"], "metrics": ["metricFOB", "metricKG"]
})

# Teste C: heading como string "0201" e "0202"
test("C-heading-str", {
    "flow": "export", "monthDetail": True, "period": PERIOD,
    "filters": [{"filter": "heading", "values": ["0201", "0202"]}],
    "details": ["heading"], "metrics": ["metricFOB", "metricKG"]
})

# Teste D: ncm como inteiro 8 digitos (ex: 02011000)
test("D-ncm-int8", {
    "flow": "export", "monthDetail": True, "period": PERIOD,
    "filters": [{"filter": "ncm", "values": [2011000, 2021000]}],
    "details": ["ncm"], "metrics": ["metricFOB", "metricKG"]
})

# Teste E: sem filtro, com detail chapter para ver estrutura retornada
test("E-nofilter-chapter-detail", {
    "flow": "export", "monthDetail": True, "period": PERIOD,
    "filters": [], "details": ["chapter"], "metrics": ["metricFOB", "metricKG"]
})

# Teste F: buscar valores validos do filtro chapter via GET
print("\n[F] Valores validos para filtro 'chapter':")
r = requests.get(f"{BASE}/general/filters/chapter", params={"language": "pt"}, timeout=30)
data = r.json().get("data", {}).get("list", [])
bovinos = [x for x in data if "02" in str(x.get("id","")) or "bov" in str(x).lower() or "carne" in str(x).lower()]
print(f"   Capitulo 02 encontrado: {bovinos[:5]}")
print(f"   Primeiros 5 valores: {data[:5]}")