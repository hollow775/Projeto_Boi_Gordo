# debug_comexstat.py
# Execute este script para diagnosticar a API ComexStat:
#   python debug_comexstat.py

import requests
import json

BASE_URL = "https://api-comexstat.mdic.gov.br"

print("=" * 60)
print("DIAGNOSTICO API COMEXSTAT")
print("=" * 60)

# ── Teste 1: filtros disponiveis ───────────────────────────────
print("\n[1] Buscando filtros disponiveis...")
r = requests.get(f"{BASE_URL}/general/filters", params={"language": "pt"}, timeout=30)
print(f"    Status: {r.status_code}")
if r.ok:
    data = r.json().get("data", {})
    print(f"    Filtros: {json.dumps(data, ensure_ascii=False)[:400]}")

# ── Teste 2: detalhamentos disponiveis ─────────────────────────
print("\n[2] Buscando detalhamentos disponiveis...")
r = requests.get(f"{BASE_URL}/general/details", params={"language": "pt"}, timeout=30)
print(f"    Status: {r.status_code}")
if r.ok:
    data = r.json().get("data", {})
    print(f"    Detalhes: {json.dumps(data, ensure_ascii=False)[:400]}")

# ── Teste 3: payload com capitulo ─────────────────────────────
print("\n[3] Testando payload com filtro 'chapter'...")
payload = {
    "flow": "export",
    "monthDetail": True,
    "period": {"from": "2022-01", "to": "2022-03"},
    "filters": [{"filter": "chapter", "values": [2]}],
    "details": ["chapter"],
    "metrics": ["metricFOB", "metricKG"],
}
r = requests.post(f"{BASE_URL}/general", json=payload, timeout=60)
print(f"    Status: {r.status_code}")
print(f"    Resposta: {r.text[:600]}")

# ── Teste 4: payload com ncm string ───────────────────────────
print("\n[4] Testando payload com filtro 'ncm' (string)...")
payload["filters"] = [{"filter": "ncm", "values": ["0201", "0202"]}]
payload["details"] = ["ncm"]
r = requests.post(f"{BASE_URL}/general", json=payload, timeout=60)
print(f"    Status: {r.status_code}")
print(f"    Resposta: {r.text[:600]}")

# ── Teste 5: payload com section ──────────────────────────────
print("\n[5] Testando payload sem filtros (apenas periodo)...")
payload_simple = {
    "flow": "export",
    "monthDetail": True,
    "period": {"from": "2022-01", "to": "2022-01"},
    "filters": [],
    "details": [],
    "metrics": ["metricFOB", "metricKG"],
}
r = requests.post(f"{BASE_URL}/general", json=payload_simple, timeout=60)
print(f"    Status: {r.status_code}")
print(f"    Resposta: {r.text[:600]}")

print("\n" + "=" * 60)
print("Cole a saida completa para analise.")
print("=" * 60)