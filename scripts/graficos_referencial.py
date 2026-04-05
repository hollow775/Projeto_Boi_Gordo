# scripts/graficos_referencial.py
# ==============================================================
# Gera gráficos para o referencial teórico:
#   - Abate de fêmeas × Preço do boi gordo
#   - Preço Boi Gordo (18@) x Preço Bezerro
#   - Relação de troca: boi gordo (18@) × bezerro
# ==============================================================

import sys
from pathlib import Path

# Garante que o pacote raiz seja encontrado
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from src.collectors.cepea import load_cepea
from src.collectors.ibge_sidra import load_sidra
from config.settings import DATA_PROCESSED

# ── Paleta ────────────────────────────────────────────────────
COR_BOI   = "#07a458"   # preço boi gordo
COR_FEMEA = "#db620c"   # abate de fêmeas / bezerro

# ── Parâmetros do boi ─────────────────────────────────────────
ARROBAS_BOI = 18

# ── Diretório de saída ────────────────────────────────────────
SAVE_DIR = DATA_PROCESSED / "graficos_referencial"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Estilo global ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.size":            11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "axes.grid.axis":       "y",
    "grid.color":           "#e0e0e0",
    "grid.linewidth":       0.7,
    "grid.linestyle":       "--",
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
})

# ══════════════════════════════════════════════════════════════
# CARGA DOS DADOS
# ══════════════════════════════════════════════════════════════

print("Carregando CEPEA (preços)...")
df_cepea = load_cepea()

print("Carregando SIDRA (abate de fêmeas)...")
df_sidra = load_sidra()

# Preços mensais (média)
preco_boi = df_cepea["preco_boi_gordo"].resample("MS").mean().dropna()
preco_bez = df_cepea["preco_bezerro"].resample("MS").mean().dropna()

# Abate mensal - usando index "MS" para plotar como linha:
abate_col = df_sidra["abate_peso_ton"]
if isinstance(abate_col, pd.DataFrame):
   abate_col = abate_col.iloc[:, 0]

abate_mensal = abate_col.resample("MS").first().dropna()
# Valores em quilogramas -> converte para mil toneladas ( / 1_000_000 )
abate_mensal_mil_ton = (abate_mensal / 1_000_000).astype(float)


# ══════════════════════════════════════════════════════════════
# GRÁFICO: Abate de Fêmeas × Preço do Boi Gordo
# ══════════════════════════════════════════════════════════════

print("\nGerando Gráfico — Abate de Fêmeas × Preço do Boi Gordo...")

fig, ax1 = plt.subplots(figsize=(13, 5.5))

# ── Linha: abate de fêmeas (eixo esquerdo) ─────────────────
ax1.plot(
    abate_mensal_mil_ton.index,
    abate_mensal_mil_ton.values,
    color=COR_FEMEA,
    linewidth=2.2,
    zorder=3,
)
ax1.set_ylabel("Peso de carcaças (mil toneladas)", color=COR_FEMEA, fontsize=11, labelpad=8)
ax1.tick_params(axis="y", colors=COR_FEMEA, labelsize=10)
ax1.spines["left"].set_color(COR_FEMEA)
ax1.spines["right"].set_visible(False)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

# ── Linha: preço boi gordo (eixo direito) ───────────────────
ax2 = ax1.twinx()
ax2.plot(
    preco_boi.index,
    preco_boi.values,
    color=COR_BOI,
    linewidth=2.2,
    zorder=3,
)
ax2.fill_between(preco_boi.index, preco_boi.values, alpha=0.07, color=COR_BOI)
ax2.set_ylabel("Preço boi gordo (R$/arroba)", color=COR_BOI, fontsize=11, labelpad=8)
ax2.tick_params(axis="y", colors=COR_BOI, labelsize=10)
ax2.spines["right"].set_color(COR_BOI)
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}")
)

# ── Eixo X ───────────────────────────────────────────────────
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis="x", labelsize=10)
ax1.set_xlabel("Período", fontsize=11, labelpad=6)

# ── Título e legenda ─────────────────────────────────────────
ax1.set_title(
    "Abate de Fêmeas e Preço do Boi Gordo — São Paulo",
    fontsize=13, fontweight="bold", pad=14,
)

handles = [
    Line2D([0], [0], color=COR_FEMEA, linewidth=2.2, label="Peso abate fêmeas (mil toneladas/mês)"),
    Line2D([0], [0], color=COR_BOI, linewidth=2.2, label="Preço boi gordo (R$/arroba)"),
]
ax1.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)



fig.tight_layout(pad=1.5)
p1 = SAVE_DIR / "abate_femeas_vs_preco_boi.png"
fig.savefig(p1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ Salvo em: {p1}")


# ══════════════════════════════════════════════════════════════
# GRÁFICO: Preços — Boi Gordo (18@) × Bezerro
# ══════════════════════════════════════════════════════════════

print("\nGerando Gráfico — Preços: Boi Gordo (18@) × Bezerro...")

# Alinha períodos
idx_comum = preco_boi.index.intersection(preco_bez.index)
boi_18    = preco_boi.loc[idx_comum] * ARROBAS_BOI   # valor total do boi (18@)
bezerro   = preco_bez.loc[idx_comum]

fig, ax_top = plt.subplots(figsize=(13, 5.5))

# Como é o mesmo eixo Y para não mascarar as diferenças
ax_top.plot(boi_18.index, boi_18.values, color=COR_BOI, linewidth=2.2,
            label=f"Boi gordo — {ARROBAS_BOI}@ (R$/cabeça)")
ax_top.fill_between(boi_18.index, boi_18.values, alpha=0.08, color=COR_BOI)

ax_top.plot(bezerro.index, bezerro.values, color=COR_FEMEA, linewidth=2.2,
            linestyle="--", label="Bezerro (R$/cabeça)")
ax_top.fill_between(bezerro.index, bezerro.values, alpha=0.08, color=COR_FEMEA)

ax_top.set_ylabel("Valor Total (R$/cabeça)", fontsize=11, labelpad=8)
ax_top.tick_params(axis="y", labelsize=10)
ax_top.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}")
)

ax_top.set_title(
    f"Valores por Cabeça — Boi Gordo ({ARROBAS_BOI} arrobas) e Bezerro",
    fontsize=13, fontweight="bold", pad=14,
)

ax_top.legend(loc="upper left", frameon=False, fontsize=10)

ax_top.xaxis.set_major_locator(mdates.YearLocator(2))
ax_top.xaxis.set_minor_locator(mdates.YearLocator(1))
ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_top.tick_params(axis="x", labelsize=10)
ax_top.set_xlabel("Período", fontsize=11, labelpad=6)



fig.tight_layout(pad=1.5)
p2 = SAVE_DIR / "precos_boi_bezerro.png"
fig.savefig(p2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ Salvo em: {p2}")

# ══════════════════════════════════════════════════════════════
# GRÁFICO: Relação de Troca
# ══════════════════════════════════════════════════════════════

print("\nGerando Gráfico — Relação de Troca...")

ratio = boi_18 / bezerro # nº de bezerros por boi gordo

fig, ax_bot = plt.subplots(figsize=(13, 5.5))

# Área colorida entre a linha e o valor médio da razão
media_ratio = ratio.mean()
ax_bot.axhline(media_ratio, color="gray", linewidth=1, linestyle=":",
               label=f"Média: {media_ratio:.2f}")

ax_bot.fill_between(
    ratio.index, ratio.values, media_ratio,
    where=(ratio.values >= media_ratio),
    alpha=0.30, color=COR_BOI, interpolate=True,
)
ax_bot.fill_between(
    ratio.index, ratio.values, media_ratio,
    where=(ratio.values < media_ratio),
    alpha=0.30, color=COR_FEMEA, interpolate=True,
)
ax_bot.plot(ratio.index, ratio.values, color="#333333", linewidth=1.8, label="Relação de troca")

ax_bot.set_title(
    "Relação de Troca (Bezerros por Boi Gordo)",
    fontsize=13, fontweight="bold", pad=14,
)

ax_bot.set_ylabel("Bezerros por boi gordo", fontsize=10, labelpad=6)
ax_bot.tick_params(axis="y", labelsize=10)
ax_bot.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.1f}")
)
ax_bot.legend(loc="upper left", frameon=False, fontsize=9)

ax_bot.xaxis.set_major_locator(mdates.YearLocator(2))
ax_bot.xaxis.set_minor_locator(mdates.YearLocator(1))
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_bot.tick_params(axis="x", labelsize=10)
ax_bot.set_xlabel("Período", fontsize=11, labelpad=6)

ax_bot.set_ylim(top=ratio.max() * 1.15)


fig.tight_layout(pad=1.5)
p3 = SAVE_DIR / "relacao_troca_boi_bezerro.png"
fig.savefig(p3, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ Salvo em: {p3}")

print("\nConcluído! Arquivos salvos em:", SAVE_DIR)
