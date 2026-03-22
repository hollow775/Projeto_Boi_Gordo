# src/sentiment/README.md

# Módulo de Análise de Sentimentos — Implementação Futura

## Objetivo

Incorporar eventos exógenos não capturados pelas variáveis quantitativas:
pandemias, crises tarifárias, embargos sanitários, choques cambiais.

## Interface esperada

O módulo deve expor uma função com a seguinte assinatura:

```python
def load_sentiment(start: str, end: str) -> pd.DataFrame:
    """
    Retorna DataFrame com índice DatetimeIndex diário e coluna:
        sentiment_score : float em [-1.0, +1.0]
            -1.0 = sentimento extremamente negativo
             0.0 = neutro
            +1.0 = sentimento extremamente positivo
    """
```

O `merger.py` já está preparado para receber este DataFrame via `join`
no `build_dataset()`. Bastará adicionar:

```python
if include_sentiment:
    from src.sentiment.loader import load_sentiment
    frames.append(load_sentiment(start, end))
```

## Fontes candidatas

| Fonte | Tipo | Acesso |
|---|---|---|
| Google News RSS | Headlines em PT/EN | Scraping |
| NewsAPI | Headlines indexadas | API paga/free-tier |
| GDELT Project | Eventos globais codificados | API aberta |
| Twitter/X Academic | Posts sobre pecuária | API restrita |

## Abordagem recomendada

1. **Coleta**: headlines de veículos especializados (Scot Consultoria,
   BeefPoint, Canal Rural) + agências generalistas (Reuters, Bloomberg
   Agribusiness).

2. **Modelo de sentimento**: para textos em português, o
   `BERTimbau` (neuralmind/bert-base-portuguese-cased, disponível no
   HuggingFace) é o ponto de partida mais sólido. Alternativa mais leve:
   `pysentimiento` com modelo treinado em PT-BR.

3. **Agregação diária**: média dos scores de todas as headlines do dia.
   Dias sem cobertura → score = 0.0 (neutro).

4. **Lags**: aplicar os mesmos LAG_DAYS de `engineering.py` ao
   sentiment_score, pois o impacto de uma notícia não é instantâneo
   sobre o preço físico do boi.

## Referência metodológica

Zhu et al. (2024) — *Integrating Structured and Unstructured Data for
Livestock Price Forecasting* — demonstra ganho de acurácia ao combinar
dados quantitativos com features de texto em modelos de previsão de
preços de commodities pecuárias.