# Projeto Boi Gordo — Previsão de Preços

Este repositório contém um pipeline em Python para coletar, tratar e usar dados econômicos,
climáticos e de mercado na previsão do preço do **Boi Gordo**. O projeto foi desenvolvido
como base acadêmica/TCC e combina processamento de dados, engenharia de atributos,
modelos de machine learning e uma interface simples em Streamlit para consulta local.

## O que o projeto faz

De forma resumida, o código:

- coleta e consolida dados de diferentes fontes, como CEPEA, Banco Central, ComexStat,
  IBGE/SIDRA, Copernicus/ERA5 e bases de inflação/deflação;
- limpa e integra as séries temporais em um dataset único;
- cria variáveis derivadas para treinamento dos modelos;
- treina modelos de previsão, incluindo Random Forest e XGBoost;
- avalia os modelos por diferentes horizontes de previsão;
- gera previsões recentes do preço real do Boi Gordo;
- disponibiliza uma interface local em Streamlit para visualização e simulação.

## Estrutura principal

Alguns arquivos importantes do projeto:

- `main.py`: pipeline principal para coleta, treino, avaliação e previsão.
- `main_split_2024_holdout_2025.py`: fluxo experimental isolado, com treino até 2024 e validação/holdout em 2025.
- `app_split_2024_holdout_2025.py`: interface web local em Streamlit.
- `production_daily.py`: fluxo avançado de atualização local com MySQL e exportação de artefatos de produção.
- `requirements.txt`: dependências Python do projeto.
- `docs/`: documentação complementar e relatórios de revisão.
- `Tests/`: testes e verificações automatizadas do projeto.

## Pré-requisitos

- Python 3.11 ou superior.
- `pip` instalado.
- Arquivos manuais do CEPEA em `data/raw/`, conforme os nomes configurados em `config/settings.py`.
- Para coletores que dependem da Copernicus/ERA5, configurar o arquivo `~/.cdsapirc` com as credenciais da API.

> Não commit senhas, tokens ou credenciais reais no repositório.

## Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone <url-do-repositorio>
cd Projeto_Boi_Gordo
```

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` inclui as bibliotecas usadas pelos comandos básicos,
incluindo visualização (`matplotlib`, `streamlit`) e testes (`pytest`).

## Comandos básicos

### Pipeline principal

Treinar os modelos:

```bash
python main.py --train
```

Avaliar os modelos:

```bash
python main.py --evaluate
```

> Observação: a avaliação usa resultados de treino em cache quando disponíveis; se o cache
> não existir, o comando pode treinar novamente antes de exibir/exportar as métricas.

Gerar previsões com modelos já treinados:

```bash
python main.py --predict
```

Executar o fluxo completo, incluindo coleta/processamento, treino, avaliação e previsão:

```bash
python main.py --full
```

### Fluxo isolado treino até 2024 + holdout 2025

Este fluxo mantém artefatos separados do pipeline principal e usa:

- treino até `2024-12-31`;
- validação/holdout entre `2025-01-01` e `2025-12-31`.

Executar treino e avaliação do fluxo isolado:

```bash
python main_split_2024_holdout_2025.py --full
```

Treinar apenas os modelos desse fluxo:

```bash
python main_split_2024_holdout_2025.py --train
```

Avaliar apenas o holdout de 2025:

```bash
python main_split_2024_holdout_2025.py --evaluate
```

Ignorar caches locais e reconstruir os datasets:

```bash
python main_split_2024_holdout_2025.py --full --no-cache
```

### Interface web local

Para abrir a interface Streamlit:

```bash
streamlit run app_split_2024_holdout_2025.py
```

A interface atual permite visualizar o histórico recente e as previsões salvas pelo pipeline.

### Testes

Para rodar a suíte de testes do projeto:

```bash
python -m pytest Tests
```

Se quiser executar apenas os testes compatíveis com `unittest`, use:

```bash
python -m unittest discover -s Tests
```

## Saídas e artefatos

Durante a execução, o projeto pode gerar arquivos em pastas como:

- `data/processed/`: datasets processados e caches intermediários;
- `data/outputs/`: previsões, métricas e arquivos exportados;
- `models_saved/`: modelos treinados e serializados;
- `models_saved/train_split_2024_holdout_2025/`: modelos do fluxo isolado 2024/2025;
- `data/processed/train_split_2024_holdout_2025/`: artefatos processados do fluxo isolado.

## Seção avançada: produção local com MySQL

Além dos comandos básicos, o repositório possui um fluxo avançado em `production_daily.py`
para atualização local, registro de metadados em MySQL, retreinamento de modelos e
exportação de arquivos para uso em interface/site.

Esse fluxo é uma trilha local-first de produção e ainda não representa um pipeline em que
todo o treino/exportação seja lido exclusivamente do banco de dados.

Ele exige configuração das variáveis de ambiente `BOI_DB_HOST`, `BOI_DB_PORT`,
`BOI_DB_NAME`, `BOI_DB_USER` e `BOI_DB_PASSWORD`, além dos arquivos manuais do CEPEA e
credenciais externas quando aplicável.

A documentação detalhada desse modo está em:

```text
docs/production_local_scheduler.md
```

## Observações

- O projeto depende de dados externos e arquivos locais; algumas execuções podem falhar
  se credenciais ou planilhas obrigatórias não estiverem configuradas.
- O fluxo principal e o fluxo experimental 2024/2025 são mantidos separados para evitar
  sobrescrita acidental de artefatos.
- Este README descreve os comandos básicos de uso local; detalhes operacionais avançados
  ficam na documentação em `docs/`.
