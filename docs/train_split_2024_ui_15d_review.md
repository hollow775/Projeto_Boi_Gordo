# Revisão e documentação — split treino até 2024 + validação 2025 + UI 15 dias

## Objetivo da entrega

Implementar um fluxo isolado do pipeline atual com as seguintes regras obrigatórias:

- treino/tuning usando dados até **2024-12-31**
- validação/holdout usando somente **2025-01-01** a **2025-12-31**
- artefatos salvos em caminhos separados do fluxo legado
- interface web simples com paleta verde/laranja
- gráfico de histórico real recente
- formulário manual com as variáveis do treinamento
- curva diária de previsão de **1 a 15 dias** composta por:
  - dia 1 → modelo `h=1`
  - dias 2..7 → modelo `h=7`
  - dias 8..15 → modelo `h=15`
- exemplo de preenchimento baseado no último dia de treino, sem auto-preenchimento obrigatório

Fonte funcional: `.omx/specs/deep-interview-train-split-2024-ui-15d.md`.

## Evidência de baseline inspecionada

Na branch atual desta worker, o fluxo novo ainda **não está implementado**. A inspeção mostrou:

- `config/settings.py` mantém `DATE_RANGE["end"] = "2025-12-31"`
- `src/models/train.py` usa `CUTOFF_DATE = pd.Timestamp("2025-12-31")`
- `src/processing/cleaner.py` usa `HOLDOUT_CUTOFF = pd.Timestamp("2025-12-31")`
- `src/processing/merger.py` usa `HOLDOUT_CUTOFF = pd.Timestamp("2025-12-31")`
- `main.py` expõe apenas os modos legados `--train`, `--predict`, `--evaluate` e `--full`
- não existe entrypoint web/documentação raiz para a interface pedida

Conclusão: a feature precisa chegar preservando o pipeline legado e adicionando um caminho novo, isolado e documentado.

## Checklist de revisão de código

Ao integrar a implementação final, revisar os pontos abaixo:

### 1. Separação temporal

- [ ] nenhum dado de 2025 entra em treino/tuning
- [ ] o corte de treino está fixado em `2024-12-31`
- [ ] o holdout cobre somente `2025-01-01` a `2025-12-31`
- [ ] o fluxo legado continua disponível sem regressão

### 2. Isolamento de artefatos

- [ ] métricas, gráficos, CSVs e modelos do novo fluxo usam diretórios dedicados
- [ ] o fluxo novo não sobrescreve `data/processed` ou `models_saved` do pipeline atual sem namespace próprio
- [ ] nomes de arquivos deixam claro que pertencem ao recorte 2024/2025

### 3. Interface web

- [ ] interface abre localmente sem passos ocultos
- [ ] histórico real recente do boi gordo aparece em gráfico
- [ ] todas as variáveis exigidas pelo modelo ficam visíveis/editáveis
- [ ] existe exemplo de preenchimento com base no último dia de treino
- [ ] o envio gera curva diária completa de 1..15 dias
- [ ] a composição da curva usa `h=1`, `h=7` e `h=15` na regra da spec

### 4. Qualidade de manutenção

- [ ] não há duplicação desnecessária do pipeline legado
- [ ] helpers/configurações novas têm nomes explícitos para o fluxo 2024/2025
- [ ] regras de cutoff/holdout ficam centralizadas, não espalhadas em constantes mágicas
- [ ] documentação de execução e verificação acompanha a feature

## Evidência mínima esperada na verificação final

Quando a implementação for integrada, a documentação final deve registrar:

1. comando(s) usados para gerar o fluxo 2024/2025
2. comando(s) usados para subir a UI
3. caminhos dos artefatos gerados
4. saída resumida de:
   - lint
   - testes
   - checagem estática/typecheck aplicável ao projeto
5. evidência de funcionamento da curva 1..15 dias
6. evidência de que o pipeline legado continuou funcional

## Comandos de auditoria recomendados

Os comandos abaixo já são compatíveis com o estado atual do repositório e servem como checklist objetivo de revisão quando a implementação chegar.

### 1. Confirmar pontos de corte e holdout

```powershell
Get-ChildItem -Recurse -File src,config,main.py |
  Select-String -Pattern '2024-12-31|2025-01-01|2025-12-31|holdout|cutoff'
```

Objetivo:

- garantir que o novo fluxo use `2024-12-31` como limite de treino
- garantir que `2025` apareça apenas como validação/holdout no novo fluxo
- preservar o fluxo legado quando a separação for intencional

### 2. Confirmar artefatos isolados

```powershell
Get-ChildItem -Recurse -Directory data,models_saved |
  Select-Object FullName
```

Objetivo:

- verificar criação de namespace próprio para o fluxo 2024/2025
- evitar sobrescrita silenciosa do pipeline legado

### 3. Confirmar entrypoints disponíveis

```powershell
python main.py
```

Objetivo:

- checar ajuda/uso do pipeline CLI atual
- confirmar que o fluxo novo foi adicionado sem quebrar os modos existentes

### 4. Confirmar dependências/documentação da UI

```powershell
Get-Content requirements.txt
Get-ChildItem -Recurse -File | Select-String -Pattern 'streamlit|gradio|flask|fastapi'
```

Objetivo:

- identificar a stack web escolhida
- confirmar que a documentação final explica exatamente como subir a interface

## Observações de baseline sobre tooling

Durante a auditoria inicial desta branch:

- não foi encontrado `pytest.ini`, `pyproject.toml` ou configuração dedicada de lint/typecheck
- a pasta `Tests/` existe, mas hoje contém scripts utilitários/debug, não uma suíte automatizada formal
- o entrypoint principal documentado do projeto continua sendo `python main.py`

Implicação para a revisão final:

- se a implementação adicionar testes automatizados, o comando exato precisa ser documentado
- se a implementação adicionar lint/typecheck, os comandos e o escopo devem entrar na evidência final
- se não houver lint/typecheck formais, isso deve ser declarado explicitamente na entrega em vez de omitido

## Riscos já identificados

- uso de constantes hardcoded com `2025-12-31` em múltiplos módulos
- chance de misturar artefatos novos com diretórios já usados pelo pipeline principal
- ausência atual de convenção documentada para subir/operar a futura interface web

## Próximo passo desta lane

Assim que a implementação e os testes das outras lanes forem integrados, complementar este documento com:

- caminhos reais dos arquivos alterados
- instruções exatas de execução
- evidência concreta de verificação
- observações finais de qualidade/código
