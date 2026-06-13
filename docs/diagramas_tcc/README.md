# Diagramas do TCC — Projeto Boi Gordo

Esta pasta contém diagramas em **Mermaid editável** para explicar o código do projeto para professor e banca.

## Arquivos

1. `01_arquitetura_componentes.mmd` — visão geral dos componentes do sistema.
2. `02_fluxo_dados.mmd` — caminho dos dados desde as fontes brutas até artefatos finais.
3. `03_treino_validacao.mmd` — comportamento do treino acadêmico e validação em 2025.
4. `04_sequencia_predicao_interface.mmd` — sequência da interface Streamlit carregando histórico e previsões exportadas.
5. `05_atualizacao_producao.mmd` — atualização diária de produção, MySQL e versionamento.
6. `06_classes_estruturas.mmd` — classes e estruturas reais usadas no projeto.

## Como editar

Os arquivos `.mmd` podem ser abertos em editores com suporte a Mermaid, como VS Code com extensão Mermaid, Mermaid Live Editor ou ferramentas compatíveis.

## Renderização

A fonte principal são os arquivos `.mmd`. Se a ferramenta `mmdc` estiver instalada localmente, é possível gerar imagens, por exemplo:

```bash
mmdc -i 01_arquitetura_componentes.mmd -o 01_arquitetura_componentes.svg
```

Neste repositório, a renderização é tratada como etapa opcional para evitar adicionar dependências apenas para documentação.
