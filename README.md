# Projeto DataView

## Sobre o projeto 
O DataView PY é um projeto de análise exploratória de dados (EDA) de vendas, desenvolvido em Python em um notebook. O notebook lê, limpa, transforma, analisa e visualiza um dataset de vendas, gerando métricas e insights.

## O que o projeto analisa 
- Receita total e volume de vendas por mês e trimestre
- Top produtos e categorias por receita
- Desempenho por região - Segmentação de clientes por nível de gasto (Bronze, Prata, Ouro) 
- Comparação entre os dados com e sem tratamento de outliers (v1 e v2)
- Exportação de relatórios em CSV e JSON

## Objetivo 
Praticar os principais conceitos:
- Lógica de programação com Python
- Variáveis, tipos de dados e operadores
- Condicionais (if, elif, else) e repetição (for, while)
- Funções com parâmetros, retorno e funções lambda
- Funções reutilizáveis
- Leitura e escrita de arquivos CSV e JSON
- Módulo datetime para manipulação de datas
- Expressões regulares com o módulo re
- Pandas: DataFrames, limpeza, groupby, filtros e transformações
- NumPy: arrays, operações vetorizadas, broadcasting
- Detecção e tratamento de outliers (IQR ou z-score)
- Matplotlib e Seaborn: gráficos, customização e exportação em PNG - Uso básico do GitHub

## Como executar
###  No Google Colab (recomendado)

 1. Faça upload do notebook dataview.ipynb
 2. Abra o arquivo carregado no Colab
 3. Execute as células na ordem, de cima para baixo
   
### Localmente com VS Code

 1. Instale o Python 3.10+ e o VS Code
 2. Instale as dependências: pip install pandas numpy matplotlib seaborn

## Estrutura do projeto
```text
projeto/ 
|-- data/ 
| |-- raw/                              # Dataset bruto gerado/baixado 
| |-- processed/ 
| | |-- v1_com_outliers/                # Dados de limpeza geral, outliers mantidos
| | |-- v2_outliers_tratado/            # Limpeza v1 + tratamento de outliers 
| |-- final/                            # Dataset escolhido para uso futuro 
|-- notebooks/ 
| |-- dataview.py                       # Notebook principal de EDA 
|-- outputs/
| |-- metricas_por_mes.csv 
| |-- segmentacao_clientes.csv 
| |-- estatisticas_gerais.json | 
|-- graficos/ 
|-- README.md 
```
## Decisão de versão (dados)

A análise final usa a versão v2 (outliers tratados) e A v1 (com outliers) fica preservada em 
data/processed/ para consulta futura.

## Ferramentas utilizadas 
- Python 3.10+
- Google Colab / VS Code
- Bibliotecas: pandas, numpy, matplotlib, seaborn, re, datetime, os, random
- GitHub para versionamento

