# Projeto de Previsão de Churn - Telecom X (Parte 2)

## 1\. Visão Geral do Projeto

Este projeto foca na construção e avaliação de modelos de machine learning para prever a evasão de clientes (*churn*) da empresa Telecom X. A partir de um conjunto de dados previamente tratado, realizamos uma análise exploratória para identificar os principais fatores que influenciam o cancelamento de serviços. Em seguida, treinamos e comparamos dois modelos de classificação (Regressão Logística e Random Forest) para determinar qual oferece a melhor performance preditiva. O objetivo final é fornecer insights acionáveis e uma ferramenta capaz de identificar proativamente os clientes com maior risco de churn.

## 2\. Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
├── solução_telecom_x_parte1.ipynb    # Notebook com o tratamento inicial e limpeza dos dados.
├── dados_tratados.csv                # Arquivo CSV gerado pela Parte 1, utilizado como entrada.
├── TelecomX_parte2.ipynb             # Notebook principal com a análise, modelagem e avaliação.
└── README.md                         # Este arquivo de documentação.
```

## 3\. Metodologia

O fluxo de trabalho foi dividido em duas fases principais: preparação dos dados e modelagem.

### 3.1. Preparação dos Dados

A preparação dos dados foi uma etapa crucial para garantir a qualidade e o formato adequado para os algoritmos de machine learning.

  * **Tratamento de Dados Categóricos:** Variáveis como `gender`, `InternetService`, `Contract` e `PaymentMethod` foram transformadas em formato numérico usando a técnica de **One-Hot Encoding**. Isso cria colunas binárias (0 ou 1) para cada categoria, permitindo que os modelos matemáticos as processem.
  * **Normalização de Dados Numéricos:** As colunas numéricas (`customer.tenure`, `account.Charges.Monthly`, etc.) foram padronizadas usando `StandardScaler`. Este processo ajusta a escala das variáveis para que tenham média 0 e desvio padrão 1, evitando que características com valores maiores (como `TotalCharges`) dominem indevidamente o modelo.
  * **Balanceamento de Classes:** O dataset original era desbalanceado, com 73% dos clientes permanecendo e apenas 27% cancelando. Para evitar que os modelos aprendessem a prever apenas a classe majoritária, aplicamos a técnica **SMOTE (Synthetic Minority Oversampling Technique)** no conjunto de treino. Isso equilibrou as classes, gerando dados sintéticos da classe minoritária (`Churn = Yes`) e melhorando a capacidade do modelo de identificar padrões de evasão.
  * **Divisão em Treino e Teste:** O dataset final foi dividido em 70% para treinamento dos modelos e 30% para teste, garantindo uma avaliação imparcial da performance em dados não vistos.

### 3.2. Modelos Utilizados

1.  **Regressão Logística:** Escolhido como um modelo de *baseline* por sua simplicidade, rapidez e alta interpretabilidade. Seus coeficientes nos ajudam a entender a direção e a força da influência de cada variável no churn.
2.  **Random Forest:** Um modelo de conjunto (*ensemble*) mais complexo e robusto, conhecido por sua alta precisão e por ser menos suscetível a overfitting. Ele nos permite identificar as variáveis mais importantes de forma geral, capturando também relações não-lineares.

## 4\. Análise Exploratória e Principais Insights

A análise de correlação e as visualizações dos dados revelaram padrões claros nos clientes que cancelam o serviço:

  * **Clientes com contrato "Mês a Mês"** são os que mais cancelam. A falta de um compromisso de longo prazo é o maior indicador de risco de churn.
  * **Clientes com pouco tempo de casa ("tenure")** tendem a cancelar com muito mais frequência. A lealdade aumenta significativamente com o tempo.
  * **Clientes com serviço de Fibra Óptica** têm uma taxa de churn maior do que os com DSL, o que pode indicar problemas de preço, qualidade ou concorrência mais acirrada neste segmento.
  * **Clientes com maiores cobranças mensais** apresentam uma maior tendência a cancelar, especialmente no início do contrato.

 *Exemplo: Gráfico de Churn por Tipo de Contrato, mostrando a alta taxa no plano "Month-to-month".*

## 5\. Avaliação dos Modelos

O modelo **Random Forest** superou a Regressão Logística em todas as métricas, sendo a escolha recomendada para implementação. A métrica de **Recall**, em particular, é crucial neste contexto de negócio, pois queremos identificar o máximo possível de clientes que realmente irão cancelar. O Random Forest alcançou **86.6% de Recall**, significando que ele consegue "capturar" quase 9 em cada 10 clientes que estão prestes a sair.

| Métrica | Regressão Logística | **Random Forest (Vencedor)** |
| :--- | :---: | :---: |
| Acurácia | 78.41% | **83.41%** |
| Precisão | 75.39% | **80.68%** |
| Recall | 82.46% | **86.58%** |
| F1-Score | 0.7877 | **0.8353** |

## 6\. Recomendações Estratégicas

Com base nos fatores de churn identificados, as seguintes ações podem ser tomadas para aumentar a retenção de clientes:

1.  **Campanhas de Fidelização para Contratos Mensais:** Oferecer descontos ou benefícios (como upgrade de velocidade) para clientes de planos mensais que migrarem para contratos de 1 ou 2 anos.
2.  **Ações de Engajamento para Novos Clientes:** Criar um programa de acompanhamento focado nos primeiros 90 dias do cliente, oferecendo suporte proativo e garantindo que eles aproveitem ao máximo os serviços contratados.
3.  **Investigar a Satisfação com Fibra Óptica:** Realizar pesquisas de satisfação e analisar reclamações técnicas específicas dos clientes de fibra para identificar e corrigir possíveis falhas no serviço ou na política de preços.
4.  **Oferta de Serviços Adicionais:** Promover pacotes que incluam **Segurança Online** e **Suporte Técnico** para clientes que não os possuem, pois esses serviços aumentam o valor percebido e a dependência do ecossistema da empresa.

## 7\. Como Executar o Projeto

#### Pré-requisitos

Para executar o notebook `TelecomX_parte2.ipynb`, é necessário ter o Python instalado e as seguintes bibliotecas:

```bash
pip install pandas scikit-learn imbalanced-learn plotly
```

#### Execução

1.  Certifique-se de que o arquivo `dados_tratados.csv` (gerado pela Parte 1 do projeto) esteja no mesmo diretório que o notebook.
2.  Abra o `TelecomX_parte2.ipynb` em um ambiente Jupyter (como Jupyter Notebook, JupyterLab ou Google Colab).
3.  Execute as células sequencialmente para replicar a análise, o treinamento dos modelos e a avaliação dos resultados.

<!-- end list -->

```
```
