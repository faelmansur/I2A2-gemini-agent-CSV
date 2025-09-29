\# I2A2-gemini-agent-CSV



Um agente autônomo para análise de dados (EDA) em arquivos CSV usando LangChain e Google Gemini.



\## Funcionalidades

\- Upload de qualquer CSV.

\- Perguntas sobre descrição dos dados, padrões, anomalias, relações entre variáveis.

\- Geração de gráficos (histogramas, boxplots, scatter).

\- Memória para conclusões cumulativas.



\## Instalação

1\. Clone o repositório: `git clone https://github.com/seu-usuario/I2A2-gemini-agent-CSV.git`

2\. Instale as dependências: `pip install -r requirements.txt`

3\. Configure o `.env` com sua chave API (use `.env.example` como modelo).

4\. Rode o app: `streamlit run app.py`



\## Uso

Carregue um CSV e faça perguntas como "Existem valores atípicos?" ou "Quais conclusões você tem?".



\## Deploy

Disponível em: https://seu-app.streamlit.app



\## Segurança

O agente usa `allow\_dangerous\_code=True` para execução de código. Use em ambiente local seguro.



