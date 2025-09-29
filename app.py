import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents import create_csv_agent
import time
import tempfile
import sys

try:
    import tabulate
except ImportError:
    st.warning("⚠️ O pacote 'tabulate' não está instalado. Algumas tabelas podem não ser formatadas adequadamente. Instale com 'pip install tabulate' para melhor experiência.")
    tabulate = None

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração da Página ---
st.set_page_config(page_title="Explorador de Dados CSV", page_icon="🔍", layout="wide")
st.title("🔍 Explorador de Dados CSV com Gemini")

# --- Verificação da Chave API ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ Chave API ausente! Crie um `.env` com `GOOGLE_API_KEY=sua-chave`.")
    st.stop()
else:
    st.write("🔑 Chave API detectada. Validando...")

# --- Inicialização do Modelo e Memória ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="memory_log", return_messages=True)
    st.session_state.conclusions = []
    st.session_state.attempts = 0
    st.session_state.chat_history = []  # Inicialização corrigida aqui

# Tentar inicializar o modelo com opções disponíveis
try:
    available_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025"]
    llm = None
    for model in available_models:
        try:
            st.write(f"Tentando inicializar modelo: {model}")
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            # Teste inicial do modelo
            test_response = llm.invoke("Teste: diga 'OK'")
            if "OK" in str(test_response):
                st.success(f"✅ Gemini pronto para análise com modelo {model}!")
                break
            else:
                st.write(f"❌ Resposta de teste falhou com {model}")
        except Exception as e:
            st.write(f"❌ Falha com {model}: {str(e)}")
            continue
    if llm is None:
        st.error("❌ Nenhum modelo Gemini disponível. Verifique sua chave API em https://aistudio.google.com/api-keys, ative faturamento no Google Cloud, ou consulte https://ai.google.dev/gemini-api/docs/models.")
        st.stop()
except Exception as e:
    st.error(f"❌ Falha geral na inicialização: {e}. Verifique a configuração.")
    st.stop()

# --- Interface do Usuário ---
st.sidebar.header("Guia Rápido")
st.sidebar.info(
    """
    1. **Obtenha sua chave API** em [Google AI Studio](https://aistudio.google.com/).
    2. **Configure `.env`** com `GOOGLE_API_KEY`.
    3. **Carregue um CSV**.
    4. **Faça perguntas** como:
       - 'Quais os tipos de dados?'
       - 'Existem padrões ou tendências temporais?'
       - 'Quais os valores mais frequentes ou menos frequentes?'
       - 'Existem valores atípicos nos dados? '
       - 'Quais conclusões você tem?'
    5. Gráficos aparecem se gerados.
    """
)

st.write("## 1. Carregue seu arquivo CSV")
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file:
    try:
        # Salvar o arquivo temporariamente para passar o caminho
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Carregar e pré-processar o DataFrame
        df = pd.read_csv(tmp_path)
        if df.empty:
            raise ValueError("CSV vazio detectado!")

        # Inspecionar a primeira coluna e forçar conversão
        first_column = df.columns[0]
        st.write(f"Tipos de dados iniciais - Coluna 0 ({first_column}): {df[first_column].dtype}")
        df[first_column] = pd.to_numeric(df[first_column], errors='coerce')  # Converte para numérico, NaN para inválidos
        df = df.fillna(df.mean(numeric_only=True))  # Preenche NaN com a média das colunas numéricas

        # Verificar tipos após conversão
        st.write("Tipos de dados após conversão:", df.dtypes)

        st.success(f"✅ CSV carregado: {len(df)} linhas, {len(df.columns)} colunas.")
        st.write("### Visualização Inicial")
        st.dataframe(df.head(5))

        with st.expander("📋 Detalhes do Dataset"):
            st.write("Tipos de dados:", df.dtypes)
            if tabulate:
                st.write("Estatísticas básicas:", df.describe().to_string())
            else:
                st.write("Estatísticas básicas (sem formatação avançada):", df.describe())

        st.write("## 2. Faça sua pergunta")
        user_query = st.chat_input("Pergunte algo sobre os dados...")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Processando sua solicitação..."):
                # Criar agente com o caminho do arquivo temporário e allow_dangerous_code
                agent = create_csv_agent(
                    llm=llm,
                    path=tmp_path,
                    memory=st.session_state.memory,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True  # Habilitado para permitir execução de código
                )
                enhanced_query = (
                    f"Você é um especialista em análise de dados. Use o histórico para insights. "
                    f"Responda em português, de forma clara e estruturada, seguindo este formato: "
                    f"- 'Thought: [seu raciocínio]' "
                    f"- 'Action: [código Python usando python_repl_ast]' "
                    f"- 'Action Input: [código Python a ser executado]' "
                    f"Após executar a ação, forneça a 'Final Answer' com a análise e inclua 'INSIGHT: [descrição clara e completa do insight]' obrigatoriamente. "
                    f"Para perguntas sobre valores atípicos, crie um boxplot (ex.: sns.boxplot para 'Amount' ou por 'Class') usando matplotlib ou seaborn, salvando como 'data_viz.png' (plt.savefig('data_viz.png'); plt.close()), "
                    f"e analise outliers com estatísticas (média, mediana, desvio padrão) na 'Final Answer'. "
                    f"Para padrões temporais, use um gráfico de 'Time' vs. 'Class'. "
                    f"Para relações, use scatter plots ou tabelas cruzadas. Para descrições, use df.describe() ou histogramas. "
                    f"Sempre gere uma resposta completa, sem placeholders ou mensagens adicionais. Pergunta: {user_query}"
                )

                st.session_state.attempts += 1
                attempt_msg = st.empty()
                attempt_msg.info(f"Tentativa {st.session_state.attempts} em andamento...")

                max_attempts = 7  # Aumentado para lidar com falhas temporárias
                while st.session_state.attempts <= max_attempts:
                    try:
                        response = agent.invoke({"input": enhanced_query})
                        if st.session_state.attempts > 1:
                            attempt_msg.success(f"Sucesso na tentativa {st.session_state.attempts}!")
                        else:
                            attempt_msg.empty()

                        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
                        with st.chat_message("assistant"):
                            st.markdown(response["output"])

                        viz_path = os.path.join(os.getcwd(), "data_viz.png")
                        if os.path.exists(viz_path):
                            st.image(viz_path, caption="Visualização gerada")
                            os.remove(viz_path)
                        else:
                            st.info("Nenhuma visualização gerada.")

                        output_text = response["output"]
                        if "INSIGHT:" in output_text.upper():
                            insight_part = output_text.split("INSIGHT:")[-1].strip()
                            insight = insight_part.split("\n")[0] if "\n" in insight_part else insight_part  # Pega a primeira linha ou todo o texto
                            if insight and insight.lower() != "[resumo]":  # Evita placeholders
                                st.session_state.conclusions.append(insight)
                                st.success(f"📝 Novo insight adicionado: {insight}")

                        st.session_state.memory.save_context({"input": user_query}, {"output": response["output"]})
                        break

                    except Exception as e:
                        if "quota exceeded" in str(e).lower() or "429" in str(e):
                            retry_delay = int(e.retry_delay.seconds) if hasattr(e, 'retry_delay') and e.retry_delay else min(st.session_state.attempts * 30, 120)
                            attempt_msg.error(f"Quota atingida. Aguardando {retry_delay}s (Tentativa {st.session_state.attempts}/{max_attempts}). Consulte https://ai.google.dev/gemini-api/docs/rate-limits.")
                            time.sleep(retry_delay)
                            st.session_state.attempts += 1
                        elif "No generation chunks were returned" in str(e):
                            attempt_msg.error(f"Sem resposta do modelo. Tentativa {st.session_state.attempts}/{max_attempts}. Verifique a cota ou conexão.")
                            time.sleep(min(st.session_state.attempts * 15, 60))  # Aumenta delay progressivo
                            st.session_state.attempts += 1
                        else:
                            st.error(f"❌ Erro: {e}")
                            attempt_msg.empty()
                            break

                if st.session_state.attempts > max_attempts:
                    st.error("❌ Limite de tentativas atingido. Verifique sua cota em https://ai.google.dev/gemini-api/docs/rate-limits ou ative faturamento no Google Cloud.")

                # Remover o arquivo temporário após o uso
                import os
                os.unlink(tmp_path)

            with st.expander("🕒 Histórico de Análise"):
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if st.session_state.conclusions:
                st.write("## 📚 Conclusões Registradas")
                for i, concl in enumerate(st.session_state.conclusions, 1):
                    st.write(f"{i}. {concl}")

    except ValueError as ve:
        st.error(f"❌ Erro de dados: {ve}")
    except pd.errors.EmptyDataError:
        st.error("❌ Arquivo CSV inválido ou vazio.")
    except Exception as e:
        st.error(f"❌ Erro inesperado: {e}")

st.markdown("---")
st.markdown("Desenvolvido para EDA de CSVs. Toque pessoal: Análise interativa com histórico visual.")