import pandas as pd
import string
import nltk
import joblib
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from deep_translator import GoogleTranslator

# Garante que os dados do NLTK estão disponíveis (necessário em deploys novos)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Aqui foi definido o titulo da aba do navegador ecentralizou o conteudo, em seguida verifica se a imagem 
# ibmrexiste e carrega
st.set_page_config(page_title="Detector de SPAM - IA", page_icon="🤖", layout="centered")

if os.path.exists("IBMR.jpg"):
    st.image("IBMR.jpg", use_container_width=True)

# esse comando é para permitir colocar codigo html e css direto no python
# nomes do grupo edo professor
st.markdown("<h1 style='text-align: center;'>🤖 Detector de SPAM com Inteligência Artificial</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: 10px; margin-bottom: 20px;'>
    <h3 style='margin-bottom: 15px;'><b>Trabalho A3 — UC Inteligência Artificial</b></h3>
    <p style='margin: 4px 0; font-size: 16px;'><b>Professor:</b> Rogério Bailly / 2026-1</p>
    <p style='margin: 4px 0; font-size: 16px;'><b>Curso:</b> Ciência da Computação</p>
    <p style='margin: 4px 0; font-size: 16px;'><b>Grupo:</b> Caio  Campos / Pedro Gomes/ Rafael Couto/ Victor César Corrêa</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# Aqui colocamos o pipeline de treinamento do (carrega o dataset, limpa o texto, aplica
# TF-IDF e treina o Naive Bayes)
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def limpar_texto(texto):
    texto = texto.lower()
    texto = "".join([char for char in texto if char not in string.punctuation])
    palavras = nltk.word_tokenize(texto)
    return " ".join([p for p in palavras if p not in stopwords.words('english')])

@st.cache_resource
def inicializar_e_treinar_ia():
    # Carrega do disco se já foi treinado antes (evita re-treinar a cada restart local)
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        modelo = joblib.load(MODEL_PATH)
        vetorizador = joblib.load(VECTORIZER_PATH)
        return modelo, vetorizador

    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'texto']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df['texto_limpo'] = df['texto'].apply(limpar_texto)
    X_treino, _, y_treino, _ = train_test_split(df['texto_limpo'], df['label'], test_size=0.2, random_state=42)

    vetorizador = TfidfVectorizer()
    X_treino_vetorizado = vetorizador.fit_transform(X_treino)

    modelo = MultinomialNB()
    modelo.fit(X_treino_vetorizado, y_treino)

    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(vetorizador, VECTORIZER_PATH)

    return modelo, vetorizador

modelo_ia, vetorizador = inicializar_e_treinar_ia()


# Criar a caixa de texto para escrever a mensagem e o botão para ativar o codigo de analise
st.subheader("✉️ Analisador de Mensagens ")
mensagem_usuario = st.text_area("Cole ou digite o texto do e-mail aqui:", height=150, 
                                placeholder="Ex: GANHOU! Você foi selecionado para receber um prêmio...")

if st.button("Analisar Mensagem", type="primary"):
    if mensagem_usuario.strip() == "":
        st.warning("⚠️ Por favor, digite uma mensagem para que a IA possa analisar.")
    else:
        with st.spinner("A IA está analisando o contexto da mensagem..."):
            try:
                # Traduz automaticamente de qualquer idioma para o inglês
                texto_traduzido = GoogleTranslator(source='auto', target='en').translate(mensagem_usuario)
                
                # Passa o texto traduzido pelo pipeline da IA
                msg_limpa = limpar_texto(texto_traduzido)
                msg_vetorizada = vetorizador.transform([msg_limpa])
                predicao = modelo_ia.predict(msg_vetorizada)[0]
                
                # Exibe o resultado
                st.markdown("### Resultado da Avaliação:")
                if predicao == 1:
                    st.error("🚨 **SPAM DETECTADO!** Esta mensagem possui alto risco de ser uma fraude, phishing ou propaganda abusiva.")
                else:
                    st.success("🍏 **MENSAGEM CONFIÁVEL (HAM).** O modelo identificou padrões de uma comunicação legítima e segura.")
                    
                # Nota de rodapé acadêmica mostrando a tradução em tempo real
                st.caption(f"**Nota técnica (Tradução em tempo real para o modelo):** *\"{texto_traduzido}\"*")
                
            except Exception as e:
                st.error(f"Erro ao processar a mensagem: {e}")