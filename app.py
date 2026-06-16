import pandas as pd
import string
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from deep_translator import GoogleTranslator

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="Detector de SPAM - IA", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    /* Layout base */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 760px;
    }

    /* Cabeçalho */
    .app-title {
        text-align: center;
        font-size: clamp(1.4rem, 4vw, 2rem);
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        text-align: center;
        font-size: clamp(0.8rem, 2vw, 0.95rem);
        color: #888;
        margin-bottom: 1.5rem;
    }

    /* Card de info do grupo */
    .info-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        font-size: clamp(0.8rem, 2vw, 0.92rem);
        line-height: 1.9;
    }
    .info-card p { margin: 0; }

    /* Barra de progresso */
    div[data-testid="stProgress"] > div {
        border-radius: 8px;
        height: 22px !important;
    }

    /* Boxes de resultado */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        font-size: 0.97rem;
    }

    /* Botão principal — largura total */
    div[data-testid="stButton"] > button[kind="primary"] {
        width: 100%;
        padding: 0.65rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 8px;
        letter-spacing: 0.03em;
    }

    /* Textarea */
    textarea {
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }

    hr { border-color: #333; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ────────────────────────────────────────────────────────────────
if os.path.exists("IBMR.jpg"):
    st.image("IBMR.jpg", use_container_width=True)

st.markdown("<div class='app-title'>🤖 Detector de SPAM com Inteligência Artificial</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Classificação em tempo real usando Naive Bayes + TF-IDF</div>", unsafe_allow_html=True)

st.markdown("""
<div class='info-card'>
    <p><strong>Trabalho A3 — UC Inteligência Artificial</strong></p>
    <p><strong>Professor:</strong> Rogério Bailly &nbsp;|&nbsp; 2026-1</p>
    <p><strong>Curso:</strong> Ciência da Computação</p>
    <p><strong>Grupo:</strong> Caio Campos · Pedro Gomes · Rafael Couto · Victor César Corrêa</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Regras de padrões (executadas antes do ML) ────────────────────────────────
_PADROES_SPAM = [
    r'(ganhou|premiado|selecionado).{0,30}(prêmio|brinde|vale|voucher)',
    r'clique\s+(aqui|no\s+link)',
    r'(grátis|gratuito).{0,20}(acesso|curso|ebook)',
    r'\bpix\b.{0,20}(ganhe|receba|resgate)',
    r'(whatsapp|zap).{0,20}(clique|acesse)',
    r'https?://\S+\.(xyz|top|click|online)',
]

def _detectar_por_regras(texto: str) -> bool:
    t = texto.lower()
    return any(re.search(p, t) for p in _PADROES_SPAM)

# ── Modelo ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def _limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(c for c in texto if c not in string.punctuation)
    palavras = nltk.word_tokenize(texto)
    return " ".join(p for p in palavras if p not in stopwords.words('english'))

@st.cache_resource
def inicializar_e_treinar_ia():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'texto']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['texto_limpo'] = df['texto'].apply(_limpar_texto)

    X_treino, _, y_treino, _ = train_test_split(
        df['texto_limpo'], df['label'], test_size=0.2, random_state=42
    )
    vetorizador = TfidfVectorizer()
    X_treino_vetorizado = vetorizador.fit_transform(X_treino)

    modelo = MultinomialNB()
    modelo.fit(X_treino_vetorizado, y_treino)

    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(vetorizador, VECTORIZER_PATH)
    return modelo, vetorizador

modelo_ia, vetorizador = inicializar_e_treinar_ia()

THRESHOLD_SPAM = 0.35
THRESHOLD_SUSPEITO = 0.20

# ── Interface de análise ──────────────────────────────────────────────────────
st.subheader("✉️ Analisador de Mensagens")

mensagem_usuario = st.text_area(
    "Cole ou digite o texto da mensagem aqui:",
    height=150,
    placeholder="Ex: GANHOU! Você foi selecionado para receber um prêmio exclusivo. Clique aqui!",
)

if st.button("Analisar Mensagem", type="primary"):
    if not mensagem_usuario.strip():
        st.warning("Por favor, digite uma mensagem para que a IA possa analisar.")
    else:
        with st.spinner("Analisando o contexto da mensagem..."):
            try:
                st.markdown("### Resultado da Avaliação:")

                # Camada 1 — regras de padrões clássicos
                if _detectar_por_regras(mensagem_usuario):
                    st.progress(1.0, text="Probabilidade de SPAM: 100% (padrão clássico detectado)")
                    st.error(
                        "🚨 **SPAM DETECTADO!**  \n"
                        "Padrão típico de fraude, phishing ou propaganda abusiva identificado pelas regras do sistema."
                    )
                    st.caption("Classificado por detecção de padrões (regex) — sem necessidade de processar pelo modelo ML.")

                # Camada 2 — Naive Bayes com threshold ajustado
                else:
                    texto_traduzido = GoogleTranslator(source='auto', target='en').translate(mensagem_usuario)
                    msg_limpa = _limpar_texto(texto_traduzido)
                    msg_vetorizada = vetorizador.transform([msg_limpa])
                    prob_spam = float(modelo_ia.predict_proba(msg_vetorizada)[0][1])

                    st.progress(prob_spam, text=f"Probabilidade de SPAM: {prob_spam:.1%}")

                    if prob_spam >= THRESHOLD_SPAM:
                        st.error(
                            "🚨 **SPAM DETECTADO!**  \n"
                            "A IA identificou alto risco — esta mensagem apresenta padrões de fraude, phishing ou propaganda abusiva."
                        )
                    elif prob_spam >= THRESHOLD_SUSPEITO:
                        st.warning(
                            "⚠️ **Mensagem suspeita.**  \n"
                            "Probabilidade intermediária de spam. Verifique com atenção antes de clicar em links ou responder."
                        )
                    else:
                        st.success(
                            "🍏 **Mensagem confiável (HAM).**  \n"
                            "O modelo identificou padrões de uma comunicação legítima e segura."
                        )

                    st.caption(f"**Nota técnica — tradução enviada ao modelo:** *\"{texto_traduzido}\"*")

            except Exception as e:
                st.error(f"Erro ao processar a mensagem: {e}")
