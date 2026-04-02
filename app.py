# ==============================
# IMPORT LIBRARIES
# ==============================
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🔬",
    layout="wide"
)

# ==============================
# LOAD MODEL (CACHED FOR PERFORMANCE)
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

# ==============================
# SESSION STATE (STORE USER INPUT & RESULT)
# ==============================
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "result" not in st.session_state:
    st.session_state.result = None

# ==============================
# TEXT PREPROCESSING FUNCTION
# ==============================
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text):
    text = str(text).lower()                         # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # Remove special chars
    words = [w for w in text.split() if w not in stop_words]  # Remove stopwords
    return " ".join(words)

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict(text):
    clean_text = preprocess(text)
    vector = tfidf.transform([clean_text])
    pred = model.predict(vector)[0]

    # Convert numeric labels to text
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if isinstance(pred, (int, float)):
        pred = label_map[int(pred)]

    # Get probability (if supported)
    try:
        prob = model.predict_proba(vector)[0]
    except:
        prob = None

    return pred, prob

# ==============================
# BACKGROUND COLOR FUNCTION
# ==============================
def set_bg(color):
    st.markdown(f"""
    <style>
    .stApp {{
        background: {color};
        transition: all 0.5s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

# Default background
set_bg("#03070f")

# ==============================
# STYLED CENTERED HEADER
# ==============================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    font-family: 'Segoe UI', 'Poppins', sans-serif;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    font-weight: 500;
    font-family: 'Segoe UI', 'Poppins', sans-serif;
    color: #b0b0b0;
    margin-top: -10px;
}
</style>

<div class="title">Sentiment Analyzer</div>
<div class="subtitle">Analyze emotions in text using Machine Learning & NLP</div>
""", unsafe_allow_html=True)

# ==============================
# ABOUT SECTION (NEW FEATURE)
# ==============================
st.markdown("""
### 🧠 What This App Does
This AI tool analyzes text and detects its **emotional tone**.

It classifies input into:
- 😊 Positive
- 😐 Neutral
- 😡 Negative


---
---
""")

# ==============================
# EXAMPLE BUTTONS (NEW FEATURE)
# ==============================
st.markdown("### 💡 Try Examples")

col1, col2, col3, col4 = st.columns(4)

if col1.button("😊 Positive Example"):
    st.session_state.input_text = "This product is absolutely amazing! I loved it so much."
    st.rerun()

if col2.button("😡 Negative Example"):
    st.session_state.input_text = "Worst experience ever. Totally disappointed and frustrated."
    st.rerun()

if col3.button("😐 Neutral Example"):
    st.session_state.input_text = "The product is okay, it works as expected."
    st.rerun()

# 🆕 MIXED REVIEW BUTTON
if col4.button("🤔 Mixed Review"):
    st.session_state.input_text = "The design is really good and looks premium, but the battery life is very poor and disappointing."
    st.rerun()
    
# ==============================
# INPUT TEXT AREA
# ==============================
user_input = st.text_area(
    "",
    height=200,
    placeholder="Type your review here...",
    key="input_text"
)

# ==============================
# LIVE TEXT STATS (NEW FEATURE)
# ==============================
char_count = len(user_input)
word_count = len(user_input.split())

st.markdown(f"""
<div style="display:flex;justify-content:space-between;font-size:13px;color:gray;">
<span>📝 {char_count} characters</span>
<span>📊 {word_count} words</span>
</div>
""", unsafe_allow_html=True)

# ==============================
# BUTTONS
# ==============================
colA, colB = st.columns([5,1])

with colA:
    analyze = st.button("✨ ANALYZE SENTIMENT", use_container_width=True)

with colB:
    clear = st.button("✕", use_container_width=True)

# ==============================
# CLEAR FUNCTION
# ==============================
if clear:
    st.session_state.input_text = ""
    st.session_state.result = None
    set_bg("#03070f")
    st.rerun()

# ==============================
# ANALYZE LOGIC
# ==============================
if analyze:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text")
    else:
        prediction, probs = predict(user_input)
        st.session_state.result = (prediction, probs, user_input)

# ==============================
# RESULT DISPLAY
# ==============================
if st.session_state.result:

    prediction, probs, text = st.session_state.result

    # Change background color
    bg_map = {
        "Positive": "linear-gradient(135deg, #052e1a, #00c864)",
        "Negative": "linear-gradient(135deg, #2a0505, #ff4444)",
        "Neutral":  "linear-gradient(135deg, #2a2300, #f5c400)"
    }
    set_bg(bg_map.get(prediction, "#03070f"))

    st.markdown("## 📊 Result")

    # Show result
    if prediction == "Positive":
        st.success("😊 Positive Sentiment")
    elif prediction == "Negative":
        st.error("😡 Negative Sentiment")
    else:
        st.info("😐 Neutral Sentiment")

    # ==============================
    # CONFIDENCE SCORES
    # ==============================
    if probs is not None:
        st.subheader("📈 Confidence Scores")
        labels = ["Negative", "Neutral", "Positive"]

        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[i]*100:.2f}%")
            st.progress(float(probs[i]))

    # ==============================
    # TEXT STATISTICS (NEW FEATURE)
    # ==============================
    st.subheader("📊 Text Statistics")

    col1, col2 = st.columns(2)
    col1.metric("Characters", len(text))
    col2.metric("Words", len(text.split()))

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.write("🚀 Built with Machine Learning, NLP & Streamlit")
