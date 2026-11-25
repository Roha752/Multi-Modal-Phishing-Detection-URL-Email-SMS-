
import re
import math
import joblib
import numpy as np
import streamlit as st

from scipy.sparse import hstack, csr_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PhishGuard AI | Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d19;
    --panel: #101a2c;
    --border: #25334b;
    --cyan: #4de2d0;
    --purple: #9b8afb;
    --muted: #9aabc2;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #142849 0%, transparent 38%),
        radial-gradient(ellipse at 90% 20%, #241a46 0%, transparent 35%),
        var(--bg);
    color: #f1f5ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    text-align: center;
    padding: 34px 15px 28px;
}

.hero-icon {
    display: inline-block;
    font-size: 39px;
    padding: 15px;
    border: 1px solid #375d72;
    border-radius: 22px;
    background: linear-gradient(135deg, #153a4a, #202044);
    box-shadow: 0 0 38px #30d9d01c;
}

.eyebrow {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-top: 20px;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 52px);
    line-height: 1.13;
    letter-spacing: -2px;
    font-weight: 800;
    margin: 13px 0;
    color: #f6f8ff;
}

.gradient-text {
    background: linear-gradient(90deg, #4de2d0, #a99aff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a7b5cb;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
    margin: 0 auto;
}

.section-label {
    color: #e8efff;
    font-size: 19px;
    font-weight: 750;
    margin: 18px 0 12px;
}

.feature {
    background: linear-gradient(145deg, #111d31, #0c1424);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 21px;
    min-height: 145px;
    margin-bottom: 10px;
}

.feature-icon {
    font-size: 26px;
    margin-bottom: 12px;
}

.feature h3 {
    font-size: 16px;
    margin: 0 0 8px;
    color: #f1f5ff;
}

.feature p {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.65;
    margin: 0;
}

.panel {
    background: linear-gradient(145deg, #111c30, #0d1526);
    border: 1px solid #293953;
    border-radius: 22px;
    padding: 25px;
    margin: 8px 0 20px;
}

div[data-testid="stTextArea"] textarea {
    background: #091222 !important;
    color: #edf5ff !important;
    border: 1px solid #34445f !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #4de2d0 !important;
    box-shadow: 0 0 0 1px #4de2d0 !important;
}

.stButton > button {
    border: 1px solid #58e8d8;
    border-radius: 12px;
    min-height: 52px;
    font-size: 15px;
    font-weight: 800;
    color: #07131e;
    background: linear-gradient(100deg, #4de2d0, #94f0d6);
    transition: all .2s ease;
}

.stButton > button:hover {
    color: #07131e;
    border-color: #b5fff4;
    box-shadow: 0 0 25px #4de2d044;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: #0a1425;
    border: 1px solid #293953;
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #a7b5cb;
}

div[data-testid="stMetricValue"] {
    color: #f4f7ff;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    margin: 14px 0;
    border: 1px solid #33435c;
    background: #101a2c;
}

.result-danger {
    border-color: #a84457;
    background: linear-gradient(135deg, #301725, #151726);
}

.result-safe {
    border-color: #267f72;
    background: linear-gradient(135deg, #102e2b, #111b2b);
}

.result-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 9px;
}

.result-copy {
    color: #bdc9da;
    font-size: 14px;
    line-height: 1.8;
}

.footer {
    text-align: center;
    color: #7f90aa;
    font-size: 12px;
    line-height: 1.8;
    padding: 25px 5px 5px;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4de2d0, #a18bff);
}

hr {
    border-color: #25334b;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero {
        padding-top: 20px;
    }
    .panel {
        padding: 17px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <div class="eyebrow">AI-POWERED CYBERSECURITY</div>
    <h1>PhishGuard <span class="gradient-text">AI</span></h1>
    <p>
        Detect suspicious URLs, phishing emails, and SMS messages
        with an intelligent machine-learning security assistant.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">🌐</div>
        <h3>URL Analysis</h3>
        <p>Inspect submitted URLs for patterns associated with phishing.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">✉️</div>
        <h3>Email Screening</h3>
        <p>Analyze email text for suspicious language and patterns.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">📱</div>
        <h3>SMS Protection</h3>
        <p>Check text messages for possible phishing indicators.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_models():
    vec = joblib.load("tfidf_vectorizer (2).joblib")
    clf = joblib.load("phishing_rf_model (1).joblib")
    return vec, clf


def calc_entropy(s):
    if not s:
        return 0.0

    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_url_features(text, is_url):
    if not is_url:
        return np.zeros((1, 11), dtype=np.float32)

    return np.array([[
        1,
        len(text),
        text.count("."),
        text.count("-"),
        text.count("/"),
        sum(c.isdigit() for c in text),
        int(text.lower().startswith("https://")),
        int(text.lower().startswith("http://")),
        int("@" in text),
        int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", text))),
        calc_entropy(text)
    ]], dtype=np.float32)


try:
    vectorizer, rf_model = load_models()

except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(
        "Check that both joblib model files are uploaded "
        "to the same directory as app.py."
    )
    st.stop()


# ---------------- ANALYZER PANEL ----------------
st.markdown('<div class="section-label">🔎 Security Scanner</div>',
            unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown("#### What would you like to scan?")
input_mode = st.radio(
    "Select input type",
    ["🌐 URL", "✉️ Email", "📱 SMS"],
    horizontal=True,
    label_visibility="collapsed"
)

mode_map = {
    "🌐 URL": "url",
    "✉️ Email": "email",
    "📱 SMS": "sms"
}

selected_type = mode_map[input_mode]

placeholders = {
    "url": "https://example.com/login",
    "email": "Dear customer, verify your account...",
    "sms": "Your account needs verification. Review this message..."
}

labels = {
    "url": "Enter website URL",
    "email": "Paste email content",
    "sms": "Paste SMS message"
}

sample_text = st.text_area(
    labels[selected_type],
    height=150,
    placeholder=placeholders[selected_type],
    key="scan_input"
)

st.caption(
    "🔒 Submit only content you are authorized to analyze. "
    "Avoid including passwords, verification codes, or private information."
)

threshold = st.slider(
    "Detection threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help=(
        "Adjusts the cutoff for the model's predicted phishing probability. "
        "This changes the decision threshold, not model accuracy."
    )
)

scan_clicked = st.button(
    "🛡️  ANALYZE FOR THREATS",
    type="primary",
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- ANALYSIS ----------------
if scan_clicked:

    if not sample_text.strip():
        st.warning("Please enter a URL, email, or SMS message first.")

    else:
        content = sample_text.strip()
        is_url = selected_type == "url"

        if is_url and not re.match(
            r"^(https?://|www\.)\S+$", content, re.IGNORECASE
        ):
            st.warning(
                "Enter a URL beginning with http://, https://, or www. "
                "For best results, include the full URL."
            )

        else:
            try:
                with st.spinner("Analyzing content with the ML model..."):

                    X_text = vectorizer.transform([content])

                    X_num = csr_matrix(
                        extract_url_features(content, is_url)
                    )

                    X_full = hstack([X_text, X_num]).tocsr()

                    # Assumes the trained model's phishing class is label 1.
                    classes = list(rf_model.classes_)

                    if 1 not in classes:
                        st.error(
                            "Model class label 1 was not found. "
                            "Check your training labels before deployment."
                        )
                        st.stop()

                    phishing_index = classes.index(1)

                    phishing_prob = float(
                        rf_model.predict_proba(X_full)[0][phishing_index]
                    )

                    is_phishing = phishing_prob >= threshold

                st.markdown("---")
                st.markdown(
                    '<div class="section-label">📊 Scan Results</div>',
                    unsafe_allow_html=True
                )

                r1, r2 = st.columns(2)

                with r1:
                    st.metric(
                        "Input Type",
                        selected_type.upper()
                    )

                with r2:
                    st.metric(
                        "Predicted Phishing Probability",
                        f"{phishing_prob * 100:.1f}%"
                    )

                st.progress(
                    max(0.0, min(1.0, phishing_prob)),
                    text=f"Model phishing score: {phishing_prob * 100:.1f}%"
                )

                if is_phishing:
                    st.markdown(f"""
                    <div class="result-card result-danger">
                        <div class="result-title">
                            🚨 Potential Phishing Detected
                        </div>
                        <div class="result-copy">
                            The model's phishing probability is above
                            your selected threshold. Treat this content
                            with caution and verify its source independently.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(
                        "Do not enter credentials or open suspicious links "
                        "until you have independently verified the source."
                    )

                else:
                    st.markdown(f"""
                    <div class="result-card result-safe">
                        <div class="result-title">
                            🟢 Below Detection Threshold
                        </div>
                        <div class="result-copy">
                            The model's phishing probability is below
                            your selected threshold. This does not prove
                            the content is legitimate or safe.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("🧠 Understanding your result"):
                    st.write(
                        f"**Selected input:** {selected_type.upper()}"
                    )
                    st.write(
                        f"**Threshold:** {threshold * 100:.0f}%"
                    )
                    st.write(
                        f"**Model probability:** "
                        f"{phishing_prob * 100:.2f}%"
                    )
                    st.write(
                        "**Interpretation:** The result comes from the "
                        "trained model and is not a guarantee of safety. "
                        "Its reliability depends on the training data, "
                        "feature compatibility, and evaluation performance."
                    )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info(
                    "Check that the vectorizer, model, and numerical "
                    "features match the format used during training."
                )


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div class="footer">
    <strong>🛡️ PhishGuard AI</strong><br>
    Machine-learning-assisted phishing screening<br><br>
    ⚠️ This tool provides an indicative prediction, not a security guarantee.
    Always verify suspicious content independently.<br>
    © PhishGuard AI · Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)
