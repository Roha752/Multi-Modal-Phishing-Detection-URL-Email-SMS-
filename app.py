
import re
import math
import joblib
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Multi-Modal Phishing Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Multi-Modal Phishing Detection")
st.markdown("Detect phishing threats across **URLs, Emails, and SMS messages** using TF-IDF & Random Forest.")

@st.cache_resource
def load_models():
    vec = joblib.load("tfidf_vectorizer.joblib")
    clf = joblib.load("phishing_rf_model.joblib")
    return vec, clf

def calc_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def detect_content_type(val):
    s = str(val).strip()
    if re.match(r"^(https?://|www\.)\S+$", s) or (
        " " not in s and "." in s and len(s) < 300
    ):
        return "url"
    return "text_or_email"

def extract_single_url_features(text, ctype):
    if ctype == "url":
        return np.array([[
            1,
            len(text),
            text.count("."),
            text.count("-"),
            text.count("/"),
            sum(c.isdigit() for c in text),
            1 if text.lower().startswith("https://") else 0,
            1 if text.lower().startswith("http://") else 0,
            1 if "@" in text else 0,
            1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", text) else 0,
            calc_entropy(text)
        ]], dtype=np.float32)
    return np.zeros((1, 11), dtype=np.float32)

try:
    vectorizer, rf_model = load_models()
except Exception as e:
    st.error(f"Model files load nahi ho sakein: {e}")
    st.info("Make sure `tfidf_vectorizer.joblib` aur `phishing_rf_model.joblib` aapke GitHub repository mein uploaded hain.")
    st.stop()

sample_text = st.text_area(
    "Enter a URL, Email body, or SMS message to analyze:",
    height=150,
    placeholder="e.g., http://paypal-security-check-login.com"
)

threshold = st.slider("Detection Sensitivity Threshold", 0.1, 0.9, 0.5, 0.05)

if st.button("Analyze Content", type="primary"):
    if not sample_text.strip():
        st.warning("Please enter a URL, Email, or SMS first.")
    else:
        s = sample_text.strip()
        ctype = detect_content_type(s)
        X_txt = vectorizer.transform([s])
        X_num = csr_matrix(extract_single_url_features(s, ctype))
        X_full = hstack([X_txt, X_num])

        phishing_prob = float(rf_model.predict_proba(X_full)[0][1])
        is_phishing = phishing_prob >= threshold

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Detected Input Type", "URL" if ctype == "url" else "Email / SMS Text")
        col2.metric("Phishing Risk Score", f"{phishing_prob * 100:.2f}%")

        if is_phishing:
            st.error("🚨 **PHISHING DETECTED!** This content shows strong indicators of a phishing attack.")
        else:
            st.success("✅ **LEGITIMATE:** This content appears safe.")
