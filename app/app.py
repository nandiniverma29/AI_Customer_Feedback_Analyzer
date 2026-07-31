from groq_helper import analyze_review
from pathlib import Path
import streamlit as st
import joblib
import re

model = joblib.load("model/sentiment_model.joblib")
tfidf = joblib.load("model/tfidf_vectorizer.joblib")

def load_css():
    css_path = Path(__file__).parent / "style.css"

    css = css_path.read_text()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


load_css()

def clean_text(text):
    text = text.lower()

    text = re.sub(
        '[^a-zA-Z]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text


st.markdown("""
<div class="badge">
    <span>✨ AI Powered • Machine Learning • Sentiment Analysis</span>
</div>

<h1 class="main-title">
AI Customer Feedback Analyzer
</h1>

""", unsafe_allow_html=True)


st.markdown("""
<div class="card">
    <h3>📝 Enter Customer Review</h3>
""", unsafe_allow_html=True)

review = st.text_area(
    "",
    placeholder="Type or paste the customer review here..."
)

st.markdown("</div>", unsafe_allow_html=True)


if st.button("Analyze"):

    cleaned_review = clean_text(review)

    vector = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    negative_probability = probability[0] * 100
    positive_probability = probability[1] * 100

    confidence = max(probability) * 100


    if prediction == 1:
        sentiment = "Positive 😊"

    else:
        sentiment = "Negative 😞"


    st.markdown(
        '<h2 class="results-heading">📊 Analysis Results</h2>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-icon">😊</div>
        <div class="metric-value">{sentiment}</div>
        <div class="metric-label">Sentiment</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">{confidence:.1f}%</div>
        <div class="metric-label">Model Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-icon">📈</div>
        <div class="metric-value">{positive_probability:.1f}%</div>
        <div class="metric-label">Positive Probability</div>
        </div>
        """, unsafe_allow_html=True)
    st.divider()

    with st.spinner("🤖 Generating AI insights..."):
        ai_response = analyze_review(review, sentiment)

    formatted_response = ai_response.replace("\n", "<br>")

    st.markdown(f"""
    <div class="ai-box">
    <div class="ai-header">
        🤖 AI Insights
    </div>

    <div class="ai-response-text">
        {formatted_response}
    </div>

    </div>
    """, unsafe_allow_html=True)