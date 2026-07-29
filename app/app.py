from gemini_helper import analyze_review
import streamlit as st
import joblib
import re

model = joblib.load("model/sentiment_model.joblib")
tfidf = joblib.load("model/tfidf_vectorizer.joblib")


def clean_text(text):
    text = text.lower()

    text = re.sub(
        '[^a-zA-Z]',
        ' ',
        text
    )

    text = re.sub(
        '\s+',
        ' ',
        text
    )

    return text


st.title("AI Customer Feedback Analyzer")

st.write(
    "Analyze customer reviews using Machine Learning"
)


review = st.text_area(
    "Enter Customer Review"
)


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


    if confidence < 60:
        sentiment = "Mixed / Uncertain Sentiment ⚖️"

    elif prediction == 1:
        sentiment = "Positive Feedback 😊"

    else:
        sentiment = "Negative Feedback 😞"


    # Display result
    st.subheader("Analysis Result")

    st.write(
        f"**Sentiment:** {sentiment}"
    )

    st.write(
        f"**Positive Probability:** {positive_probability:.2f}%"
    )

    st.write(
        f"**Negative Probability:** {negative_probability:.2f}%"
    )

    st.write(
        f"**Model Confidence:** {confidence:.2f}%"
    )
    st.divider()

    st.subheader("🤖 AI Analysis")

    with st.spinner("Generating AI insights..."):
        ai_response = analyze_review(
            review,
            sentiment
            )

    st.write(ai_response)