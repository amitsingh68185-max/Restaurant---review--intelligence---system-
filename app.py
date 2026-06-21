import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.markdown(
    "<h1 style='text-align:center;color:#00FFAA;'>🍴 AI Restaurant Review Intelligent System</h1>",
    unsafe_allow_html=True
)

review = st.text_area("Enter Restaurant Review")

import pandas as pd

data = pd.read_excel("Project_RRIS_Gomtinagar.xlsx",sheet_name="Review_Data")

texts = data["Review_Text"].fillna("").astype(str)

labels = data["Sentiment"].fillna("").astype(str)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

model = LogisticRegression()

model.fit(X, labels)

if st.button("Predict Sentiment"):

    review_vec = vectorizer.transform([review])
    with st.spinner("Analyzing Review..."):


        prediction = model.predict(review_vec)

    if prediction[0] == "Positive":
       st.success("😊 Positive Review")
       st.progress(85)

    else:
        st.error("😡 Negative Review")
        st.progress(20)