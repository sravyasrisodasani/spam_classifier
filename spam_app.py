import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------- PAGE STYLE ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg,#667eea,#764ba2);
}

/* Center container */
.block-container {
    max-width: 700px;
    margin: auto;
    padding-top: 40px;
}

/* Title */
.title {
    text-align: center;
    color: #4c1d95;
    font-size: 36px;
}

/* Textarea */
textarea {
    caret-color: black;
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 10px !important;
    border: 2px solid #7c3aed !important;
    padding: 10px !important;
}

/* Button */
div.stButton > button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #6d28d9;
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)


# ---------- TEXT PROCESSING ----------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ---------- LOAD MODEL ----------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


# ---------- UI ----------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>📩 Email / SMS Spam Classifier</h1>", unsafe_allow_html=True)

st.write("### Enter your message")

input_sms = st.text_area("Message", height=150)


if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        # preprocessing
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # prediction
        result = model.predict(vector_input)[0]

        st.subheader("Prediction Result")

        if result == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

        # confidence score
        prob = model.predict_proba(vector_input)[0][result]
        st.write(f"Confidence Score: {prob*100:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Built with ❤️ using Python, NLP and Streamlit")