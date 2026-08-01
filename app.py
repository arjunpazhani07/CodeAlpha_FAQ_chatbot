import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🤖 AI FAQ Chatbot")
st.write("Welcome! Ask any question related to the available FAQs.")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Chatbot Menu")

st.sidebar.info(
"""
This chatbot uses:

✅ NLP (NLTK)

✅ TF-IDF Vectorizer

✅ Cosine Similarity

to find the best answer.
"""
)

st.sidebar.markdown("---")

st.sidebar.success("CodeAlpha AI Internship")

# ---------------- SAMPLE QUESTIONS ----------------

st.info("💡 Example Questions")

st.markdown("""
- What is Python?
- What is AI?
- What is Machine Learning?
- What is SQL?
- What is GitHub?
- What is Streamlit?
- What is TensorFlow?
- What is OpenCV?
""")

st.markdown("---")

# ---------------- LOAD FAQ ----------------

faq = pd.read_csv("faq.csv")

# ---------------- PREPROCESS ----------------

stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = str(text).lower()

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(words)

faq["processed"] = faq["Question"].apply(preprocess)
# ---------------- TF-IDF MODEL ----------------

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(faq["processed"])

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- USER INPUT ----------------

user_question = st.text_input(
    "💬 Enter your question",
    placeholder="Example: What is Python?"
)

col1, col2 = st.columns(2)

with col1:
    ask = st.button("🔍 Get Answer", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear Chat", use_container_width=True)

if clear:
    st.session_state.history = []
    st.rerun()

# ---------------- ANSWER LOGIC ----------------

if ask:

    if user_question.strip() == "":
        st.warning("⚠️ Please enter a question.")

    else:

        processed = preprocess(user_question)

        user_vector = vectorizer.transform([processed])

        similarity = cosine_similarity(user_vector, vectors)

        best_index = similarity.argmax()

        score = similarity[0][best_index]

        if score >= 0.10:

            answer = faq.iloc[best_index]["Answer"]

            st.session_state.history.append(
                ("You", user_question)
            )

            st.session_state.history.append(
                ("Bot", answer)
            )

            st.success(answer)

            st.progress(score)

            st.caption(
                f"🎯 Match Confidence : {score*100:.1f}%"
            )

        else:

            st.warning(
                "❌ Sorry! I couldn't find a matching answer. Please try another question."
            )
            # ---------------- CHAT HISTORY ----------------

if st.session_state.history:

    st.markdown("---")
    st.subheader("💬 Chat History")

    for role, message in st.session_state.history:

        if role == "You":
            st.markdown(f"**🧑 You:** {message}")

        else:
            st.markdown(f"**🤖 Bot:** {message}")

# ---------------- FAQ LIST ----------------

st.markdown("---")

with st.expander("📚 View Available FAQ Questions"):

    for question in faq["Question"]:

        st.write("•", question)

# ---------------- PROJECT INFO ----------------

st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write("""
This project was developed as part of the **CodeAlpha AI Internship**.

### Technologies Used
- Python
- Streamlit
- Pandas
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

### Features
- NLP-based text preprocessing
- Intelligent FAQ matching
- Match confidence score
- Chat history
- Clear chat option
- Professional user interface
""")

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "🤖 AI FAQ Chatbot | Developed by Arjun P | CodeAlpha AI Internship 2026"
)