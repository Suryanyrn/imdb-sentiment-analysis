import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Download required NLTK data for the server
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# 2. Load the Model and Vectorizer
@st.cache_resource # This makes the app load faster
def load_assets():
    with open('Best_Model_of_IMDB_linear_regression.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

model, vectorizer = load_assets()

# 3. Setup Preprocessing Functions
stopwords_list = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tag_re = re.compile(r"<[^>]+>")

def preprocess_text(text):
    text = text.lower()
    text = tag_re.sub('', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r"\b[a-zA-Z]\b", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords_list]
    return ' '.join(words)

# 4. Build the UI
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("Type a movie review below, and my machine learning model will predict if it is Positive or Negative!")

user_input = st.text_area("Enter your review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        # Process and Predict
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        
        # Display Results
        if prediction == 1:
            st.success("🟢 Positive Review Detected!")
            st.balloons()
        else:
            st.error("🔴 Negative Review Detected.")
    else:
        st.warning("Please enter a review first.")