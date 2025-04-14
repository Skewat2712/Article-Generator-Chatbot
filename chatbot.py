''' 
Article Generator Chatbot using three different Local LLM Models.
    1. Mistral
    2.Gemma
    3.Llama2
This project is made on local system using Ollama.
'''
import streamlit as st
import ollama
import time
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt')

# Prompt to specify and describe the model to generate the result.
def build_prompt(topic):
    return f"Write a detailed, informative, and engaging article about: {topic}"

# This will generate the article.
def generate_article(model_name, prompt):
    start = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    return {
        "model": model_name,
        "article": response['message']['content'],
        "latency": latency
    }
# Word count will return the length of the text generated.
def word_count(text):
    return len(text.split())

# How fluent (readable and natural) the article is. And returns the average sentence length 
def estimate_fluency(text):
    sentences = sent_tokenize(text)
    words = text.split()
    if len(sentences) == 0:
        return 0
    avg_sentence_length = len(words) / len(sentences)
    return min(round(avg_sentence_length / 5.0, 2), 5.0)

# Measure how related the article is to the given topic, Calculates TF-IDF vectorizeration - cosine similarity between the topic and the full article. 
def estimate_relevance(topic, text):
    vectorizer = TfidfVectorizer().fit_transform([topic, text])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return round(float(similarity[0][0] * 5), 2)

# Re-uses the same TF-IDF + cosine similarity approach from relevance.
def estimate_accuracy(topic, text):
    first_3_sentences = " ".join(sent_tokenize(text)[:3])
    return estimate_relevance(topic, first_3_sentences)

# Check for harmful, unsafe, or offensive content.
def estimate_safety(text):
    unsafe_keywords = ["kill", "hate", "violence", "racist", "bomb", "terror"]
    text = text.lower()
    for word in unsafe_keywords:
        if word in text:
            return 1.0
    return 5.0

# --- Streamlit App ---
st.set_page_config(page_title="Multi-LLM Article Generator Chatbot", layout="wide")
st.title(" Article Generator Chatbot \n with Mistral, Gemma and Ollama2 LLM Models.")

if "history" not in st.session_state:
    st.session_state.history = []

topic = st.text_input(" Enter a topic for the article:")

# This step includes the parallel generation of the articles from the LLM Models locally. Using thread Pool Executor.
if st.button(" Generate with Mistral, Gemma and Ollama2 Models"):
    prompt = build_prompt(topic)
    models = ["mistral", "gemma", "llama2"]
    with st.spinner("Generating articles from Mistral, Gemma and Ollama2 LLM models..."):
        # Parallel generation
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda m: generate_article(m, prompt), models))

        best_model = None
        best_score = 0
        full_result_set = []

        for res in results:
            article = res['article']
            metrics = {
                "Word Count": word_count(article),
                "Fluency": estimate_fluency(article),
                "Relevance": estimate_relevance(topic, article),
                "Accuracy": estimate_accuracy(topic, article),
                "Safety": estimate_safety(article)
            }
            avg_score = round(
                (metrics["Fluency"] + metrics["Relevance"] + metrics["Accuracy"] + metrics["Safety"]) / 4, 2
            )
            res.update({
                "topic": topic,
                "metrics": metrics,
                "avg_score": avg_score
            })
            full_result_set.append(res)

        st.session_state.history.append({
            "topic": topic,
            "results": full_result_set
        })

# --- Display Results in Columns ---
if st.session_state.history:
    st.subheader(" Latest Comparison")

    latest = st.session_state.history[-1]
    topic = latest['topic']
    results = latest['results']

    cols = st.columns(3)
    for idx, col in enumerate(cols):
        if idx < len(results):
            chat = results[idx]
            with col:
                st.markdown(f"###  {chat['model'].capitalize()}")
                st.markdown(f"** Time:** `{chat['latency']:.2f} sec`")
                st.markdown(f"** Word Count:** `{chat['metrics']['Word Count']}`")
                st.markdown("#### Article")
                st.write(chat['article'])
                st.markdown("####  Evaluation on scale of 1 - 5 ")
                st.markdown(f"- **Fluency:** `{chat['metrics']['Fluency']}`")
                st.markdown(f"- **Relevance:** `{chat['metrics']['Relevance']}`")
                st.markdown(f"- **Accuracy:** `{chat['metrics']['Accuracy']}`")
                st.markdown(f"- **Safety:** `{chat['metrics']['Safety']}`")
                st.markdown(f"** Average Score:** `{chat['avg_score']}`")

    best_model = max(results, key=lambda x: x['avg_score'])
    st.success(f"🏆 **Best Performing Model:** `{best_model['model']}` with an average score of **{best_model['avg_score']} / 5**")

# Chat History 
if len(st.session_state.history) > 1:
    st.subheader(" Full Chat History")

    for i, past in enumerate(reversed(st.session_state.history[:-1])):  
        st.markdown(f"###  Topic: **{past['topic']}**")
        inner_cols = st.columns(3)
        for idx, col in enumerate(inner_cols):
            if idx < len(past["results"]):
                chat = past["results"][idx]
                with col:
                    st.markdown(f"** {chat['model'].capitalize()}**")
                    st.markdown(f"** {chat['latency']:.2f} sec**")
                    st.markdown(f"** Word Count:** `{chat['metrics']['Word Count']}`")
                    st.markdown(f"Average Score:** `{chat['avg_score']}`")
                    with st.expander(" Show Article"):
                        st.write(chat['article'])
                    st.markdown("---")
