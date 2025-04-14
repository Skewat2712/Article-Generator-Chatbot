# 📰 Article Generator Chatbot (Local LLMs + Auto-Evaluation)

A **Streamlit-based chatbot** that uses local open-source LLMs (Mistral, Gemma, LLaMA2 via Ollama) to generate detailed articles from user-provided topics. It automatically evaluates the output for **fluency, relevance, accuracy**, and **safety**, then recommends the best-performing model.

---

## 🚀 Features

- 🔁 **Parallel generation** using 3 local LLMs
- 📊 **Auto-evaluation** with metrics scaled 1–5
- 🧠 Models compared: `mistral`, `gemma`, `llama2`
- 🏆 **Best model recommendation**
- 🧾 **Full chat history** with expandable article views
- 💡 Built with **Streamlit**, **Ollama**, and **sklearn**

---

## 🛠️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/article-generator-chatbot.git
cd article-generator-chatbot
```

### 2. Install Python Dependencies
Make sure you're using Python 3.9+
```bash
pip install -r requirements.txt
```

### 3. Install & Run Ollama (Local LLM Backend)
If you haven’t already, install [Ollama](https://ollama.com/):

```bash
# For macOS
brew install ollama

# For Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama in the background:
```bash
ollama serve
```

### 4. Pull the Required Models
Use the following commands to pull the models locally:

```bash
ollama pull mistral
ollama pull gemma
ollama pull llama2
```

> 🧠 These models are run **locally** using your machine's resources. Ensure you have enough RAM (ideally 8GB+ per model).

---

## ▶️ Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

---

## ✨ Example Screenshot

> _Add your own screenshot here if needed:_

```
📸 ![screenshot](screenshot.png)
```

---

## 📂 File Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📋 Evaluation Metrics (Auto-Scored 1–5)

| Metric    | Description                                       |
|-----------|---------------------------------------------------|
| Fluency   | Measures natural flow using avg sentence length   |
| Relevance | Cosine similarity between topic and full article  |
| Accuracy  | Cosine similarity with first 3 sentences          |
| Safety    | Checks for harmful or unsafe keywords             |

---

## 📜 License

MIT License. Feel free to fork, improve, and use!

---

## 🙌 Credits

- [Ollama](https://ollama.com/) for local model support  
- Streamlit for the clean and easy UI  
- Scikit-learn for vectorization and similarity scoring