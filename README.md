#  💬 Loan Approval Q&A Chatbot with RAG

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg) ![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green.svg) ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)

This project is an intelligent Q&A chatbot that allows users to ask natural language questions about a loan approval dataset. Instead of manually filtering and analyzing the data, you can simply ask the bot questions like "How many self-employed applicants were approved?" or "What is the average loan amount for graduates in urban areas?".

The chatbot is built using a **Retrieval-Augmented Generation (RAG)** pipeline, which ensures that the answers are grounded in the specific data from the provided CSV file.

### Live Demo (GIF)


📽️ **[Video Demo](https://jklujaipur-my.sharepoint.com/:v:/g/personal/dishaarora_jklu_edu_in/EZnXc-yeSa5Pjo4kCQZFaQ4BK2b6sOqizrQmqYTJQDs82w?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=VtdbgN)**



## 📌 Features

-   **Interactive Chat Interface**: A user-friendly web interface built with Streamlit.
-   **Natural Language Understanding**: Ask complex questions in plain English.
-   **Data-Grounded Answers**: The AI model uses information retrieved directly from the loan dataset to formulate its answers, preventing hallucination.
-   **Conversation History**: The chatbot remembers the context of the conversation, allowing for follow-up questions.
-   **Powered by Open Source**: Utilizes powerful models and libraries from Hugging Face and LangChain.

## 🚀 How It Works: The RAG Pipeline

The core of this application is the Retrieval-Augmented Generation (RAG) pipeline. This approach enhances the power of a Large Language Model (LLM) by providing it with relevant, context-specific information from an external knowledge base (our CSV file).

Here’s a simplified breakdown of the process:

1.  **Indexing**: The `loan_data.csv` is loaded, and each row is converted into a text "document". These documents are then transformed into numerical vectors (embeddings) and stored in a searchable FAISS vector store.
2.  **Retrieval**: When you ask a question, your question is also converted into an embedding. The system then searches the vector store to find the most similar documents (i.e., the most relevant rows from the CSV).
3.  **Generation**: The original question and the retrieved documents are bundled together into a detailed prompt. This prompt is then sent to a Large Language Model (e.g., Google's Flan-T5), which generates a coherent, human-readable answer based on the provided context.

```
User Question --> Embed --> Search Vector Store --> Retrieve Data --> Stuff into Prompt --> LLM --> Generate Answer
```

### 🧰Technology Stack

-   **Backend & Logic**: Python
-   **Web Framework**: Streamlit
-   **LLM Orchestration**: LangChain
-   **LLM & Embeddings**: Hugging Face (`google/flan-t5-xxl`, `all-MiniLM-L6-v2`)
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Data Manipulation**: Pandas

## 🛠️ Setup and Installation

Follow these steps to run the application locally.

### 1. ✅Prerequisites

-   Python 3.9 or higher
-   `pip` package manager

### 2. 📥Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. 🧪Create a Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. 📦Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. 🔐Get a Hugging Face API Token

The application uses a model from the Hugging Face Hub. You will need a free API token to access it.

1.  Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  Create a new "Access Token" with "read" permissions.
3.  Set this token as an environment variable.

**On macOS/Linux:**
```bash
export HUGGINGFACEHUB_API_TOKEN='your_api_token_here'```

**On Windows:**
```bash
set HUGGINGFACEHUB_API_TOKEN=your_api_token_here
```
*(Note: This sets the variable for the current terminal session only.)*

### 6. 🧾Download the Dataset

1.  Download the training data from this [Kaggle Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv).
2.  Rename the downloaded file from `Training Dataset.csv` to **`loan_data.csv`**.
3.  Place `loan_data.csv` in the root directory of the project.

## ▶️ Running the Application

Once you have completed the setup, run the following command in your terminal:

```bash
streamlit run app.py
```

Your web browser will automatically open to the application's URL.

---

## 📜License

This project is licensed under the MIT License.
