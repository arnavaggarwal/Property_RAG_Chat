# 🏡 AI Real Estate Agent

An end-to-end AI-powered property recommendation system using Streamlit for the frontend, FastAPI for the backend, and LangChain + Qdrant for conversational retrieval. Checkout the hosted version on here : https://arnavaggarwal-property-rag-chat-new-app-rnxkro.streamlit.app/

---

## Prerequisites

* **Python 3.12.11** or higher installed.
* **Git** (to clone the repo).
* Ensure the property data file, **enhanced_property_data_with_rich_descriptions.csv**, is present in the root of your project directory.
* An Azure account with access to Azure OpenAI Service. You will need:

    **Azure API Key**

    **Azure Endpoint URL**

    **A Chat Model Deployment Name** (e.g., for gpt-4)

    **An Embedding Model Deployment Name** (e.g., for text-embedding-ada-002)

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/arnavaggarwal/Property_RAG_Chat.git
   cd Property_RAG_Chat
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3.12 -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

This app supports two ways to supply secrets & settings:

1. **Local dev with `config.toml`** (ignored by Git via `.gitignore`):

   ```toml
   [azure_openai]
   api_key = "<YOUR_KEY>"
   endpoint = "https://<your-endpoint>.openai.azure.com/"
   chat_deployment_name = "gpt-4.1"
   embedding_deployment_name = "text-embedding-ada-002"

   [qdrant]
   host = "localhost"
   port = 6333
   ```

   Place this file at the project root: `config.toml`.

2. **Streamlit Cloud / Production**
   Create a file at `.streamlit/secrets.toml` with the same structure under `[azure_openai]` and `[qdrant]`. The app will auto-fallback to `st.secrets` if `config.toml` is absent.

---

## Running the App

Launch the Streamlit interface:

```bash
streamlit run new_app.py
```

Then open `http://localhost:8501` in your browser.


**Initialization:**

    The application will connect to the Qdrant vector database.

    If it's the first time running, it will create and populate the database from your CSV file. This may take a minute.

**Chat with the AI:**

    Once the "Database Status" shows it is connected, you can start chatting.

    Use the chat input box at the bottom of the screen to type your requests.

**View Sources:**

    For answers that involve property listings, an expandable section titled "See the X properties I considered" will appear below the AI's message. Click on it to see the details.

---

## File Structure

```
├── .gitignore
├── config.toml            # Local-only secrets (ignored)
├── requirements.txt       # Python dependencies
├── new_app.py             # Streamlit frontend + orchestration
├── backend/
│   └── train_model.py     # ML training script
├── utils/
│   └── rate_limit.py      # TokenBucket rate-limiter
├── enhanced_property_data_with_rich_descriptions.csv
└── README.md              # This file
```

---

