import streamlit as st
import pandas as pd
import qdrant_client
from dotenv import load_dotenv
import os

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from qdrant_client.http import models as rest

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import (
    AttributeInfo,
)

load_dotenv()


# --------------- USER EDITABLE VARIABLES ---------------

csv_file_path = "enhanced_property_data_with_rich_descriptions.csv"

placeholder_chat = (
    "e.g., 'Show me 5 bedroom homes in Chicago with a pool and EV charging'"
)

DOCUMENT_PROMPT_TEMPLATE = """
---
Property Listing:
Address: {address}, {city}
Price: ${price}
Bedrooms: {bedrooms}
Bathrooms: {bathrooms}
Size (sqft): {size_sqft}
Year Built: {year_built}
School Rating (out of 10): {school_rating}
Has Pool: {has_pool}
Garage Spaces: {garage_spaces}
Description: {page_content}
"""

QA_PROMPT_TEMPLATE = """You are a world-class AI real estate agent: charismatic, empathetic, and incredibly knowledgeable. Your primary goal is to build rapport with the client and guide them to their perfect home from the available listings.

**Your Core Task:**
Use the **Available Property Listings** provided in the context to answer the client's questions. These listings are your ONLY source of truth. Do not make up properties or features that are not explicitly listed in the context.

**Personality and Selling Style:**
- **Opener:** Start with a warm, friendly greeting. Ask open-ended questions like "What's bringing you to the market today?" or "Tell me a bit about the dream home you're imagining."
- **Storytelling:** Don't just list facts. Weave a narrative. Instead of "It has a great backyard," say "Imagine hosting summer barbecues in that spacious backyard..."
- **Lifestyle Focus:** Sell the lifestyle, not just the house. Connect features to benefits. "The short commute time means you'll have more time for family in the evenings."
- **Create Subtle Urgency:** For a great match, say "A property like this in this market tends to get a lot of attention. Would you be interested in a viewing?"

**Handling Edge Cases - This is CRITICAL:**
1.  **If No Perfect Match is Found:**
    - Do NOT just say "I couldn't find anything."
    - Act as a consultant. State clearly which criteria were met and which weren't.
    - If there are close matches, present them and explain the trade-offs. Example: "While I don't have a listing with both a pool and an EV charger, I found this amazing eco-friendly home with an EV charger. It has a huge backyard that would be perfect for adding a pool. Would that be of interest?"
2.  **If the User's Request is Ambiguous (e.g., "I want something nice"):**
    - Do NOT guess. Ask clarifying questions to understand their needs better.
    - Example: "I can certainly help find a 'nice' home! To narrow it down, are you picturing a modern design, a cozy traditional feel, or perhaps something with a great view?"
3.  **If the User's Request is Impossible (e.g., "a mansion in Denver for $50k"):**
    - Be a gentle realist. Acknowledge their goal but politely guide them based on the provided listings.
    - Example: "A mansion for $50k would be the deal of the century! Based on the current listings, properties in Denver start closer to the $300k mark. Would you like me to show you some beautiful homes in that range?"

**Final Instructions:**
- Base every property-related answer on the **Available Property Listings** provided below.
- If the listings are empty, it means no properties matched the user's specific filters. You must state this clearly and politely ask them if they'd like to broaden their search.

**Available Property Listings:**
{context}

**Chat History:**
{chat_history}

**Client's Latest Message:**
{question}

**Your Professional Response:**
"""

metadata_field_info = [
    AttributeInfo(
        name="address",
        description="The full street address of the property.",
        type="string",
    ),
    AttributeInfo(
        name="city",
        description="The city where the property is located. Should be one of 'Denver', 'Miami', 'Chicago', 'Seattle', 'Austin'.",
        type="string",
    ),
    AttributeInfo(
        name="price",
        description="The price of the property in USD.",
        type="integer",
    ),
    AttributeInfo(
        name="bedrooms",
        description="The number of bedrooms in the property.",
        type="integer",
    ),
    AttributeInfo(
        name="bathrooms",
        description="The number of bathrooms in the property.",
        type="integer",
    ),
    AttributeInfo(
        name="size_sqft",
        description="The total size of the property's interior in square feet.",
        type="integer",
    ),
    AttributeInfo(
        name="lot_size_sqft",
        description="The total size of the land or lot in square feet.",
        type="integer",
    ),
    AttributeInfo(
        name="has_pool",
        description="Whether the property has a pool. Can be 1 for yes or 0 for no.",
        type="integer",
    ),
    AttributeInfo(
        name="year_built",
        description="The year the property was built.",
        type="integer",
    ),
    AttributeInfo(
        name="garage_spaces",
        description="The number of garage spaces available at the property.",
        type="integer",
    ),
    AttributeInfo(
        name="school_rating",
        description="The rating of the nearby school, from 1 (worst) to 10 (best).",
        type="integer",
    ),
    AttributeInfo(
        name="commute_time_min",
        description="The commute time to a downtown area, in minutes.",
        type="integer",
    ),
]

document_content_description = "A textual description of a real estate property, including its features, style, and amenities like 'EV charging station' or 'solar panels'."

# ------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "real_estate_properties"


@st.cache_resource
def get_qdrant_client():
    os.makedirs("local_qdrant", exist_ok=True)
    return qdrant_client.QdrantClient(path="local_qdrant")


@st.cache_resource
def get_llm_and_embeddings():
    if not all(
        [
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        ]
    ):
        st.error(
            "Azure OpenAI environment variables are not fully set. Please check your .env file."
        )
        st.stop()

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_version="2023-05-15",
    )
    llm = AzureChatOpenAI(
        api_version="2025-01-01-preview",
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        temperature=0.1,
        max_tokens=None,
    )
    return llm, embeddings


def initialize_database(
    client: qdrant_client.QdrantClient, embeddings_model: AzureOpenAIEmbeddings
):
    collection_exists = client.collection_exists(collection_name=COLLECTION_NAME)

    if not collection_exists:
        st.sidebar.warning(
            f"Collection '{COLLECTION_NAME}' not found. Creating and populating..."
        )
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
        )

        df = pd.read_csv(csv_file_path)
        df["has_pool"] = df["has_pool"].astype(int)
        texts = df["description"].tolist()
        metadata = df.drop(columns=["description"]).to_dict(orient="records")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings_model,
        )
        vector_store.add_texts(
            texts=texts,
            metadatas=metadata,
        )
        st.sidebar.success(f"Successfully created and populated '{COLLECTION_NAME}'.")
    else:
        st.sidebar.info(f"Connected to existing collection '{COLLECTION_NAME}'.")


DOCUMENT_PROMPT = PromptTemplate.from_template(DOCUMENT_PROMPT_TEMPLATE)

QA_PROMPT = PromptTemplate.from_template(QA_PROMPT_TEMPLATE)

st.set_page_config(page_title="AI Real Estate Agent", layout="wide")
st.title("🏡 AI Real Estate Agent")

qdrant_cli = get_qdrant_client()
llm_model, embeddings = get_llm_and_embeddings()

with st.sidebar:
    st.header("Database Status")
    initialize_database(qdrant_cli, embeddings)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):
            with st.expander(
                f"See the {len(message['sources'])} properties I considered"
            ):
                for doc in message["sources"]:
                    st.markdown(
                        f"**Address:** {doc.metadata.get('address', 'N/A')}, {doc.metadata.get('city', 'N/A')}"
                    )
                    st.markdown(f"**Price:** ${doc.metadata.get('price', 0):,}")
                    st.markdown(
                        f"**Details:** {doc.metadata.get('bedrooms', 'N/A')} bed, {doc.metadata.get('bathrooms', 'N/A')} bath, {doc.metadata.get('size_sqft', 'N/A')} sqft"
                    )
                    st.markdown(f"**Description:** *{doc.page_content}*")
                    st.markdown("---")

vector_store = QdrantVectorStore(
    client=qdrant_cli,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm_model,
    vectorstore=vector_store,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_model,
    retriever=retriever,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT, "document_prompt": DOCUMENT_PROMPT},
    return_source_documents=True,
)

if prompt := st.chat_input(placeholder_chat):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing your request and filtering properties..."):
        response = qa_chain.invoke({"question": prompt})

    assistant_message = {
        "role": "assistant",
        "content": response["answer"],
        "sources": response["source_documents"],
    }
    st.session_state.messages.append(assistant_message)
    st.rerun()
