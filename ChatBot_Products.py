#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Pet Product Assistant",
    page_icon="🧴",
    layout="centered"
)

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----------------------------
# Load catalog
# ----------------------------
@st.cache_data
def load_catalog():
    df = pd.read_csv("C:/Users/furca/Downloads/pet_store_shampoo_conditioner_products_full.csv").fillna("")
    return df

df = load_catalog()

# ----------------------------
# Convert each row into text for retrieval
# ----------------------------
def row_to_text(row):
    return f"""
Product: {row['product']}
Brand: {row['brand']}
Type: {row['type']}
Pet Type: {row['pet_type']}
Concerns: {row['concerns']}
USP: {row['usp']}
Key Ingredients: {row['key_ingredients']}
Coat Type: {row['coat_type']}
Suitable For: {row['suitable_for']}
Avoid If: {row['avoid_if']}
Usage Notes: {row['usage_notes']}
Size: {row['size_ml']} ml
""".strip()

if "search_text" not in df.columns:
    df["search_text"] = df.apply(row_to_text, axis=1)

# ----------------------------
# Embeddings
# ----------------------------
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

@st.cache_data(show_spinner=False)
def build_embeddings(search_texts):
    return [get_embedding(text) for text in search_texts]

if "embedding" not in df.columns:
    with st.spinner("Preparing product knowledge base..."):
        df["embedding"] = build_embeddings(df["search_text"].tolist())

# ----------------------------
# Retrieval
# ----------------------------
def retrieve_products(query, df, top_k=6):
    query_embedding = get_embedding(query)
    product_embeddings = np.array(df["embedding"].tolist())
    similarities = cosine_similarity([query_embedding], product_embeddings)[0]

    result = df.copy()
    result["score"] = similarities
    result = result.sort_values("score", ascending=False).head(top_k)
    return result

def build_context(retrieved_df):
    blocks = []

    for _, row in retrieved_df.iterrows():
        blocks.append(
            f"""Product: {row['product']}
Brand: {row['brand']}
Type: {row['type']}
Pet Type: {row['pet_type']}
Concerns: {row['concerns']}
USP: {row['usp']}
Key Ingredients: {row['key_ingredients']}
Coat Type: {row['coat_type']}
Suitable For: {row['suitable_for']}
Avoid If: {row['avoid_if']}
Usage Notes: {row['usage_notes']}
Size: {row['size_ml']} ml"""
        )

    return "\n\n".join(blocks)

# ----------------------------
# Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are a friendly, professional pet retail assistant for a pet store.

You help customers choose shampoos, conditioners, balms, foams, and cleansing products from the store catalog.

Rules:
1. Only recommend products that appear in the provided store catalog context.
2. Do not invent products, ingredients, medical effects, or prices.
3. Explain why each recommendation fits the customer's pet, concern, coat type, or grooming need.
4. Keep the tone calm, helpful, and easy to understand.
5. Suggest 1 to 4 products when suitable.
6. If the customer mentions severe skin issues, wounds, bleeding, swelling, infection, strong redness, constant scratching, or pain, do not diagnose. Recommend seeing a vet first.
7. Shampoos and conditioners can support grooming comfort, coat condition, odour control, and cleansing, but should not be presented as medical treatment.
8. If a product has an avoid_if note, respect it clearly.
9. If the catalog context does not contain a suitable product, say so honestly.
10. Do not pressure the customer to buy.

Output style:
1. Briefly acknowledge the customer's concern.
2. Give the best recommendation first.
3. For each product, explain:
   Product name
   Why it fits
   Key ingredients or USP
   Usage or safety note
4. End with a simple follow-up question if useful.
"""

def generate_answer(user_question, df, chat_history):
    retrieved = retrieve_products(user_question, df, top_k=6)
    context = build_context(retrieved)

    history_text = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in chat_history[-6:]]
    )

    prompt = f"""
Chat history:
{history_text}

Customer question:
{user_question}

Relevant store products:
{context}
"""

    response = client.responses.create(
        model="gpt-5.2",
        instructions=SYSTEM_PROMPT,
        input=prompt
    )

    return response.output_text, retrieved

# ----------------------------
# UI styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        padding-top: 1.2rem;
    }
    .block-container {
        max-width: 950px;
        padding-top: 1.5rem;
    }
    .product-card {
        background: #f7f7f8;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid #e5e7eb;
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧴 Pet Shampoo & Conditioner Assistant")
st.caption("Grounded on your store's product catalog")

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything about shampoos, conditioners, skin comfort, odour control, shedding, coat softness, sensitive skin, or which product may suit your pet."
        }
    ]

# ----------------------------
# Render chat
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Chat input
# ----------------------------
user_prompt = st.chat_input("Ask about shampoos, conditioners, coat or skin concerns...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, retrieved = generate_answer(
                user_prompt,
                df,
                st.session_state.messages
            )

            st.markdown(answer)

            with st.expander("Top matched products"):
                for _, row in retrieved.iterrows():
                    st.markdown(
                        f"""
                        <div class="product-card">
                            <b>{row['product']}</b><br>
                            <span class="small-muted">
                                Brand: {row['brand']} | Type: {row['type']} | Pet Type: {row['pet_type']} | Size: {row['size_ml']} ml
                            </span><br><br>
                            <b>Concerns:</b> {row['concerns']}<br>
                            <b>USP:</b> {row['usp']}<br>
                            <b>Key Ingredients:</b> {row['key_ingredients']}<br>
                            <b>Coat Type:</b> {row['coat_type']}<br>
                            <b>Suitable For:</b> {row['suitable_for']}<br>
                            <b>Avoid If:</b> {row['avoid_if']}<br>
                            <b>Usage Notes:</b> {row['usage_notes']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})


# In[ ]:




