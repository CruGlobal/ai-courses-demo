from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# --------- Environment Setup ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI()

# --------- Load GodTools Data and Embeddings ----------
embedding_file = "godtools_dataset_with_embeddings.csv"

df = pd.read_csv("godtools_dataset.csv")
df = df[['item_id', 'type', 'title', 'description', 'keywords', 'language_support', 'available_countries', 'available_regions']]
df['text'] = df.apply(lambda row: f"{row['title']}. {row['description']} Keywords: {row['keywords']}", axis=1)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.strip().replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def get_batch_embeddings(texts, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]

if not os.path.exists(embedding_file):
    print("\nCreating embeddings... This may take a few minutes.")
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(df), batch_size):
        batch = df['text'].iloc[i:i+batch_size].tolist()
        embeddings = get_batch_embeddings(batch)
        all_embeddings.extend(embeddings)
    df['embedding'] = all_embeddings
    df.to_csv(embedding_file, index=False)
else:
    df = pd.read_csv(embedding_file, converters={"embedding": eval})

embedding_matrix = np.array(df['embedding'].tolist()).astype('float32')
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# --------- Memory for follow-up context ----------
last_discussed_tool = None
top_tools = None

# --------- API Models ----------
class ChatRequest(BaseModel):
    user_input: str

# --------- Helper: Get referenced tool index ----------
def get_referenced_tool_index(user_input, top_tools):
    titles = [tool['title'] for tool in top_tools]
    joined_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
    prompt = f"""
You previously recommended these 3 GodTools resources:

{joined_titles}

The user now said: "{user_input}"

Which resource are they referring to? Answer ONLY with 1, 2, 3, or "None" if unclear.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    return int(answer) - 1 if answer in ["1", "2", "3"] else None

# --------- Chat Endpoint ----------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    global last_discussed_tool, top_tools

    user_input = request.user_input.strip()

    # First time: run similarity search
    if not top_tools:
        query_vec = np.array([get_embedding(user_input)]).astype('float32')
        D, I = index.search(query_vec, k=3)
        top_tools = df.iloc[I[0]].to_dict(orient='records')

        top_results = []
        for i, tool in enumerate(top_tools, start=1):
            top_results.append(
                f"{i}. {tool['title']} ({tool['type']})\nDescription: {tool['description']}\nLanguages: {tool['language_support']}\n"
            )

        return {
            "reply": "Top 3 matching GodTools resources:\n\n" + "\n".join(top_results) + "\nYou can now ask follow-up questions about these." }

    # Try to resolve reference
    idx = get_referenced_tool_index(user_input, top_tools)

    if idx is not None and 0 <= idx < len(top_tools):
        tool = top_tools[idx]
        last_discussed_tool = tool

        # Inline language check
        if "available in" in user_input.lower():
            user_lang = user_input.lower().split("available in")[-1].strip()
            available_langs = [lang.strip().lower() for lang in tool['language_support'].split(",")]

            if user_lang in available_langs:
                return {"reply": f"Yes, this resource is available in {user_lang.title()}."}
            else:
                return {"reply": f"No, this resource is not available in {user_lang.title()}. Available languages: {tool['language_support']}"}

        # General answer about the selected tool
        system_prompt = (
            "You are a helpful assistant answering user questions about GodTools resources. "
            "For factual questions (languages, availability, region), answer briefly in 1-2 sentences. "
            "For explanation questions, you can give more detail."
        )
        user_prompt = (
            f"Resource details:\n"
            f"Title: {tool['title']}\n"
            f"Type: {tool['type']}\n"
            f"Description: {tool['description']}\n"
            f"Keywords: {tool['keywords']}\n"
            f"Languages: {tool['language_support']}\n"
            f"Regions: {tool['available_regions']}\n\n"
            f"User asked: \"{user_input}\""
        )
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return {"reply": chat.choices[0].message.content}

    # If no reference but last discussed tool exists
    if last_discussed_tool:
        system_prompt = (
            "You are a helpful assistant answering follow-up questions about a GodTools resource. "
            "Be short and specific for factual questions (1-2 sentences max). For explanations, provide more detail."
        )
        user_prompt = (
            f"Here are the resource details:\n"
            f"Title: {last_discussed_tool['title']}\n"
            f"Type: {last_discussed_tool['type']}\n"
            f"Description: {last_discussed_tool['description']}\n"
            f"Keywords: {last_discussed_tool['keywords']}\n"
            f"Languages: {last_discussed_tool['language_support']}\n"
            f"Regions: {last_discussed_tool['available_regions']}\n\n"
            f"The user asked: \"{user_input}\""
        )
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return {"reply": chat.choices[0].message.content}

    # Fallback
    context = "\n".join([f"{i+1}. {c['title']}" for i, c in enumerate(top_tools)])
    fallback_prompt = f"""
Here are the top 3 recommended GodTools resources:

{context}

User asked: "{user_input}"

Respond helpfully, even if the user didn’t reference a resource directly.
"""
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": fallback_prompt}]
    )
    return {"reply": chat.choices[0].message.content}
