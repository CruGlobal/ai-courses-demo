from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

#Loading .env Variable
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize FastAPI App and CORS 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading  Dataset 
df = pd.read_csv("godtools_dataset_with_embeddings.csv", converters={"embedding": eval})
embedding_matrix = np.array(df['embedding'].tolist()).astype('float32')
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Chat Memory 
conversation_history = []
last_discussed_tool = None
top_tools = None

# API Request (schema)
class ChatRequest(BaseModel):
    user_input: str

# Listing all tools/lessons 
def list_all_items(item_type):
    filtered_df = df[df['type'].str.lower() == item_type.lower()]
    items = filtered_df['title'].tolist()
    if not items:
        return f"No {item_type}s found in the dataset."
    reply = f"Here are all the {item_type}s available in GodTools:\n\n"
    for idx, title in enumerate(items, 1):
        reply += f"{idx}. {title}\n"
    return reply

# Detecting for the above List Commands 
def is_list_request(user_input, target_type):
    lowered = user_input.lower()
    return (
        "list" in lowered and target_type in lowered
    ) or (
        "show" in lowered and target_type in lowered
    ) or (
        "what" in lowered and target_type in lowered
    )

# follow up on the top 3 Tools 
def get_referenced_tool_index(user_input, top_tools):
    titles = [tool['title'] for tool in top_tools]
    joined_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
    prompt = f"""
You previously recommended these 3 GodTools resources:

{joined_titles}

User now said: "{user_input}"

Which resource are they referring to? Answer ONLY with 1, 2, 3, or "None" if unclear.
"""
    history_copy = conversation_history.copy()
    history_copy.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=history_copy
    )
    answer = response.choices[0].message.content.strip()
    return int(answer) - 1 if answer in ["1", "2", "3"] else None

# Chat Endpoint 
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global last_discussed_tool, top_tools, conversation_history

    user_input = request.user_input.strip()
    lowered_input = user_input.lower()
    conversation_history.append({"role": "user", "content": user_input})

    # checking for list requests
    if is_list_request(lowered_input, "tools"):
        reply = list_all_items("tool")
        conversation_history.append({"role": "assistant", "content": reply})
        return {"reply": reply}

    if is_list_request(lowered_input, "lessons"):
        reply = list_all_items("lesson")
        conversation_history.append({"role": "assistant", "content": reply})
        return {"reply": reply}

    # if not lists, Run FAISS search 
    if not top_tools:
        query_embedding = client.embeddings.create(
            input=[user_input],
            model="text-embedding-ada-002"
        ).data[0].embedding

        D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
        top_tools = df.iloc[I[0]].to_dict(orient='records')

        reply = "Top 3 matching GodTools resources:\n\n"
        for i, tool in enumerate(top_tools, start=1):
            reply += f"{i}. {tool['title']} ({tool['type']})\nDescription: {tool['description']}\nLanguages: {tool['language_support']}\n\n"

        reply += "You can now ask follow-up questions about these resources."
        conversation_history.append({"role": "assistant", "content": reply})
        return {"reply": reply}

    #  Followup (desccription for the specific tool)
    idx = get_referenced_tool_index(user_input, top_tools)

    if idx is not None and 0 <= idx < len(top_tools):
        tool = top_tools[idx]
        last_discussed_tool = tool

        prompt = (
            f"You are a helpful assistant answering questions about GodTools resources.\n\n"
            f"Resource:\nTitle: {tool['title']}\nType: {tool['type']}\nDescription: {tool['description']}\n"
            f"Languages: {tool['language_support']}\nRegions: {tool['available_regions']}\n\n"
            f"User asked: \"{user_input}\""
        )

        conversation_history.append({"role": "user", "content": prompt})

        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )

        bot_reply = chat_response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": bot_reply})
        return {"reply": bot_reply}

    # Fallback 
    if last_discussed_tool:
        fallback_prompt = (
            f"User follow-up:\n\n"
            f"Resource details:\nTitle: {last_discussed_tool['title']}\nDescription: {last_discussed_tool['description']}\n\n"
            f"User said: \"{user_input}\""
        )

        conversation_history.append({"role": "user", "content": fallback_prompt})

        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )

        bot_reply = chat_response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": bot_reply})
        return {"reply": bot_reply}

    # Total Fallback
    total_fallback_prompt = (
        f"The user said: \"{user_input}\" and there's no clear reference to a GodTools resource. Please provide a helpful, GodTools-related response."
    )

    conversation_history.append({"role": "user", "content": total_fallback_prompt})

    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    bot_reply = chat_response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": bot_reply})
    return {"reply": bot_reply}

# Refresh chatbot 
@app.post("/reset")
def reset_conversation():
    global conversation_history, top_tools, last_discussed_tool
    conversation_history.clear()
    top_tools = None
    last_discussed_tool = None
    return {"message": "Conversation history cleared."}


