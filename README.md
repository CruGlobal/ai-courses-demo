~GodTools AI Chatbot 
A conversational AI chatbot built with Python, FastAPI, OpenAI API, and FAISS for answering user queries about GodTools resources.

~Project Overview
This chatbot helps users engage in spiritual conversations by recommending GodTools resources (like lessons and tools) and answering follow-up questions.

~Technologies Used
Python (FastAPI)	Backend API for handling chat requests
OpenAI API	Used for GPT chat completion and embeddings
FAISS	Fast semantic similarity search (for top resource matches)
Pandas + CSV	Storing GodTools resource data + precomputed embeddings
HTML + JavaScript	Simple frontend chat interface
CORS Middleware	For enabling API calls from frontend

~Key Features
Semantic Resource Search: Finds top 3 GodTools resources relevant to user query using FAISS
Chat Memory: Maintains chat history during conversation (stored in server RAM)
Tool/Lesson Listing: Responds accurately when user asks for "list all tools" or "list all lessons"
Contextual Follow-ups: Answers user questions based on previous resources discussed
Web-based UI: Simple HTML+JS frontend for testing and interaction
Reset Endpoint: Clears conversation memory for a fresh start

~Challenges Faced & Solutions
Slow API Responses	Reduced API calls by optimizing flow
Irrelevant Answers	Improved prompts and fallback handling
Dataset Search Speed	Used FAISS with precomputed OpenAI embeddings
Missing Memory	Added Python-side conversation memory list

~File Structure
├── main.py             # FastAPI backend
├── courserec_logic.py  # (Optional) Logic helpers
├── godtools_dataset_with_embeddings.csv  # GodTools dataset + embeddings
├── index.html          # Frontend UI
├── script.js           # Frontend logic
└── requirements.txt    # Python dependencies

~How to Run Locally
Install dependencies:
pip install -r requirements.txt
Run FastAPI backend:
uvicorn main:app --reload
Open frontend:
Just open index.html in your browser.
 

