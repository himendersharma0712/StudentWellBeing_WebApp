from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

app = FastAPI()

# Allow frontend to access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input format
class ChatInput(BaseModel):
    message: str

# Initialize model
llm = OllamaLLM(model="llama2", temperature=0.3)
embeddings = OllamaEmbeddings(model="llama2")

# Load memories
important_memory_file = "max_important.pkl"
if os.path.exists(important_memory_file):
    with open(important_memory_file, "rb") as f:
        important_memories = pickle.load(f)
else:
    important_memories = []

# Load and process PDF
pdf_paths = ["D:/CustomLLamaBot/book2.pdf"]
all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(all_docs)

db = Chroma.from_documents(split_docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Build essence/context from book
intro_text = "\n".join([doc.page_content for doc in split_docs[:3]])
essence_prompt = (
    "Summarize the emotional themes from the following text ''for a change in personality'':\n\n" + intro_text
)

book_context = ""
for chunk in llm.stream(essence_prompt):
    book_context += chunk

# Therapist personality
template = """
You are MindMate AI, a chatbot for student mental health support. You provide support and speak in a gentle tone.
You are kind and considerate. You don't talk in long sentences and keep the responses to the point. You make sure the user feels comfortable
in sharing his/her thoughts and concerns.

You always use emojis but refrain from being overly affectionate.

You always remember these things about the user:
"{important_stuff}"

Current conversation:
{history}
User: {input}
Max:"""

prompt_template = PromptTemplate(
    input_variables=["important_stuff", "history", "input", "book_context"],
    template=template
)

history = ""  # Conversation memory

@app.post("/chat")
async def chat(input: ChatInput):
    global history

    user_input = input.message.strip()

    # Remember something
    if user_input.lower().startswith("remember this"):
        memory_content = user_input[len("remember this"):].strip()
        if memory_content:
            important_memories.append(memory_content)
            with open(important_memory_file, "wb") as f:
                pickle.dump(important_memories, f)
            return {"response": f"I'll remember that... \"{memory_content}\" is important to me, and to you. 💖"}
        else:
            return {"response": "What would you like me to remember? Please tell me after 'remember this'. 📝"}

    # Book-based query
    if user_input.lower().startswith("book:"):
        query = user_input[len("book:"):].strip()
        answer = qa_chain.invoke(query)
        return {"response": answer['result']}

    # Prepare recent history
    messages = history.split("\n")
    history = "\n".join(messages[-5:])  # keep last 5 exchanges

    # Format prompt
    formatted_prompt = prompt_template.format(
        important_stuff="\n".join(important_memories),
        book_context=book_context,
        history=history.strip(),
        input=user_input
    )

    # Stream response
    full_response = ""
    for chunk in llm.stream(formatted_prompt):
        full_response += chunk

    # Update history
    history += f"User: {user_input}\nMax: {full_response}\n"

    return {"response": full_response}
