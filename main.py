from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel

# Constants
SESSION_TIMEOUT = timedelta(minutes=30)
FUZZY_MATCH_THRESHOLD = 80
MODEL_NAME = "gpt-4o"
MODEL_TEMPERATURE = 0.7


@dataclass
class Session:
    messages: List[BaseMessage]
    last_active: datetime


class ChatResponse(BaseModel):
    text: str
    products: List[str] = []


class NudgeResponse(BaseModel):
    text: str


# Load configuration
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources
with open('prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

with open('products.txt', 'r') as f:
    PRODUCT_LIST = [line.strip() for line in f.readlines() if line.strip()]

# Initialize LLM
llm = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)

# Session storage
sessions: Dict[str, Session] = {}


def is_session_expired(last_active: datetime) -> bool:
    return datetime.utcnow() - last_active > SESSION_TIMEOUT


def get_or_create_session(session_id: str) -> Session:
    now = datetime.utcnow()
    if session_id not in sessions or is_session_expired(sessions[session_id].last_active):
        sessions[session_id] = Session(
            messages=[SystemMessage(content=SYSTEM_PROMPT)],
            last_active=now
        )
    else:
        sessions[session_id].last_active = now
    return sessions[session_id]


@lru_cache(maxsize=100)
def generate_title_with_llm(user_input: str, products_str: str) -> str:
    title_prompt = [
        SystemMessage(content="You are a creative assistant. Generate a short and catchy title summarizing the type of skin care items based on user intent and product names."),
        HumanMessage(
            content=f"User is shopping for: {user_input}\n\nRecommended products:\n{products_str}\n\nGive me a short catchy title (under 8 words).")
    ]
    return llm.invoke(title_prompt).content.strip().strip('"')


@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    try:
        body = await request.json()
        user_input = body.get("message")
        if not user_input:
            raise HTTPException(status_code=400, detail="No message provided")

        session = get_or_create_session(session_id)
        session.messages.append(HumanMessage(content=user_input))

        ai_content = llm.with_structured_output(
            ChatResponse).invoke(session.messages)

        matched_products = ai_content.products
        cleaned_text = ai_content.text
        session.messages.append(AIMessage(content=cleaned_text))

        if matched_products:
            title = generate_title_with_llm(
                user_input, ", ".join(matched_products))
            return {
                "response": {
                    "text": cleaned_text,
                    "products": matched_products,
                    "title": title
                },
                "history": [msg.content for msg in session.messages if isinstance(msg, (HumanMessage, AIMessage))]
            }

        return {
            "response": {"text": cleaned_text},
            "history": [msg.content for msg in session.messages if isinstance(msg, (HumanMessage, AIMessage))]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nudge/{session_id}")
async def generate_nudge(session_id: str, request: Request):
    try:
        body = await request.json()
        product_name = body.get("product_name")
        if not product_name:
            raise HTTPException(
                status_code=400, detail="No product name provided")

        session = get_or_create_session(session_id)
        history = [msg.content for msg in session.messages if isinstance(
            msg, (HumanMessage, AIMessage))]

        prompt = [
            SystemMessage(content="You are a persuasive, friendly fashion assistant. Based on the conversation, write a short, encouraging nudge for why this product would be a great choice for the user, incorporating styling tips, benefits, and making the user feel confident."),
            HumanMessage(content=f"Conversation:\n{chr(10).join(history)}\n\nProduct: {product_name}\n\nProvide a brief, upbeat nudge that includes 1 fun tip (with emojis). Ensure that the skincare tip has a punchy, engaging vibe, and the nudge should inspire confidence and excitement about the choice. Do not add any fluff words / non-meaningful words. It should be maximum 1 sentence.")
        ]

        nudge_response = llm.with_structured_output(
            NudgeResponse).invoke(prompt)
        nudge_text = nudge_response.text
        session.messages.append(AIMessage(content=nudge_text))

        return {"nudge": nudge_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted."}
    raise HTTPException(status_code=403, detail="Session not found")
