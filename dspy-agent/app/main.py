import os
from typing import Literal

import dspy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


load_dotenv()


class HistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    input_text: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    output_text: str
    model: str


class AgentSignature(dspy.Signature):
    """YOU are ICS commander please ask user about the problem there're facing then guide them to solve it."""

    history: str = dspy.InputField(desc="Conversation history from the current session.")
    input_text: str = dspy.InputField(desc="The latest user message.")
    output_text: str = dspy.OutputField(desc="Helpful guide answer for the user.")


app = FastAPI(title="Lightweight DSPy Agent API")
agent = dspy.Predict(AgentSignature)


def configure_dspy() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY.")

    model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
    lm = dspy.LM(model=model, api_key=api_key)
    dspy.configure(lm=lm)
    return model


def format_history(history: list[HistoryMessage]) -> str:
    if not history:
        return "No prior session history."

    return "\n".join(f"{message.role}: {message.content}" for message in history)


@app.on_event("startup")
def startup() -> None:
    app.state.model = configure_dspy()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": app.state.model}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = agent(history=format_history(payload.history), input_text=payload.input_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    return ChatResponse(output_text=result.output_text, model=app.state.model)
