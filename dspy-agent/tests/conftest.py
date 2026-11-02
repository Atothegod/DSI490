import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import Agent1  # noqa: E402
from app.knowledge import KnowledgeBase  # noqa: E402
from app.response import ResponseGenerator  # noqa: E402
from app.schemas import Extraction, MessageIntent  # noqa: E402
from app.store import Store  # noqa: E402

KB_PATH = Path(__file__).resolve().parents[1] / "app" / "knowledge_th.csv"


class FakeExtractor:
    """แทน Gemini: คืน Extraction ตามข้อความที่ผ่าน Privacy Filter แล้ว และจดทุกข้อความที่ "LLM" เห็น"""

    def __init__(self):
        self.script: dict[str, Extraction] = {}
        self.seen: list[str] = []

    def on(self, text: str, **kw) -> None:
        kw.setdefault("intents", [MessageIntent.INCIDENT])
        self.script[text] = Extraction(**kw)

    def __call__(self, text, draft, pending, summary):
        self.seen.append(text)
        return self.script.get(text, Extraction()).model_copy(deep=True)


@pytest.fixture
def llm():
    return FakeExtractor()


@pytest.fixture
def store():
    s = Store("sqlite:///:memory:")
    yield s
    s.close()


@pytest.fixture
def agent(store, llm):
    kb = KnowledgeBase(KB_PATH, allow_unreviewed=True)
    return Agent1(store, kb, extractor=llm, generator=ResponseGenerator(use_llm=False))
