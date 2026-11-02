"""Response Validator (หัวข้อ 8) — กฎตรวจคำตอบที่ Gemini เรียบเรียง"""

from app.knowledge import KnowledgeBase
from app.response import Sentence, template_reply, validate
from tests.conftest import KB_PATH

KB = KnowledgeBase(KB_PATH, allow_unreviewed=True)
ELEC = KB.by_id["core_electricity_safe_cutoff"]


def s(kind, text, doc=None):
    return Sentence(kind=kind, text=text, kb_doc_id=doc)


def test_uncited_advice_is_dropped():
    kept, ok = validate([s("advice", "ให้ตัดไฟจากจุดที่ปลอดภัยครับ", ELEC.kb_doc_id), s("advice", "ไปศูนย์พักพิงวัดใหม่ครับ")], [ELEC], set())
    assert ok and [k.kb_doc_id for k in kept] == [ELEC.kb_doc_id]


def test_citation_must_be_retrieved_this_turn():
    kept, ok = validate([s("advice", "ให้ตัดไฟครับ", "kb_snakes_flood_home")], [ELEC], set())
    assert ok and kept == []


def test_forbidden_claims_fail():
    for text in ("เจ้าหน้าที่กำลังเดินทางไปครับ", "ตอนนี้คุณปลอดภัยแล้วครับ", "ระบบส่งทีมกู้ภัยให้แล้วครับ"):
        assert validate([s("empathy", text)], [ELEC], set()) == ([], False)


def test_unknown_numbers_and_phones_fail():
    assert not validate([s("advice", "โทร 1555 ครับ", ELEC.kb_doc_id)], [ELEC], set())[1]
    assert not validate([s("advice", "ติดต่อ 0812345678", ELEC.kb_doc_id)], [ELEC], {"0812345678"})[1]
    assert validate([s("advice", "โทร 1784 ครับ", ELEC.kb_doc_id)], [ELEC], {"1784"})[1]


def test_empathy_must_not_contain_advice():
    assert not validate([s("empathy", "เข้าใจครับ ควรรีบขึ้นที่สูง")], [], set())[1]
    assert validate([s("empathy", "เข้าใจครับว่าตอนนี้น่ากังวล")], [], set())[1]


def test_template_uses_kb_text_verbatim():
    out = template_reply("เข้าใจครับ", [ELEC])
    assert ELEC.text in out.text and out.kb_doc_ids == [ELEC.kb_doc_id] and out.used_template
