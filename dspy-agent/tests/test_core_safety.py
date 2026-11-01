"""Core safety bypasses ranking, never human approval or source validity."""
import csv
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlparse

import pytest

from app.knowledge import KnowledgeBase
from app.schemas import now_bkk

DATA = Path(__file__).parents[1] / 'app' / 'knowledge_th.csv'


def reviewed_copy(tmp_path, **patch):
    with DATA.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    checked=now_bkk().isoformat()
    for row in rows:
        row.update(review_status='approved',reviewed_by='test_fixture_reviewer',reviewed_at=checked,verified_at=checked)
        row.update(patch)
    path=tmp_path/'reviewed_fixture.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return KnowledgeBase(path,threshold=1.1)


def test_shipped_drafts_have_sources_but_cannot_bypass_review():
    kb=KnowledgeBase(DATA)
    assert len(kb.rows)==14
    assert sum(r["is_core_safety"]=="true" for r in kb.rows)==5
    assert sum(r["is_core_safety"]=="false" for r in kb.rows)==9
    assert all(urlparse(r['url']).hostname in {'ddc.moph.go.th','www.ddc.moph.go.th'} and r['source_checked_at'] for r in kb.rows)
    assert all(r['review_status']=='pending_review' and not r['reviewed_by'] and not r['reviewed_at'] and not r['verified_at'] for r in kb.rows)
    assert kb.core_snippets()==[]
    assert kb.search('ตัดไฟ')==[]
    assert kb.evidence_for('ตัดไฟ')==[]


def test_core_evidence_is_available_even_when_score_threshold_cannot_pass(tmp_path):
    kb=reviewed_copy(tmp_path)
    assert kb.search('น้ำเข้าบ้านต้องตัดไฟไหม')==[]
    evidence=kb.evidence_for('น้ำเข้าบ้านต้องตัดไฟไหม')
    assert [e.chunk_id for e in evidence]==['core_electricity_safe_cutoff']


def test_core_does_not_broadcast_all_snippets_for_unrelated_queries(tmp_path):
    kb=reviewed_copy(tmp_path)
    assert kb.evidence_for('วันนี้น้ำในแม่น้ำสูงกี่เมตร')==[]
    assert [e.chunk_id for e in kb.core_snippets('เดินฝ่าน้ำไหลเชี่ยวได้ไหม')]==['core_avoid_strong_current']


def test_core_preserves_approval_and_expiration_guards(tmp_path):
    for patch in ({'review_status':'pending_review'},{'reviewed_by':''},{'reviewed_at':'not-a-date'},
                  {'verified_at':(now_bkk()+timedelta(days=1)).isoformat()},
                  {'valid_until':(now_bkk()-timedelta(days=1)).isoformat()}, {'category':'B'}):
        assert reviewed_copy(tmp_path,**patch).core_snippets()==[]


def test_normal_and_core_hits_are_deduplicated(tmp_path):
    kb=reviewed_copy(tmp_path)
    kb.threshold=0
    evidence=kb.evidence_for('ตัดไฟ')
    ids=[e.chunk_id for e in evidence]
    assert ids[0]=='core_electricity_safe_cutoff' and len(ids)==len(set(ids))


@pytest.mark.parametrize("query,expected", [
    ("อาหารที่โดนน้ำท่วมยังกินได้ไหม", "kb_food_flood_contamination"),
    ("น้ำสำหรับทำอาหารช่วงน้ำท่วมควรใช้อะไร", "kb_clean_drinking_cooking_water"),
    ("ป้องกันโรคฉี่หนูช่วงน้ำท่วมอย่างไร", "kb_leptospirosis_exposure_prevention"),
    ("ดูแลเท้าหลังลุยน้ำท่วมอย่างไร", "kb_skin_wounds_after_floodwater"),
    ("งูหนีน้ำเข้าบ้านทำอย่างไร", "kb_snakes_flood_home"),
    ("ก่อนน้ำท่วมควรเตรียมอะไร", "kb_prepare_essentials"),
    ("หลังน้ำลดทำความสะอาดบ้านอย่างไร", "kb_cleanup_protective_equipment"),
    ("ตู้เย็นที่โดนน้ำท่วมเสียบปลั๊กได้ไหม", "kb_submerged_appliances_check"),
    ("ป้องกันเด็กจมน้ำช่วงน้ำท่วมอย่างไร", "kb_children_near_water"),
])
def test_general_kb_queries_find_reviewed_fixture_rows(tmp_path, query, expected):
    kb=reviewed_copy(tmp_path)
    kb.threshold=0.12
    assert expected in [e.chunk_id for e in kb.search(query)]


def test_rows_are_unique_complete_and_general_rows_have_no_core_triggers():
    rows=KnowledgeBase(DATA).rows
    assert len({row["chunk_id"] for row in rows})==len(rows)
    assert {row["topic"] for row in rows}=={"safety","health","preparation","recovery"}
    for row in rows:
        assert all(row[key].strip() for key in ("original_text","reviewed_text","example_questions","source_org","review_notes"))
        assert row["reviewed_text"].endswith("ค่ะ")
        if row["is_core_safety"]=="false":
            assert row["core_triggers"]==""
