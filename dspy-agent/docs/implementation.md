# Agent 1 — สถานะการ Implement

อ้างอิงเอกสาร "DSI490 — การออกแบบ Agent 1 ฉบับปรับปรุง" (Nov 2, 2026) ซึ่งแทนที่ Design Decisions v1.3 ทั้งหมด
เลขหัวข้อในโค้ดอ้างตามเอกสารฉบับนี้

## Module → ไฟล์

| Module (หัวข้อ 8) | ไฟล์ | ใช้ LLM |
| --- | --- | --- |
| Consent Manager | `app/agent.py` (`_consent`) | ไม่ใช้ |
| Privacy Filter | `app/privacy.py` | ไม่ใช้ |
| Information Extractor + Intent Classifier | `app/extraction.py` (Signature เดียว + `ground()` ตัดค่าที่ไม่มีหลักฐาน) | ใช้ |
| Slot Manager | `app/case.py` (`refresh_slots`, `next_slot`) | ไม่ใช้ |
| Decision & Safety Logic | `app/agent.py` (`_support`, `_continue`) | ไม่ใช้ |
| Knowledge Retriever | `app/knowledge.py` + `app/knowledge_th.csv` | ไม่ใช้ |
| Response Generator + Validator | `app/response.py` | ใช้ / กฎ |
| Priority Engine (Ruleset 1.1.0) | `app/priority.py` | ไม่ใช้ |
| Summary Generator | `app/case.py` (`summary_text`, Template) | ไม่ใช้ |
| Confirmation Handler + Version Manager | `app/agent.py` (`_confirm`), `app/store.py` | ไม่ใช้ |
| Pipeline Data View | `app/store.py` (View `pipeline_confirmed_cases`, `pipeline_case_contacts`), `db/pipeline_roles.sql` | ไม่ใช้ |

ข้อความทั้งหมดที่ส่งถึงผู้ใช้ (ยกเว้นส่วนที่ Gemini เรียบเรียง) อยู่ใน `app/templates_th.json`

## Postback ของปุ่ม Quick Reply

| data | ความหมาย |
| --- | --- |
| `consent=yes` / `consent=no` | ยินยอม / ไม่ยินยอม |
| `confirm=INC-000001:<draft_rev>` / `edit=...` | ยืนยัน / แก้ไขสรุป (revision กันการกดปุ่มของสรุปเก่า) |
| `new_case` | แจ้งเหตุใหม่ (พิมพ์ "แจ้งเหตุใหม่" ได้ผลเหมือนกัน) |
| `case_select=INC-...` | เลือกเคสเมื่อข้อความกำกวม |
| `location_choice=relocate` / `=new` | หมุดใหม่: ย้ายจุดเคสเดิม / แจ้งเหตุใหม่ |
| `reuse_contact=yes` / `=no` | เคสใหม่ใช้ชื่อ นามสกุล เบอร์เดิมหรือไม่ |

## ทดสอบ

```bash
pip install -r requirements.txt pytest
pytest tests
```

ชุดทดสอบไม่เรียก Gemini (`tests/conftest.py` ใช้ FakeExtractor) และครอบคลุม T1–T16, U1–U24, MC1–MC7, R1–R7, S1–S7

ทดลองคุยกับ Gemini จริงได้ที่ `lab/1/agent1_chat.ipynb` (ติดตั้ง `requirements-notebook.txt`)

## สิ่งที่ยังไม่ได้ทำ / ต้องตัดสินใจ

- **Knowledge Base `mock-0.1` 29 เอกสารไม่อยู่ใน repo** ตอนนี้มี 18 เอกสาร (`kb_version = mock-0.0`): 14 ชิ้นแปลงจาก KB เดิม + 4 เบอร์ฉุกเฉิน ยังขาดหมวดสนับสนุนด้านจิตใจ ทุกชิ้นเป็น `mock_unreviewed`
- **Retrieval Threshold** ยังเป็นค่าชั่วคราว 0.12 ต้องรันกับ `kb_eval_questions` (27 ข้อ) ซึ่งยังไม่อยู่ใน repo Retriever เป็น trigram overlap ยังไม่ใช่ Embedding/pgvector
- **เข้ารหัส Personal Data** ยังไม่ทำ ตอนนี้แยกตาราง (`case_personal_data`, `case_version_contacts`, `user_line_ids`) และแยก View/Role แล้ว แต่ยังไม่มี encryption at rest และ access log
- **สิทธิ PDPA (ลบข้อมูลส่วนบุคคล)** ยังไม่มี endpoint
- **conversation_summary** ตอนนี้เป็นข้อความ 3 ข้อความล่าสุดที่ผ่าน Privacy Filter ไม่ใช่สรุปจาก LLM
- **วิธีอัปเดตสถานะปิดเคส** ใช้ `POST /cases/{case_id}/close` ชั่วคราว รอตกลงกับทีม Pipeline
- Privacy Filter ใช้ regex: ชื่อที่ไม่มีคำนำหน้า (เช่น "สมชายแจ้งว่า...") ในข้อความอิสระยังจับไม่ได้ ยกเว้นเป็นคำตอบของคำถามชื่อโดยตรง
- ยังไม่ได้ทดสอบกับ PostgreSQL จริงและ Gemini จริง (ทดสอบด้วย SQLite + DummyLM)
