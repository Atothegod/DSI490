# DSI490 — Agent 1 (Flood Incident Intake & Support)

FastAPI service ของ Agent 1 บน LINE Official Account: รับแจ้งเหตุน้ำท่วม ให้คำแนะนำเบื้องต้นจาก Knowledge Base
รับฟังความกังวล และบันทึกเคสที่ผู้ใช้ยืนยันแล้วลง PostgreSQL ใช้ DSPy + Gemini

โค้ดหลักอยู่ใน `dspy-agent/app` ทดลองผ่านหน้าเว็บหรือ notebook ใน `lab/1` ได้

## Current data contract (v4)

`problem` รวมปัญหา สถานการณ์ อาการ และความช่วยเหลือที่ต้องการไว้ด้วยกัน ไม่มี `needs` แยก
Priority คือความเร่งด่วนในการช่วยเหลือ ไม่ใช่ความรุนแรงของน้ำท่วม ใช้ Project-defined Ruleset `1.1.1` ใน `app/priority.py`
DSPy สกัด Risk Facts แบบ true / false / unknown พร้อมหลักฐานราย fact โค้ดเลือกกฎแรกที่เข้าเงื่อนไข ข้อมูลไม่พอจะพักเป็น P1 และถามข้อมูลเพิ่มก่อนยืนยัน ไม่ส่งระดับที่เดาหรือ null ไป Pipeline
เมื่อ context ครบ ให้ผู้ใช้ตรวจและยืนยันก่อนส่งหลังบ้าน สำหรับการปรึกษาจะคุยต่อได้โดยไม่บังคับถามข้อมูลติดต่อ
เมื่อพร้อมส่งต่อกด “กรอกข้อมูลส่งต่อ” เพื่อเติมจำนวนคน พิกัด และข้อมูลติดต่อ
`problem` ในฉบับยืนยันใช้ข้อความปัจจุบันที่ LLM เรียบเรียงและแก้ข้อมูลเก่าแล้ว ไม่ส่งข้อความ “ขอแก้…” ทั้งบทสนทนา
ข้อความดิบยังอยู่ใน `case_raw_messages` และร่าง เพื่อเก็บประวัติ; confirmed version เก่าไม่ถูกเขียนทับ

Pipeline ปัจจุบันอ่าน `pipeline_confirmed_cases_v4` ผ่าน `Store.pipeline_rows()` (schema_version `4.0`)
ผลเป็น `incident_priority: {level, reason}` เท่านั้น ไม่ส่ง Risk Facts, source, rule_id, ruleset_version หรือเวลาประเมิน
Audit ทุกการประเมินอยู่ใน `case_events`: `rule_id`, `ruleset_version`, `evaluated_at` พร้อม level และ reason
`pipeline_case_contacts` แยกข้อมูลติดต่อ ส่วน views v2/v3 และ confirmed versions เก่ายังคงไว้โดยไม่เขียนทับ
PostgreSQL role เดิมต้องเพิ่ม `GRANT SELECT ON pipeline_confirmed_cases_v4 TO pipeline_reader;` ก่อนเปลี่ยนตัวอ่าน Pipeline

| Rule | เงื่อนไข | ระดับ |
| --- | --- | --- |
| C1 | life_threat=true | critical |
| H1 | trapped=true และ water_rising=true | high |
| H2 | cannot_self_evacuate=true และ in_safe_place ไม่ใช่ true (รวม unknown ตามกฎนี้โดยเฉพาะ) | high |
| H3 | water_rising=true และ in_safe_place=false | high |
| H4 | essential_medication_risk=true | high |
| H5 | trapped=true และ unsafe_to_evacuate=true | high |
| M1 | basic_needs_unmet=true | medium |
| M2 | trapped=true | medium |
| L1 | in_safe_place=true และผู้แจ้งบอกว่าไม่ต้องการความช่วยเหลือ | low |
| P1 | ไม่เข้ากฎใด / ข้อมูลไม่พอ | รอประเมิน — ยังยืนยันหรือส่งต่อไม่ได้ |

กลุ่มเปราะบาง อายุ ชื่อยา คำขอแพทย์ หรือระดับน้ำอย่างเดียวไม่ยกระดับอัตโนมัติ ไม่ถามระดับน้ำเพื่อจัด priority
ข้อความใหม่ที่มีหลักฐานแก้ข้อเท็จจริงเดิมได้ ส่วน unknown ไม่ล้างค่าที่มีหลักฐานแล้ว
Critical ให้คำแนะนำจาก KB และช่องทางฉุกเฉินที่มีใน KB ก่อนถามข้อมูล ขอหมุดก่อนเมื่อยังไม่มีพิกัด
จากนั้นเก็บข้อมูลและให้ผู้ใช้ยืนยันตาม flow เดิม Draft ไม่ถูกส่งเข้า Pipeline และไม่อ้างว่าเจ้าหน้าที่รับเคสแล้ว

## Run

เชื่อมบัญชี LINE จริง: [คู่มือ deploy และ Flex Messages](dspy-agent/DEPLOY_LINE.md)
พร้อมไฟล์ตั้งค่าตัวอย่าง [`.env.line.example`](dspy-agent/.env.line.example)

```bash
docker compose up -d --build
```

API อยู่ที่ `http://localhost:8010`

## หน้าแชตจำลอง LINE และแผงหลังบ้าน

```bash
cd dspy-agent
python -m app.dev_chat
```

เปิด http://127.0.0.1:8011/dev/chat มีปุ่มยินยอม ส่งพิกัด ยืนยัน และแผงอ่านข้อมูลร่าง/ฉบับยืนยันจาก Database จริง อัปเดตหลังทุกข้อความ
ก่อนยืนยัน Pipeline จะเป็น `[]`; หลังยืนยันแสดง payload พร้อม `incident_priority` จาก `pipeline_confirmed_cases_v4` และข้อมูลติดต่อแยกส่วน
ใช้ Python environment ที่ติดตั้ง requirements แล้ว อ่าน key จาก `dspy-agent/.env` หรือ `lab/1/.env`
SQLite ชั่วคราวแยกจากฐานข้อมูลเดิม ปิด server แล้วข้อมูลหาย เปิดรับเฉพาะ localhost และปิด LINE transport
แผงตรวจข้อมูลเปิดเฉพาะ launcher นี้กับฐานข้อมูลในหน่วยความจำ ใช้ session cookie และไม่เปลี่ยนสิทธิ์ admin ของแอปปกติ
ใช้ข้อมูลสมมติ เรียก Gemini จริงมีค่าใช้จ่าย เปลี่ยนพอร์ตด้วย `python -m app.dev_chat --port 8012`

## ทดลองคุย

```bash
curl -X POST http://localhost:8010/chat -H "Content-Type: application/json" \
  -d '{"user_id": "test01", "postback": "consent=yes"}'
curl -X POST http://localhost:8010/chat -H "Content-Type: application/json" \
  -d '{"user_id": "test01", "input_text": "น้ำท่วมบ้านสูงถึงเอว ออกไม่ได้ มีเด็ก 2 คน"}'
curl -X POST http://localhost:8010/chat -H "Content-Type: application/json" \
  -d '{"user_id": "test01", "location": {"latitude": 14.95, "longitude": 100.36}}'
```

ส่ง `input_text`, `location` หรือ `postback` อย่างใดอย่างหนึ่ง ระบบโหลดเคสจาก Database เอง จึงไม่ต้องส่ง history
คำตอบมี `output_text`, `quick_replies` (กดปุ่มโดยส่ง `data` เป็น `postback`), `case_id`, `intake_status`, `pending_question`, `missing_slots`

## Endpoints

- `GET /health`
- `POST /chat`
- `POST /line/webhook` ตรวจ signature และตอบผ่าน LINE Reply API
- `POST /cases/{case_id}/close` เจ้าหน้าที่หลังบ้านปิดเคส ต้องใช้ admin bearer token
- `GET /docs`

## Pipeline

Pipeline ไม่เรียก API ของ Agent 1 แต่อ่านจาก PostgreSQL โดยตรง

- `pipeline_confirmed_cases_v4` — Confirmed Version ล่าสุดพร้อม `incident_priority` เฉพาะ level/reason ไม่มีข้อมูล audit หรือข้อมูลติดต่อ
- `pipeline_confirmed_cases_v3` — View สำหรับ integration รุ่นเดิม
- `pipeline_confirmed_cases_v2` — Confirmed Version ล่าสุด ไม่มีชื่อ เบอร์ หรือ priority (schema_version ของแถวใหม่เป็น 4.0 ตามเวอร์ชันที่บันทึก)
- `pipeline_case_contacts` — ชื่อ นามสกุล เบอร์ เฉพาะ Role ที่ต้องติดต่อผู้แจ้ง
- `case_events` — `incident_submitted`, `case_updated` ใช้ตรวจจับเคสใหม่และเคสที่แก้ไข

กำหนดสิทธิ์หลังบ้านผ่าน SQL views โดยแยกสิทธิ์ข้อมูลติดต่อออกจากข้อมูลเหตุ

## Notebook สำหรับทดสอบ

- [agent1_line_chat.ipynb](lab/1/agent1_line_chat.ipynb) — หน้าแชตจำลอง LINE กับ Gemini จริง
- [test_extraction_v2.ipynb](lab/1/test_extraction_v2.ipynb) — ตรวจผลสกัด หลักฐาน และข้อมูลที่บันทึกในฐานข้อมูล

เมื่อช่องข้อมูลครบแต่ยัง P1 บอทถาม Risk Facts ที่ยังขาดทีละเรื่อง ไม่ถามระดับน้ำเชิงความลึก ไม่ถามคำถามเดิมซ้ำ หากผู้แจ้งไม่ทราบจริง ๆ จะพักเคสไว้รับข้อมูลใหม่ โดยไม่แปลง unknown เป็น false
