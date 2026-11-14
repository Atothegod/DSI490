# DSI490 — Agent 1 (Flood Incident Intake & Support)

FastAPI service ของ Agent 1 บน LINE Official Account: รับแจ้งเหตุน้ำท่วม ให้คำแนะนำเบื้องต้นจาก Knowledge Base
รับฟังความกังวล และบันทึกเคสที่ผู้ใช้ยืนยันแล้วลง PostgreSQL ใช้ DSPy + Gemini

ออกแบบตามเอกสาร "DSI490 — การออกแบบ Agent 1 ฉบับปรับปรุง" ดูการแบ่ง Module และสิ่งที่ยังไม่ได้ทำที่
[dspy-agent/docs/implementation.md](dspy-agent/docs/implementation.md)

## Current data contract (v2)

`problem` รวมปัญหา สถานการณ์ อาการ และความช่วยเหลือที่ต้องการไว้ด้วยกัน ไม่มี `needs` แยก
และไม่มีการประเมิน/ส่ง `incident_priority` อีกต่อไป Extractor ถาม context ที่ยังขาดครั้งละหนึ่งเรื่อง
เมื่อ context ครบ จึงเก็บจำนวนคน พิกัด และข้อมูลติดต่อ แล้วให้ผู้ใช้ยืนยันเหมือนเดิม

Pipeline รุ่นใหม่อ่าน `pipeline_confirmed_cases_v2` (schema_version `2.0`) หรือ `Store.pipeline_rows()`
และใช้ `pipeline_case_contacts` สำหรับข้อมูลติดต่อ แถวเก่า schema_version `1.0` ยังเป็น snapshot เดิม
SQL view เก่าและคอลัมน์ priority เดิมคงไว้เพื่อไม่ลบประวัติ แต่เคส/เวอร์ชันใหม่ไม่ประเมินความเสี่ยง
ผู้ใช้ PostgreSQL ที่มี role เดิมแล้วต้องเพิ่ม `GRANT SELECT ON pipeline_confirmed_cases_v2 TO pipeline_reader;`
หลังแอปสร้าง view ใหม่ จากนั้นเปลี่ยน consumer ไปอ่าน view v2

```json
{"incident": {"problem": "น้ำท่วมบ้าน ออกไม่ได้ ต้องการเรือ", "affected_people": {"other": 2}}, "location": {"latitude": 14.9, "longitude": 100.4}}
```

## Run

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
ก่อนยืนยัน Pipeline จะเป็น `[]`; หลังยืนยันแสดง payload จาก `pipeline_confirmed_cases_v2` และข้อมูลติดต่อแยกส่วน
ใช้ Python environment ที่ติดตั้ง requirements แล้ว อ่าน key จาก `dspy-agent/.env` หรือ `lab/1/.env`
SQLite ชั่วคราวแยกจากฐานข้อมูลเดิม ปิด server แล้วข้อมูลหาย เปิดรับเฉพาะ localhost และปิด LINE transport
แผงตรวจข้อมูลเปิดเฉพาะ launcher นี้กับฐานข้อมูลในหน่วยความจำ ใช้ session cookie และไม่เปลี่ยนสิทธิ์ admin ของแอปปกติ
ใช้ข้อมูลสมมติ เรียก Gemini จริงมีค่าใช้จ่าย เปลี่ยนพอร์ตด้วย `python -m app.dev_chat --port 8012`

## ทดลองคุย

ทดลอง Agent กับ Gemini จริงใน Terminal โดยไม่เปิด Docker หรือ LINE:

```bash
python dspy-agent/scripts/chat_real.py
```

ใช้ Python environment ที่ติดตั้ง `dspy-agent/requirements.txt` แล้ว สคริปต์อ่าน key
จาก `dspy-agent/.env` หรือ `lab/1/.env` โดยไม่แสดง key เริ่มคุยด้วย `/consent`
กดปุ่มด้วย `/button DATA` และส่งพิกัดด้วย `/location LAT LNG` ออกด้วย `/quit`
ใช้ข้อมูลสมมติเท่านั้น: เรียก Gemini จริง มีค่าใช้จ่ายตามบัญชี ใช้ SQLite ในหน่วยความจำ
และอนุญาต KB ที่ยังไม่ผ่านผู้ตรวจเฉพาะการทดลอง ปิดโปรแกรมแล้วข้อมูลหาย ไม่ส่งข้อความเข้า LINE
รันบทสนทนาสมมติสั้น ๆ อัตโนมัติด้วย `python dspy-agent/scripts/chat_real.py --smoke`

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

- `pipeline_confirmed_cases_v2` — Confirmed Version ล่าสุด ไม่มีชื่อ เบอร์ หรือ priority (แยกรุ่นด้วย schema_version)
- `pipeline_case_contacts` — ชื่อ นามสกุล เบอร์ เฉพาะ Role ที่ต้องติดต่อผู้แจ้ง
- `case_events` — `incident_submitted`, `case_updated` ใช้ตรวจจับเคสใหม่และเคสที่แก้ไข

สิทธิ์ Role อยู่ที่ [dspy-agent/db/pipeline_roles.sql](dspy-agent/db/pipeline_roles.sql)
