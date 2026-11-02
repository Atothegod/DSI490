# DSI490 — Agent 1 (Flood Incident Intake & Support)

FastAPI service ของ Agent 1 บน LINE Official Account: รับแจ้งเหตุน้ำท่วม ให้คำแนะนำเบื้องต้นจาก Knowledge Base
รับฟังความกังวล และบันทึกเคสที่ผู้ใช้ยืนยันแล้วลง PostgreSQL ใช้ DSPy + Gemini

ออกแบบตามเอกสาร "DSI490 — การออกแบบ Agent 1 ฉบับปรับปรุง" ดูการแบ่ง Module และสิ่งที่ยังไม่ได้ทำที่
[dspy-agent/docs/implementation.md](dspy-agent/docs/implementation.md)

## Run

```bash
docker compose up -d --build
```

API อยู่ที่ `http://localhost:8010`

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

- `pipeline_confirmed_cases` — Confirmed Version ล่าสุดของแต่ละเคส (ไม่มีชื่อและเบอร์)
- `pipeline_case_contacts` — ชื่อ นามสกุล เบอร์ เฉพาะ Role ที่ต้องติดต่อผู้แจ้ง
- `case_events` — `incident_submitted`, `case_updated` ใช้ตรวจจับเคสใหม่และเคสที่แก้ไข

สิทธิ์ Role อยู่ที่ [dspy-agent/db/pipeline_roles.sql](dspy-agent/db/pipeline_roles.sql)
