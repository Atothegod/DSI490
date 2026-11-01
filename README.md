# Lightweight DSPy Agent API

FastAPI service ของ Agent 1 (Flood Intake & Support) จังหวัดสิงห์บุรี ใช้ DSPy + Gemini 3.1 Flash-Lite
รับข้อความทีละข้อความต่อผู้ใช้ ระบบจำบทสนทนาเองฝั่ง server จึงไม่ต้องส่ง history

## Run

```bash
docker compose up -d --build
```

API จะอยู่ที่:

```txt
http://localhost:8010
```

## Payload

```bash
curl -X POST http://localhost:8010/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test01", "input_text": "น้ำท่วมบ้าน ช่วยด้วย"}'
```

ใช้ `user_id` เดิมเพื่อคุยต่อในเรื่องเดิม คำตอบเป็น `AgentTurnOutput` เช่น `output_text`, `incident_status`, `next_actions`, `urgent_flags`, `pending_question`

ปรับ runtime ตาม Design Decisions v1.3: บันทึกด้วย PostgreSQL/SQLite และ transactional outbox; เมื่อยังไม่ตั้ง transport ของ Agent 2 สถานะจะเป็น pending ตามจริง

ดูรายละเอียดการตั้งค่า สัญญาส่งต่อฉบับร่าง และสิ่งที่รอทีมตรวจที่ [v1.3 implementation](dspy-agent/docs/v1.3-implementation.md)

ตัวอย่าง input ให้ Agent 2: [ส่งครั้งแรก](dspy-agent/agent2_input_example.json) และ [อัปเดตภายหลัง](dspy-agent/agent2_update_example.json)

## Endpoints

- `GET /health`
- `POST /chat`
- `POST /chat/reset` เริ่มบทสนทนาใหม่ของ `user_id` นั้น
- `GET /docs`

- `POST /line/webhook` ตรวจ signature และตอบผ่าน LINE Reply API เมื่อกำหนด credentials
- `POST /incidents/operation`, `POST /incidents/priority`, `GET /contacts/{contact_ref}` ต้องใช้ admin bearer token
