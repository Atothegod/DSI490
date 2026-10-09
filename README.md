# Lightweight DSPy Agent API

FastAPI service สำหรับเรียก DSPy ด้วย Gemini 2.5 Flash โดยรับ payload แค่ข้อความล่าสุดกับ history จาก session

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
  -d '{
    "input_text": "ช่วยสรุปบทสนทนานี้ให้หน่อย",
    "history": [
      {"role": "user", "content": "ผมอยากทำ AI agent"},
      {"role": "assistant", "content": "เริ่มจาก FastAPI + DSPy ได้ครับ"}
    ]
  }'
```

## Endpoints

- `GET /health`
- `POST /chat`
- `GET /docs`
