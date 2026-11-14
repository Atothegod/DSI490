# ต่อ Flood Care เข้า LINE Official Account

โค้ดพร้อมรับ webhook และสร้าง Flex Messages แล้ว ขั้นตอนนี้ยังไม่ส่งข้อความหรือ deploy เข้า LINE จนกว่าจะตั้งค่า channel และ HTTPS ของจริง

## 1. ตั้งค่าในเครื่องเซิร์ฟเวอร์

```bash
cp dspy-agent/.env.line.example dspy-agent/.env
```

กรอก `GEMINI_API_KEY`, `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` ใน `.env` โดยไม่ส่งค่าลับผ่านแชตหรือ commit ลง Git
สร้าง `LINE_USER_HMAC_SECRET` และ `AGENT_ADMIN_TOKEN` แยกกัน ด้วยคำสั่งนี้ครั้งละหนึ่งค่า:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

เก็บ HMAC secret ให้คงที่ เพราะใช้ผูก LINE user กับบัญชีภายใน การเปลี่ยนค่าจะเปลี่ยนรหัสบัญชีภายใน
`APP_ENV=production` ปิด `/dev/chat` และ `/dev/inspector`; `/chat` ต้องใช้ admin bearer token
`LINE_MESSAGE_STYLE=flex` เปิดการ์ด หรือใช้ `text` เพื่อกลับเป็นข้อความและ Quick Reply

## 2. เอกสารความรู้

โหมด production อ่านเฉพาะเอกสาร `approved` ใน `app/knowledge_th.csv` เอกสารร่าง `mock_unreviewed` จะไม่ถูกนำมาตอบ
ต้องให้ผู้รับผิดชอบตรวจเนื้อหาและแหล่งข้อมูลก่อนเปลี่ยนสถานะเป็น approved โค้ดไม่ได้อนุมัติเนื้อหาให้โดยอัตโนมัติ
หาก KB ยังว่าง บอทไม่สามารถให้คำแนะนำหรือช่องทางฉุกเฉินที่ตรวจสอบแล้วได้ จึงต้องจัดการข้อนี้ก่อนเปิดรับผู้ใช้จริง

## 3. เปิดบริการและ HTTPS

จากโฟลเดอร์หลักของโปรเจกต์:

```bash
docker compose up -d --build
curl http://127.0.0.1:8010/health
```

Compose ใช้ PostgreSQL พร้อม persistent volume จากนั้นตั้ง reverse proxy ของเซิร์ฟเวอร์ให้โดเมน HTTPS ชี้มายัง port 8010
Webhook ที่ตั้งใน LINE Developers คือ:

```text
https://YOUR-DOMAIN/line/webhook
```

การเปิด `/health` ได้บน localhost ยังไม่ยืนยันว่า LINE เข้าถึง HTTPS ภายนอกได้ ต้องใช้ Verify ของ LINE ทดสอบด้วย

## 4. ตั้งค่า LINE

ใน channel ของ LINE Official Account:

1. กรอก Webhook URL ด้านบน แล้วกด Verify
2. เปิด Use webhook และ webhook redelivery
3. ปิดข้อความตอบกลับอัตโนมัติและข้อความทักทายเดิมที่ซ้ำกับบอท
4. เพิ่มเพื่อนบัญชีทดสอบ บอทจะเริ่มด้วยการขอ consent
5. ทดสอบยินยอม → แจ้งเหตุ → แชร์หมุด → ชื่อ/เบอร์ → ตรวจสรุป → ยืนยัน

ปุ่มยินยอม/ยืนยัน/แก้ไขเป็นปุ่มในการ์ด Flex สีเขียวและสีรอง ส่วนแชร์หมุดเป็น Location Quick Reply ของ LINE โดยตรง
ปุ่มยังมี case_id และ draft_rev เดิม จึงปฏิเสธปุ่มจากสรุปฉบับเก่าได้
ข้อความยาวจะกลับเป็นข้อความธรรมดา เพื่อรักษาคำแนะนำครบถ้วน ไม่มีการย่อเนื้อหาทางการแพทย์ใน renderer
ยังมี Quick Reply กำกับการ์ดเพื่อให้กดตอบได้สะดวก

## 5. ตรวจข้อมูลหลังยืนยัน

Pipeline อ่าน `pipeline_confirmed_cases_v4` ใน PostgreSQL ได้เฉพาะ Confirmed Version เท่านั้น
`incident_priority` มีเพียง level และ reason; audit ของกฎอยู่ใน case_events
Draft, ชื่อ และเบอร์โทรไม่อยู่ใน view นี้ ข้อมูลติดต่อแยกใน `pipeline_case_contacts`
ถ้าใช้ role เดิม ให้ผู้ดูแล DB เพิ่มสิทธิ์อ่าน view v4 ให้ role ที่โครงการกำหนด

## ขอบเขตที่ยังต้องตรวจบนระบบจริง

ชุดทดสอบใช้ LINE transport จำลอง ไม่ได้ยืนยัน credential, DNS, ใบรับรอง HTTPS หรือหน้าตาบนอุปกรณ์จริง
ต้องลองบน LINE iOS/Android และตรวจพฤติกรรมเมื่อ LLM ตอบช้า ก่อนเปิดใช้งานในวงกว้าง
ปัจจุบันการประมวลผลยังใช้ lock เดียว จึงควรทดสอบโหลดตามจำนวนผู้ใช้จริง ไม่ควรเพิ่มหลาย worker เพื่อแก้คิวโดยไม่จัดการ case concurrency

อ้างอิง: [Flex Messages](https://developers.line.biz/en/docs/messaging-api/using-flex-messages/), [Quick Reply](https://developers.line.biz/en/docs/messaging-api/using-quick-reply/), [Webhook](https://developers.line.biz/en/docs/messaging-api/receiving-messages/)
