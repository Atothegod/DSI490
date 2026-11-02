-- สิทธิ์ของ Pipeline (หัวข้อ 7, 12) — รันหลัง Agent 1 สร้างตารางแล้ว
-- Pipeline อ่านได้เฉพาะ Confirmed Version ผ่าน View ไม่เห็น Draft, Raw Message หรือ Risk Facts
-- ชื่อ นามสกุล เบอร์ อ่านได้เฉพาะ Role ที่ต้องติดต่อผู้แจ้ง

CREATE ROLE pipeline_reader NOLOGIN;
GRANT SELECT ON pipeline_confirmed_cases TO pipeline_reader;
GRANT SELECT (case_id, event, payload, created_at) ON case_events TO pipeline_reader;

CREATE ROLE pipeline_contact_reader NOLOGIN;
GRANT pipeline_reader TO pipeline_contact_reader;
GRANT SELECT ON pipeline_case_contacts TO pipeline_contact_reader;

-- ตัวอย่างผู้ใช้จริง (เปลี่ยนรหัสผ่าน)
-- CREATE ROLE pipeline LOGIN PASSWORD 'change-me' IN ROLE pipeline_reader;
-- CREATE ROLE contact_officer LOGIN PASSWORD 'change-me' IN ROLE pipeline_contact_reader;
