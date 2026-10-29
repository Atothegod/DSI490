"""
extraction.py — จุดเรียก Gemini จุดที่ 1: ดึงส่วนที่เปลี่ยน (delta) จากข้อความล่าสุด

ย้ายมาจาก flood_agent.ipynb ส่วน C (prompt ของทีม) + กฎที่เพิ่มในส่วน E
ตั้งแต่นี้ให้แก้ prompt ที่ไฟล์นี้ที่เดียว เพื่อให้ Notebook และ API ใช้ prompt เดียวกัน

ข้อความที่ส่งเข้ามาต้องปิดบังเบอร์แล้ว (safety.redact_phones) — Gemini ต้องไม่เห็นเบอร์โทร
"""

from __future__ import annotations

from typing import Iterable

import dspy

from .schemas import (
    ExtractionResult,
    IncidentFieldName,
    IncidentState,
    Intent,
    Need,
    SituationTag,
    VulnerableType,
    WaterDescriptor,
)


def _vals(e) -> str:
    return ", ".join(x.value for x in e)


# ---- prompt ของทีม (ส่วน C) ------------------------------------------------
EXTRACT_INSTRUCTIONS = f"""ดึงข้อมูลเหตุการณ์น้ำท่วมจากข้อความล่าสุดของประชาชนเท่านั้น
บันทึกเฉพาะข้อมูลที่ระบุไว้อย่างชัดเจน ห้ามเพิ่มข้อเท็จจริงหรือสรุปสถานการณ์ที่ข้อความไม่ได้ยืนยัน
ตัวอย่าง: หากข้อความระบุว่า "มีผู้สูงอายุ" ให้บันทึกเพียงว่ามีผู้สูงอายุ ห้ามระบุว่าผู้สูงอายุติดค้าง เว้นแต่ข้อความจะยืนยันว่าติดค้าง

intents (เลือกได้หลายค่า): {_vals(Intent)}

target:
- active_incident: ข้อความเกี่ยวกับเหตุที่กำลังคุยอยู่
- new_incident: แจ้งเหตุใหม่ หรือยังไม่มีเหตุที่กำลังคุย
- unclear: กล่าวถึงตำแหน่งที่ต่างจากเหตุที่กำลังคุย หรือไม่แน่ใจว่าเป็นเหตุไหน
- none: ไม่เกี่ยวกับเหตุการณ์ (เช่น ถามความรู้อย่างเดียว ทักทาย)

ไม่ว่า target จะเป็นอะไร ต้องดึง updates ทุก field ที่มีในข้อความ
ห้ามละข้อมูลเพียงเพราะยังไม่รู้ว่าเป็นเหตุไหน

updates: หนึ่งรายการต่อ field ที่ข้อความนี้ให้ข้อมูล รูปแบบ value ตาม field:
- location_area: ชื่อหมู่บ้าน ตำบล หรืออำเภอเท่านั้น เช่น "ตำบลทับยา อำเภออินทร์บุรี"
  ถ้าเป็นสถานที่ เช่น ถนน วัด โรงเรียน ตลาด ให้ใส่ location_landmark แทน
- location_landmark: ชื่อสถานที่หรือจุดสังเกต เช่น "หน้าวัดโพธิ์ชัย", "ถนนเข้าหมู่บ้าน"
- situation: {{"text": ข้อความเต็มของผู้ใช้ ไม่ตัดทอน, "tags": [{_vals(SituationTag)}]}}
- needs: list จาก [{_vals(Need)}] ใช้ info_only เมื่อผู้ใช้บอกว่าแค่แจ้งให้ทราบ
- affected_people: {{"present": true/false, "count": จำนวนหรือ null, "is_approximate": true/false}}
- vulnerable_people: {{"present": true/false, "groups": [{{"type": หนึ่งใน [{_vals(VulnerableType)}], "count": จำนวนหรือ null, "is_approximate": true/false}}]}}
- water_level: {{"raw": ข้อความดิบ, "descriptor": หนึ่งใน [{_vals(WaterDescriptor)}] หรือ null, "cm": ตัวเลขเมื่อผู้ใช้ให้ตัวเลขเท่านั้น หรือ null}}
  ห้ามแปลง "ถึงเอว" เป็นเซนติเมตรเอง

status ของแต่ละ update:
- known: ผู้ใช้ให้ข้อมูล (ต้องมี value)
- unknown: ผู้ใช้บอกว่าไม่ทราบ (value = null)
- declined: ผู้ใช้ไม่อยากบอก (value = null)
is_correction = true เมื่อผู้ใช้แก้คำตอบเดิม เช่น "แก้เป็น", "พิมพ์ผิด" (ต่างจากสถานการณ์เปลี่ยนจริง เช่น "น้ำขึ้นเป็นเมตรแล้ว")

ถ้ามี pending_question และข้อความเป็นคำตอบสั้น ๆ ให้ตีความว่าตอบ field ที่ถามค้างไว้

urgent_flags: ใส่เฉพาะเมื่อผู้ใช้บอกตรง ๆ ห้ามประเมินเอง
- people_trapped: มีคนติดอยู่ ออกไม่ได้
- injury_reported: มีคนบาดเจ็บ ป่วยฉุกเฉิน
- help_requested: ขอให้ช่วยด่วน เช่น "ช่วยด้วย"

mentioned_location: ตำแหน่งที่ข้อความกล่าวถึง (ถ้ามี)
question: คำถามความรู้ที่ต้องตอบ (ถ้ามี) เช่น "ต้องตัดไฟตอนไหน"
"""

# ---- กฎที่เพิ่มจากผลรันส่วน E ------------------------------------------------
# memory.py นับเป็น correction เมื่อ Gemini บอกว่าแก้ไข AND มีคำบอกแก้ไขในช่วงประโยคเดียวกับ raw
RAW_RULE = """
raw ของแต่ละ update (สำคัญ): คัดลอกคำจากข้อความผู้ใช้ส่วนที่ใช้ดึงค่านั้น แบบคำต่อคำ ห้ามแต่งหรือสรุปใหม่
ตัวอย่าง: ข้อความ "แก้เป็น 50 เซนติเมตรค่ะ พิมพ์ผิด" → water_level.raw = "50 เซนติเมตร"
"""

# needs ส่งเฉพาะเมื่อมีหลักฐาน ไม่เดา rescue จากการมีคนติดค้าง
# (conversation.reconcile_needs เอา info_only ที่ขัดกับคำขอช่วยเหลือออกเอง)
SITUATION_NEEDS_RULE = """
situation: ถ้าข้อความเล่าสถานการณ์ใหม่หรือเพิ่มเติม เช่น มีคนติดค้าง น้ำขึ้นเร็ว ถนนขาด ต้องส่ง situation ทุกครั้ง แม้มี situation เดิมแล้ว
needs: ส่งเฉพาะเมื่อข้อความระบุสิ่งที่ต้องการชัดเจน เช่น "ขอเรือ" "อยากอพยพ" "ขอน้ำดื่ม" หรือ "แค่อยากแจ้งไว้"
- ข้อความไม่ได้ระบุสิ่งที่ต้องการ → ไม่ต้องส่ง needs ห้ามเดา
- "ช่วยด้วย" หรือการมีคนติดค้าง ยังไม่ใช่หลักฐานว่าต้องการ rescue ให้ใส่ใน urgent_flags แทน
- ข้อความที่ขอความช่วยเหลือ ห้ามใส่ info_only
"""

INSTRUCTIONS = EXTRACT_INSTRUCTIONS + RAW_RULE + SITUATION_NEEDS_RULE


class ExtractIntentAndSlots(dspy.Signature):
    """(instructions ถูกแทนด้วย INSTRUCTIONS)"""

    history: dspy.History = dspy.InputField(desc="ข้อความล่าสุดก่อนหน้า")
    active_incident: str = dspy.InputField(desc="IncidentState ของเหตุที่กำลังคุย เป็น JSON หรือ 'ไม่มี'")
    pending_question: str = dspy.InputField(desc="field ที่ Agent ถามค้างไว้ หรือ 'ไม่มี'")
    message: str = dspy.InputField(desc="ข้อความล่าสุดของประชาชน")
    result: ExtractionResult = dspy.OutputField()


ExtractIntentAndSlots = ExtractIntentAndSlots.with_instructions(INSTRUCTIONS)


class Extractor(dspy.Module):
    """Memory อยู่ใน state + pending question จึงไม่ส่งประวัติแชตเข้า Gemini"""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExtractIntentAndSlots)

    def forward(
        self,
        message: str,
        state: IncidentState | None = None,
        pending: Iterable[IncidentFieldName] | None = None,
    ) -> ExtractionResult:
        pending = list(pending or [])
        return self.predict(
            history=dspy.History(messages=[]),
            # พิกัดไม่ต้องส่ง Gemini และเบอร์ไม่อยู่ใน state อยู่แล้ว (เก็บใน ContactRecord)
            active_incident=state.model_dump_json(exclude={"location_coordinates"}) if state else "ไม่มี",
            pending_question=", ".join(f.value for f in pending) if pending else "ไม่มี",
            message=message,
        ).result
