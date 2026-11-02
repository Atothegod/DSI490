"""Privacy Filter — ชุดทดสอบ PII ภาษาไทย (หัวข้อ 7) เพิ่มเคสที่พบจริงต่อได้"""

import pytest

from app.privacy import ADDRESS, NAME, PHONE, contains_pii, filter_text


def test_t8_name_phone_address_removed():
    r = filter_text("ผมชื่อสมชาย เบอร์ 0812345678 บ้านเลขที่ 12 น้ำท่วมบ้าน")
    assert "สมชาย" not in r.text and "0812345678" not in r.text and "12" not in r.text
    assert NAME in r.text and PHONE in r.text and ADDRESS in r.text
    assert r.first_names == ["สมชาย"] and r.phones == ["0812345678"] and r.address_masked


@pytest.mark.parametrize("text, phone", [
    ("โทร 08 1234 5678", "0812345678"),
    ("081-234-5678", "0812345678"),
    ("0 8 1 2 3 4 5 6 7 8", "0812345678"),
    ("๐๘๑๒๓๔๕๖๗๘", "0812345678"),
    ("+66 81 234 5678", "0812345678"),
    ("ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปด", "0812345678"),
    ("เบอร์บ้าน 036-123456", "036123456"),
])
def test_phone_variants(text, phone):
    r = filter_text(text)
    assert r.phones == [phone] and PHONE in r.text


def test_hotlines_are_not_pii():
    r = filter_text("โทร 1669 กับ 1784 ไม่ติด")
    assert r.phones == [] and "1669" in r.text


@pytest.mark.parametrize("text, first, last", [
    ("ชื่อเล่นแดง", ["แดง"], []),
    ("นายสมชาย นามสกุลใจดี", ["สมชาย"], ["ใจดี"]),
    ("ดิฉันชื่อ มาลี ค่ะ", ["มาลี"], []),
])
def test_names(text, first, last):
    r = filter_text(text)
    assert r.first_names == first and r.last_names == last
    for n in first + last:
        assert n not in r.text


@pytest.mark.parametrize("text", ["อยู่หมู่ 3 ซอยวัดใหม่", "ม.5 ถนนสายเอเชีย", "เลขที่ 45/2"])
def test_informal_addresses(text):
    r = filter_text(text)
    assert r.address_masked and ADDRESS in r.text


def test_contains_pii():
    assert contains_pii("ติดต่อ 0812345678") and not contains_pii("โทร 1669")
