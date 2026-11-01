"""Notebook chat surface for the actual FloodAgent; no model or intake rules here."""
from html import escape
import json
from time import perf_counter

import ipywidgets as widgets
from IPython.display import display
from app.safety import redact_identifiers, redact_phones


class NotebookChat:
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        self.agent = agent_factory()
        self.last_output = None
        self.messages = []
        self.busy = False
        self.transcript = widgets.HTML()
        self.status = widgets.HTML("พร้อมพิมพ์แจ้งเหตุสมมติได้เลยค่ะ")
        self.message = widgets.Textarea(placeholder="พิมพ์ข้อความ เช่น น้ำท่วมบ้านค่ะ", layout=widgets.Layout(width="100%", height="75px"))
        self.send = widgets.Button(description="ส่งข้อความ", button_style="success", icon="paper-plane")
        self.reset = widgets.Button(description="เริ่มแชตใหม่", icon="refresh")
        self.inspect = widgets.Button(description="ดูข้อมูลที่ระบบเก็บ", icon="search")
        self.quick_replies = widgets.HBox(layout=widgets.Layout(flex_flow="row wrap"))
        self.latitude = widgets.FloatText(value=14.9512, description="ละติจูด:")
        self.longitude = widgets.FloatText(value=100.3646, description="ลองจิจูด:")
        self.share = widgets.Button(description="ส่งพิกัดสมมติ", icon="map-marker")
        self.location_panel = widgets.VBox([
            widgets.HTML("<small>พิกัดสำหรับทดลอง กรอกจุดเกิดเหตุเอง ไม่ใช่ GPS ของเครื่อง</small>"),
            self.latitude, self.longitude, self.share,
        ], layout=widgets.Layout(display="none"))
        self.debug = widgets.Output()
        self.debug_panel = widgets.Accordion(children=[self.debug], selected_index=None)
        self.debug_panel.set_title(0, "State และรายงานที่เตรียมส่ง")
        self.ui = widgets.VBox([
            widgets.HTML("<b>ทดลองแจ้งเหตุ · Gemini จริง</b><br><small>พิมพ์คุยต่อเนื่องได้ ใช้ข้อมูลสมมติ · ยังไม่เชื่อม LINE</small>"),
            self.transcript, self.quick_replies, self.location_panel, self.message,
            widgets.HBox([self.send, self.reset, self.inspect]), self.status, self.debug_panel,
        ], layout=widgets.Layout(width="100%", max_width="800px"))
        self.send.on_click(self._send_text)
        self.reset.on_click(self._reset)
        self.inspect.on_click(self._inspect)
        self.share.on_click(self._share_location)
        self._draw()

    def display(self):
        display(self.ui)
        return self

    def _draw(self):
        bubbles = []
        for role, text in self.messages:
            mine = role == "user"
            label = "คุณ" if mine else "ผู้ช่วย"
            color = "#c8f5ba" if mine else "#fff"
            side = "flex-end" if mine else "flex-start"
            safe = escape(text).replace("\n", "<br>")
            bubbles.append(f'<div style="display:flex;justify-content:{side};margin:10px 0"><div style="background:{color};color:#17202a;padding:12px 16px;border-radius:14px;max-width:85%;overflow-wrap:anywhere"><small>{label}</small><br>{safe}</div></div>')
        body = "".join(bubbles) or '<span style="color:#536270">เริ่มพิมพ์ข้อความด้านล่างได้เลย</span>'
        self.transcript.value = f'<div style="background:#edf3f8;padding:16px;border-radius:12px;max-height:500px;overflow-y:auto">{body}</div>'

    def _set_busy(self, busy):
        self.busy = busy
        for control in (self.message, self.send, self.reset, self.inspect, self.share, self.latitude, self.longitude, *self.quick_replies.children):
            control.disabled = busy

    def _send_text(self, _=None):
        text = self.message.value.strip()
        if not text or self.busy:
            return
        self.message.value = ""
        self.submit(text, input_text=text)

    def submit(self, label, **kwargs):
        if self.busy:
            return
        self._set_busy(True)
        self.quick_replies.children = ()
        self.location_panel.layout.display = "none"
        # Even a notebook preview must not retain phone numbers in its transcript.
        self.messages.append(("user", redact_phones(redact_identifiers(label)).text))
        self._draw()
        self.status.value = "กำลังประมวลผล…"
        started = perf_counter()
        try:
            output = self.agent(**kwargs)
            self.last_output = output
            self.messages.append(("assistant", output.output_text))
            self._draw()
            buttons = []
            for reply in output.quick_replies:
                button = widgets.Button(description=reply.label)
                if reply.type == "location":
                    button.on_click(lambda _: self._show_location())
                else:
                    button.on_click(lambda _, data=reply.data, label=reply.label: self.submit(label, postback=data))
                buttons.append(button)
            self.quick_replies.children = tuple(buttons)
            self.status.value = f"ตอบแล้ว · {perf_counter() - started:.2f} วินาที"
        except Exception as exc:
            # Provider errors can contain request URLs or credentials; don't print raw exceptions.
            error_name = type(exc).__name__
            if error_name == "LMServerError":
                hint = "บริการโมเดลตอบข้อผิดพลาดฝั่งเซิร์ฟเวอร์ รอสักครู่แล้วลองส่งอีกครั้ง"
            elif "RateLimit" in error_name:
                hint = "ติดโควตาหรือจำนวนคำขอ กรุณาตรวจ Usage ของ Gemini"
            elif error_name == "LMUnsupportedModelError":
                hint = "โมเดลหรือไลบรารีที่ใช้อยู่ไม่รองรับคำขอนี้"
            else:
                hint = "คำขอไม่สำเร็จ ต้องตรวจการตั้งค่าหรือรายละเอียดผู้ให้บริการเพิ่มเติม"
            # UI-only notice: never add diagnostic text to the agent's model context.
            self.messages.append(("assistant", f"ส่งข้อความรอบนี้ไม่สำเร็จ ({error_name}) กรุณาลองใหม่ค่ะ"))
            self._draw()
            self.message.value = kwargs.get("input_text", "")
            self.status.value = f"ประมวลผลไม่สำเร็จ ({escape(error_name)}) · {hint}"
        finally:
            self._set_busy(False)

    def _show_location(self):
        self.location_panel.layout.display = "flex"

    def _share_location(self, _=None):
        lat, lng = self.latitude.value, self.longitude.value
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            self.status.value = "พิกัดอยู่นอกช่วงที่ใช้ได้"
            return
        self.submit("[แชร์พิกัดสมมติ]", location=(lat, lng))

    def _reset(self, _=None):
        if self.busy:
            return
        old = self.agent
        self.agent = self.agent_factory()
        old.repository.close()
        self.last_output = None
        self.messages.clear()
        self.message.value = ""
        self.quick_replies.children = ()
        self.location_panel.layout.display = "none"
        self.debug.clear_output()
        self.debug_panel.selected_index = None
        self.status.value = "เริ่มแชตใหม่แล้ว พิมพ์เหตุการณ์สมมติได้เลยค่ะ"
        self._draw()

    def _inspect(self, _=None):
        state = self.agent.state
        records = self.agent.repository.outbox(self.agent.internal_user_id)
        with self.debug:
            self.debug.clear_output(wait=True)
            print("STATE (ไม่มีระเบียนเบอร์จริง):")
            print(state.model_dump_json(indent=2) if state else "ยังไม่มี incident")
            print("\nOUTBOX (เตรียมส่ง ยังไม่ได้เชื่อม Agent 2):")
            print(json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=2))
        self.debug_panel.selected_index = 0
