"""제안서 초안 마크다운 → PDF 조판 (검토용).

## 목적

제출물은 hwpx 이지 PDF 가 아니다 (`docs/memory/competition-overview.md`).
이 스크립트는 **내용 검토와 분량 확인**을 위한 것이다. 10쪽 제한을 지키는지
실제 조판으로 확인해야 감축 대상을 정할 수 있다.

## 서식

대회 양식 규정(`docs/memory/proposal-template.md`)에 맞춘다:
  `<요약>` 13pt / 줄간격 130%
  `□` `◦` `-`  13pt / 줄간격 160%
  `※`          11pt / 줄간격 130%
A4, 여백은 한글 기본값에 준한다.

폰트는 Pretendard (OFL, `assets/fonts/`). 제출본은 함초롬돋움이므로 **여기서 나온 쪽수는 근사치**다.
자간·장평이 달라 실제 hwpx 쪽수와 차이가 날 수 있다.

## 지원 문법

마크다운 전체를 지원하지 않는다. 초안이 쓰는 것만 처리한다:
  `## `/`### ` 제목, `|` 표, ```` ``` ```` 코드블록(도식), `□◦-※` 글머리,
  `**굵게**`, `` `코드` ``, `> ` 인용, `---` 구분선, `- [ ]` 체크박스

실행: .venv/bin/python src/tools/render_pdf.py [입력.md] [출력.pdf]
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Flowable,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

# Pretendard (SIL Open Font License 1.1) — assets/fonts/LICENSE.txt 참조.
# ※ PretendardStd(서브셋) TTF 는 reportlab 에서 한글이 전부 .notdef 로 나온다. 전체판을 쓸 것.
FONT_DIR = Path("assets/fonts")
DEFAULT_IN = Path("docs/proposal-draft-v1.md")

# 2026-08-04 고정 공지: 표지·목차·<요약>·기타 참고자료를 **제외한 제안 내용 본문만** 산정.
# 게시판 답변(8/4)으로 항목 5(연구윤리)·7(API/GPU)도 분량 미포함 확정.
# → 채점 분량 = 항목 1·2·3·4·6·8 합계 ≤ 10쪽. (구 가정 "표지·목차 포함, 본문 8쪽"은 폐기)
PAGE_LIMIT = 10
BODY_LIMIT = PAGE_LIMIT
DEFAULT_OUT = Path("docs/proposal-draft-v1.pdf")

INK = colors.HexColor("#1A1D21")
MUTED = colors.HexColor("#5A6270")
RULE = colors.HexColor("#D4D8DE")
BAND = colors.HexColor("#EEF1F4")
ACCENT = colors.HexColor("#1F4E79")  # 공문서 계열 남색


def register_fonts() -> bool:
    """Pretendard 등록. 없으면 내장 폰트로 폴백하고 경고한다."""
    try:
        for weight, name in (
            ("Regular", "KR"), ("Medium", "KR-M"), ("SemiBold", "KR-SB"), ("Bold", "KR-B"),
        ):
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f"Pretendard-{weight}.ttf")))
        pdfmetrics.registerFontFamily("KR", normal="KR", bold="KR-B", italic="KR", boldItalic="KR-B")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Pretendard 등록 실패 ({e}). Helvetica 로 폴백 — 한글이 깨집니다.", file=sys.stderr)
        return False


def styles(ok: bool) -> dict[str, ParagraphStyle]:
    base, bold, semi = ("KR", "KR-B", "KR-SB") if ok else ("Helvetica",) * 3

    def S(name, size, leading_pct, **kw):
        kw.setdefault("textColor", INK)
        return ParagraphStyle(
            name, fontName=base, fontSize=size, leading=size * leading_pct / 100,
            alignment=TA_LEFT, **kw,
        )

    return {
        # 양식 규정: 본문 13pt/160%, ※ 11pt/130%, 요약 13pt/130%
        "body": S("body", 13, 160, spaceAfter=3),
        "summary": S("summary", 13, 130, spaceAfter=5),
        "note": S("note", 11, 130, textColor=MUTED, spaceAfter=3, leftIndent=4 * mm),
        "quote": S("quote", 12, 145, textColor=MUTED, leftIndent=6 * mm,
                   rightIndent=4 * mm, spaceBefore=3, spaceAfter=3,
                   borderPadding=(0, 0, 0, 4)),
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=16, leading=22, textColor=ACCENT,
                             spaceBefore=10, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName=semi, fontSize=13.5, leading=19, textColor=INK,
                             spaceBefore=7, spaceAfter=3),
        # 표 안 텍스트·도표 캡션도 13pt/130% 적용 (2026-07-31 주최측 공식 답변).
        # 이전에는 9.5pt/137% 로 조판해 쪽수를 과소평가하고 있었다.
        "cell": ParagraphStyle("cell", fontName=base, fontSize=13, leading=13 * 1.3,
                               textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName=semi, fontSize=13, leading=13 * 1.3,
                                textColor=INK),
        "caption": ParagraphStyle("caption", fontName=base, fontSize=13, leading=13 * 1.3,
                                  textColor=INK),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=8.6,
                               textColor=MUTED),
    }


def inline(md: str) -> str:
    """마크다운 인라인 → reportlab 마크업. 이스케이프를 먼저 하고 태그를 넣는다."""
    t = html.escape(md, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", t)
    # 코드 스팬: ASCII 만 있으면 Courier, 한글이 섞이면 본문 폰트를 유지한다.
    # (Courier 에는 한글 글리프가 없어 `프로토콜` 같은 코드 스팬이 통째로 두부가 됐다)
    def code_span(m: re.Match) -> str:
        inner = m.group(1)
        if inner.isascii():
            return f'<font face="Courier" size="10.5">{inner}</font>'
        return f'<font color="#5A6270">{inner}</font>'

    t = re.sub(r"`(.+?)`", code_span, t)
    t = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", t)
    return t


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_table(rows: list[list[str]], st: dict, avail: float) -> Table:
    head, body = rows[0], rows[1:]
    ncol = len(head)
    data = [[Paragraph(inline(c), st["cellh"]) for c in head]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]

    # 열 폭: 헤더+본문 문자수 비례, 최소폭 보장
    weights = []
    for i in range(ncol):
        m = max((len(r[i]) for r in rows if i < len(r)), default=6)
        weights.append(max(m, 5))
    total = sum(weights)
    widths = [max(avail * w / total, 16 * mm) for w in weights]
    if sum(widths) > avail:  # 최소폭 때문에 넘치면 비례 축소
        k = avail / sum(widths)
        widths = [w * k for w in widths]

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


class Architecture(Flowable):
    """아키텍처 도식 — ASCII 아트 대신 벡터로 그린다.

    ASCII 도식은 Courier 에 한글·박스문자 글리프가 없어 통째로 두부가 됐다.
    제안서 항목 2(설계 독창성 20점)의 핵심 시각물이므로 실제 도형으로 그린다.
    """

    BOX_H = 11 * mm
    GAP = 6.5 * mm

    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.rows = 8
        self.height = self.rows * self.BOX_H + (self.rows - 1) * self.GAP + 4 * mm

    def _box(self, c, x, y, w, h, title, sub, *, accent=False, dashed=False):
        c.saveState()
        c.setLineWidth(1.1 if accent else 0.6)
        c.setStrokeColor(ACCENT if accent else colors.HexColor("#98A2B0"))
        c.setFillColor(colors.HexColor("#E8EFF6") if accent else colors.white)
        if dashed:
            c.setDash(2, 2)
        c.roundRect(x, y, w, h, 2 * mm, stroke=1, fill=1)
        c.setDash()
        c.setFillColor(ACCENT if accent else INK)
        c.setFont("KR-SB", 8.6)
        c.drawCentredString(x + w / 2, y + h - 6.2 * mm, title)
        if sub:
            c.setFillColor(MUTED)
            c.setFont("KR", 7.2)
            c.drawCentredString(x + w / 2, y + h - 10 * mm, sub)
        c.restoreState()

    def _arrow(self, c, x1, y1, x2, y2, label=None, color=None):
        c.saveState()
        c.setStrokeColor(color or colors.HexColor("#7C8899"))
        c.setFillColor(color or colors.HexColor("#7C8899"))
        c.setLineWidth(0.8)
        c.line(x1, y1, x2, y2)
        # 화살촉 (수직/수평만 쓴다)
        s = 1.6 * mm
        if abs(x1 - x2) < 0.1:
            d = -1 if y2 < y1 else 1
            c.setLineWidth(0)
            p = c.beginPath()
            p.moveTo(x2, y2)
            p.lineTo(x2 - s * 0.7, y2 - d * s)
            p.lineTo(x2 + s * 0.7, y2 - d * s)
            p.close()
            c.drawPath(p, fill=1, stroke=0)
        if label:
            c.setFont("KR", 6.6)
            c.setFillColor(MUTED)
            c.drawString(x1 + 1.6 * mm, (y1 + y2) / 2 - 1 * mm, label)
        c.restoreState()

    def draw(self):
        c = self.canv
        W = self.width
        cx = W / 2
        H = self.BOX_H
        G = self.GAP
        wide = 78 * mm
        y = self.height - H - 2 * mm

        def row_y(i):
            return self.height - 2 * mm - (i + 1) * H - i * G

        # 0 입력
        y0 = row_y(0)
        self._box(c, cx - 62 * mm / 2, y0, 62 * mm, H,
                  "v0 스냅샷 — 등록 시점 원본", "결과 정보 0. 블라인드 입력", dashed=True)

        # 1 Planner
        y1 = row_y(1)
        self._box(c, cx - wide / 2, y1, wide, H,
                  "A1  Planner — 감사 계획 수립", "적용 불가 감사는 이유와 함께 기각")
        self._arrow(c, cx, y0, cx, y1 + H)

        # 2 병렬 3개
        y2 = row_y(2)
        # 왼쪽에 재감사 루프가 지나갈 통로(LOOP_L)를 비워 둔다 — 안 그러면 A2 박스를 가로지른다
        left = 13 * mm
        bw = (W - left - 2 * mm - 2 * 5 * mm) / 3
        xs = [left, left + bw + 5 * mm, left + 2 * (bw + 5 * mm)]
        self._box(c, xs[0], y2, bw, H, "A2  계량 감사", "LLM 0회 · 비용 $0")
        self._box(c, xs[1], y2, bw, H, "A3  전례 검색기", "시간 컷오프 강제")
        self._box(c, xs[2], y2, bw, H, "A7  스코프 선언", "기권 범위 명시")
        self._arrow(c, cx, y1, cx, y2 + H + 2.5 * mm)
        c.setStrokeColor(colors.HexColor("#7C8899"))
        c.setLineWidth(0.8)
        c.line(xs[0] + bw / 2, y2 + H + 2.5 * mm, xs[2] + bw / 2, y2 + H + 2.5 * mm)
        for x in xs:
            self._arrow(c, x + bw / 2, y2 + H + 2.5 * mm, x + bw / 2, y2 + H)

        # 3 레드팀
        y3 = row_y(3)
        self._box(c, cx - wide / 2, y3, wide, H,
                  "A4  레드팀 — 근거 결박 공격", "인용 없는 공격 금지")
        for x in xs[:2]:
            self._arrow(c, x + bw / 2, y2, x + bw / 2, y3 + H + 2.5 * mm)
        c.line(xs[0] + bw / 2, y3 + H + 2.5 * mm, xs[1] + bw / 2, y3 + H + 2.5 * mm)
        self._arrow(c, cx, y3 + H + 2.5 * mm, cx, y3 + H)

        # 4 Referee — 강조
        y4 = row_y(4)
        self._box(c, cx - wide / 2, y4, wide, H,
                  "A5  Referee — 인용 검증 게이트", "★ 검증의 대부분이 결정론적 코드", accent=True)
        self._arrow(c, cx, y3, cx, y4 + H)

        # 기각 반송 루프 (오른쪽)
        rx = cx + wide / 2 + 7 * mm
        c.saveState()
        c.setStrokeColor(ACCENT)
        c.setLineWidth(0.8)
        c.setDash(2.5, 2)
        c.line(cx + wide / 2, y4 + H / 2, rx, y4 + H / 2)
        c.line(rx, y4 + H / 2, rx, y3 + H / 2)
        c.setDash()
        c.restoreState()
        self._arrow(c, rx, y3 + H / 2, cx + wide / 2, y3 + H / 2, color=ACCENT)
        c.setFont("KR", 6.6)
        c.setFillColor(ACCENT)
        c.drawString(rx + 1.5 * mm, (y3 + y4) / 2 + H / 2, "기각 → 반송")

        # 5 Reviser
        y5 = row_y(5)
        self._box(c, cx - wide / 2, y5, wide, H, "A6  Reviser — 교정안 생성", "통과한 지적만 입력")
        self._arrow(c, cx, y4, cx, y5 + H, label="통과분")

        # 6 재감사
        y6 = row_y(6)
        self._box(c, cx - wide / 2, y6, wide, H,
                  "A2 재실행 — 리스크 델타 측정", "악화 시 회귀를 스스로 보고")
        self._arrow(c, cx, y5, cx, y6 + H)

        # 재감사 루프 (왼쪽)
        lx = 5 * mm
        c.saveState()
        c.setStrokeColor(colors.HexColor("#7C8899"))
        c.setLineWidth(0.8)
        c.setDash(2.5, 2)
        c.line(cx - wide / 2, y6 + H / 2, lx, y6 + H / 2)
        c.line(lx, y6 + H / 2, lx, y2 + H / 2)
        c.setDash()
        c.restoreState()
        self._arrow(c, lx, y2 + H / 2, xs[0], y2 + H / 2)
        c.setFont("KR", 6.6)
        c.setFillColor(MUTED)
        c.saveState()
        c.translate(lx - 1.2 * mm, (y2 + y6) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "교정 재감사 루프 — 델타 측정")
        c.restoreState()

        # 7 리포트
        y7 = row_y(7)
        self._box(c, cx - 74 * mm / 2, y7, 74 * mm, H,
                  "최종 리포트", "모든 주장에 인용 부착 + 기권 범위 명시", dashed=True)
        self._arrow(c, cx, y6, cx, y7 + H)


# 블록의 시작을 알리는 표지. 이 중 무엇으로도 시작하지 않는 줄은 **앞 줄의 연속**이다.
BLOCK_START = re.compile(
    r"^(\s*$|```|\||#{1,4}\s|>\s|-{3,}$|[□◦※]|-\s\[[ x]\]\s|-\s|\d+\.\s)"
)


def unwrap(md: str) -> list[str]:
    """마크다운의 접힌 줄을 하나로 합친다.

    초안은 가독성을 위해 한 문단을 여러 줄로 접어 썼는데, 줄 단위로 파싱하면
    줄을 넘어가는 `**굵게**` 가 깨진다(실제로 요약 첫 문단이 깨졌다).
    코드블록 안은 원문 그대로 둔다 — 도식이 무너지기 때문이다.
    """
    out: list[str] = []
    in_code = False
    for ln in md.split("\n"):
        if ln.strip().startswith("```"):
            in_code = not in_code
            out.append(ln)
            continue
        if in_code:
            out.append(ln)
            continue
        if out and not BLOCK_START.match(ln) and out[-1].strip() and not out[-1].strip().startswith("|"):
            out[-1] = out[-1].rstrip() + " " + ln.strip()
        else:
            out.append(ln)
    return out


def parse(md: str, st: dict, avail: float) -> list:
    flow: list = []
    lines = unwrap(md)
    i = 0
    in_summary = False

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        # 상단 주석 블록(> **작성 규칙** …)은 제출물이 아니므로 건너뛴다
        if s.startswith("> **작성 규칙**"):
            while i < len(lines) and lines[i].strip().startswith(">"):
                i += 1
            continue

        if not s:
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            block = "\n".join(buf)
            flow.append(Spacer(1, 3))
            if "A5 Referee" in block or "A1 Planner" in block:
                flow.append(Architecture(avail))          # ASCII 도식 → 벡터 도식
            else:
                flow.append(Preformatted(block, st["code"]))
            flow.append(Spacer(1, 4))
            continue

        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = [split_row(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            flow.append(Spacer(1, 3))
            flow.append(build_table(rows, st, avail))
            flow.append(Spacer(1, 5))
            continue

        if re.match(r"^-{3,}$", s):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if s.startswith("## "):
            title = s[3:].strip().strip("`")
            in_summary = "요약" in title
            flow.append(KeepTogether([Paragraph(inline(title), st["h1"])]))
            i += 1
            continue

        if s.startswith("### "):
            flow.append(KeepTogether([Paragraph(inline(s[4:].strip()), st["h2"])]))
            i += 1
            continue

        if s.startswith("# "):
            i += 1
            continue

        if s.startswith("> "):
            flow.append(Paragraph(inline(s[2:]), st["quote"]))
            i += 1
            continue

        # 글머리 — 양식의 계층 기호를 그대로 살린다
        if s.startswith("※"):
            flow.append(Paragraph(inline(s), st["note"]))
            i += 1
            continue
        if s.startswith(("□", "◦")):
            body = st["summary"] if in_summary else st["body"]
            style = ParagraphStyle(f"b{i}", parent=body, leftIndent=5 * mm, firstLineIndent=-5 * mm)
            flow.append(Paragraph(inline(s), style))
            i += 1
            continue
        if re.match(r"^- \[[ x]\] ", s):
            mark = "☑" if s[3] == "x" else "☐"
            style = ParagraphStyle(f"c{i}", parent=st["body"], leftIndent=6 * mm,
                                   firstLineIndent=-6 * mm)
            flow.append(Paragraph(f"{mark} " + inline(s[6:]), style))
            i += 1
            continue
        if s.startswith("- "):
            style = ParagraphStyle(f"d{i}", parent=st["body"], leftIndent=10 * mm,
                                   firstLineIndent=-5 * mm)
            flow.append(Paragraph("– " + inline(s[2:]), style))
            i += 1
            continue

        flow.append(Paragraph(inline(s), st["summary"] if in_summary else st["body"]))
        i += 1

    return flow


def make_page_decorator(title: str):
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("KR", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 12 * mm, title)
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, f"- {doc.page} -")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, 15 * mm, A4[0] - doc.rightMargin, 15 * mm)
        canvas.restoreState()
    return on_page


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    if not src.exists():
        raise SystemExit(f"입력 파일이 없습니다: {src}")

    ok = register_fonts()
    st = styles(ok)

    lm, rm, tm, bm = 20 * mm, 20 * mm, 20 * mm, 20 * mm
    avail = A4[0] - lm - rm

    doc = BaseDocTemplate(
        str(dst), pagesize=A4,
        leftMargin=lm, rightMargin=rm, topMargin=tm, bottomMargin=bm,
        title="제4회 JUMP AI 경진대회 제안서 초안", author="",
    )
    frame = Frame(lm, bm, avail, A4[1] - tm - bm, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="p", frames=[frame],
                     onPage=make_page_decorator("제4회 JUMP AI 신약개발 경진대회 — 제안서 초안 v1"))
    ])

    doc.build(parse(src.read_text(), st, avail))

    n = len(__import__("pypdf").PdfReader(str(dst)).pages)
    print(f"{dst} 생성 — 총 {n}쪽 (A4, Pretendard)")
    if n > BODY_LIMIT:
        print(f"⚠ 본문 한도 {BODY_LIMIT}쪽을 {n - BODY_LIMIT}쪽 초과합니다.")
    print(f"※ 분량 산정 = 항목 1·2·3·4·6·8 본문만 ≤ {PAGE_LIMIT}쪽. 표지·목차·요약·항목5·7·참고자료 제외 (2026-08-04 공지).")
    print("※ 입력 md에 제외 구역이 섞여 있으면 이 쪽수는 채점 분량보다 크게 나온다.")
    print("※ 제출본은 hwpx·함초롬돋움이므로 실제 쪽수는 이 값과 다를 수 있습니다.")


if __name__ == "__main__":
    main()
