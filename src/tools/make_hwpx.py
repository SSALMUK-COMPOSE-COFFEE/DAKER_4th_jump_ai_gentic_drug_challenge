"""마크다운 초안 → 대회 규정 서식 hwpx 제안서.

## 왜 필요한가

예선 제출물은 **hwpx 1개**다(`docs/memory/competition-overview.md`). 초안은 마크다운으로
쓰고 있으므로 옮겨 담는 작업이 반드시 한 번 필요하다. 손으로 옮기면 서식 규정
(함초롬돋움 13pt·글머리별 줄간격 160%/130%)을 문단마다 지켜야 해서 오류가 나기 쉽다.

이 스크립트는 **주최측 배포 양식 파일 자체를 템플릿으로 삼는다.** 서식을 흉내내는 것이
아니라, 양식 안에 들어 있는 문단의 XML 을 그대로 복제해 텍스트만 갈아끼운다. 따라서
줄간격·글꼴·들여쓰기가 양식과 정의상 동일하다.

## 서식 대응 (양식 실측)

| 마크다운 | 문단 | 서식 |
|---|---|---|
| `□ 텍스트`   | paraPr 1  | 함초롬돋움 13pt / 160% |
| `◦ 텍스트`   | paraPr 1  | 〃 (한 칸 들여쓰기) |
| `- 텍스트`   | paraPr 1  | 〃 (세 칸 들여쓰기) |
| `※ 텍스트`   | paraPr 10 | 함초롬돋움 11pt / 130% |
| 그 외 문단   | paraPr 1  | 함초롬돋움 13pt / 160% (글머리 없음) |

양식의 견본 문단은 **파란색 기울임**(charPr 17/25)이라 그대로 쓰면 본문이 파랗게 나온다.
그래서 같은 글꼴·크기의 **검정 정체(正體)** charPr 을 header.xml 에 새로 추가해서 쓴다.

## 사용법

    .venv/bin/python src/tools/make_hwpx.py <입력.md> [출력.hwpx]
    .venv/bin/python src/tools/make_hwpx.py --check <파일.hwpx>   # 생성물 검증

입력 마크다운 규약:

    @@선택분야: 분야 3          ← 표지 표를 채운다 (없으면 양식 기본값 유지)
    @@팀명: 하진권씨
    @@에이전트명: PreMortem
    @@키워드국문: 가, 나, 다
    @@키워드영문: A, B, C

    ## <요약>                   ← 요약 항목
    □ 문제. ...

    ## 1. 신약개발에서의 ...     ← 번호 항목. 제목은 양식 문구를 쓴다
    □ ...
    ◦ ...
    - ...
    ※ ...

`**굵게**` 는 굵은 run 으로 변환한다. 표(`| a | b |`)는 hwpx 표로 변환한다.
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path

# 주최측 배포 양식. 이 파일의 문단 XML 을 복제해 서식을 승계한다.
FORM = Path(".orca/drops/★[붙임] AI_신약개발_경진대회_제안서 양식.hwpx")

# header.xml 에 새로 추가하는 charPr. 양식의 마지막 id 가 25 이므로 26 부터 쓴다.
CP_BODY, CP_BODY_B, CP_NOTE, CP_NOTE_B = 26, 27, 28, 29

# DACON 8항목 구성. 게시판 답변(2026-08-04)상 5·7번은 분량 산정에서 제외된다.
SECTIONS = [
    ("1", "1. 신약개발에서의 에이전트 활용의 필요성과 배경"),
    ("2", "2. 에이전트 설계 및 설계의 독창성과 창의성"),
    ("3", "3. 기술적 실현 가능성"),
    ("4", "4. 에이전트 평가 적절성"),
    ("5", "5. AI 활용 투명성 및 연구 윤리"),
    ("6", "6. 본선 시연 시나리오"),
    ("7", "7. API 및 GPU 관련 예상 소요 규모"),
    ("8", "8. 에이전트 도입에 따른 파급효과"),
]
SCORED = {"1", "2", "3", "4", "6", "8"}  # 분량 산정 대상


def resolve(path: Path) -> Path:
    """한글 파일명의 NFC/NFD 불일치를 흡수한다."""
    if path.exists():
        return path
    want = unicodedata.normalize("NFC", path.name)
    for cand in path.parent.iterdir():
        if unicodedata.normalize("NFC", cand.name) == want:
            return cand
    raise FileNotFoundError(path)


# ---------------------------------------------------------------- 마크다운 파싱

class Block:
    """문단 하나. kind 는 l1/l2/l3/note/plain/table/heading/summary."""

    def __init__(self, kind: str, text: str = "", rows: list[list[str]] | None = None):
        self.kind = kind
        self.text = text
        self.rows = rows or []


def clean_meta(v: str) -> str:
    """표지 값에서 마크다운 강조·작성 주석을 걷어낸다.

    초안 헤더 표가 `**하진권씨** (2026-08-03 확정)` 처럼 쓰여 있어서 그대로 넣으면
    제출본에 작업 메모가 실린다. `【사용자 기재】` 같은 자리표시자는 값 없음으로 본다.
    """
    if "【" in v:            # 【사용자 기재】 등 자리표시자 → 아직 값이 없는 칸
        return ""
    v = re.sub(r"[*`]", "", v)
    v = re.split(r"\s+[—–]\s+", v)[0]                        # em dash 뒤 작성 주석 제거
    v = re.sub(r"\((?:[^()]*(?:확정|기재|예정|미정|참고)[^()]*)\)", "", v)
    return re.sub(r"\s+", " ", v).strip(" -—·")


def parse_md(md: str) -> tuple[dict[str, str], dict[str, list[Block]]]:
    meta: dict[str, str] = {}
    docs: dict[str, list[Block]] = {}
    cur: list[Block] | str | None = None
    skipped: list[int] = []          # 건너뛴 코드블록(도식) 개수 집계용
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1

        # 코드블록(ASCII 아키텍처 도식)은 통째로 건너뛴다. hwpx 본문은 가변폭이라
        # 등폭 도식이 깨지고, 한글에서 그림·표로 다시 그려야 한다.
        if line.lstrip().startswith("```"):
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                i += 1
            i += 1
            skipped.append(len(docs))
            continue

        m = re.match(r"^@@(\S+?):\s*(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip()
            continue

        m = re.match(r"^##\s+(.*)$", line)
        if m:
            title = m.group(1).strip()
            key = None
            if "요약" in title:
                key = "요약"
            else:
                mm = re.match(r"^`?(\d)[.．]", title)
                if mm:
                    key = mm.group(1)
            if key:
                cur = docs.setdefault(key, [])
            elif "헤더" in title:
                cur = "HEADER"  # 표지 표 정보를 담은 구역
            else:
                cur = None  # 인식 못 한 제목 구역은 통째로 버린다
            continue

        # `## 헤더` 구역의 표에서 표지 값을 줍는다 (@@ 메타를 안 쓴 초안 지원)
        if cur == "HEADER":
            if line.lstrip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 2:
                    label = re.sub(r"[*`\s]", "", cells[0])
                    val = clean_meta(cells[1])
                    for want, k in [("선택분야", "선택분야"), ("팀명", "팀명"),
                                    ("에이전트명", "에이전트명"),
                                    ("키워드(국문)", "키워드국문"),
                                    ("키워드(영문)", "키워드영문")]:
                        if label == want and val and k not in meta:
                            meta[k] = val
            continue

        if cur is None or not line.strip():
            continue

        # 표: 연속된 | 줄을 모은다
        if line.lstrip().startswith("|"):
            rows = []
            j = i - 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                j += 1
            i = j
            if rows:
                cur.append(Block("table", rows=rows))
            continue

        s = line.strip()
        if s.startswith("□"):
            cur.append(Block("l1", s[1:].strip()))
        elif s.startswith("◦"):
            cur.append(Block("l2", s[1:].strip()))
        elif s.startswith("※"):
            cur.append(Block("note", s[1:].strip()))
        elif re.match(r"^-\s+", s) and not s.startswith("---"):
            cur.append(Block("l3", s[1:].strip()))
        elif s.startswith(("#", ">", "---", "```")):
            continue
        # 글머리 없는 줄은 **앞 문단의 이어짐**이다. 마크다운은 긴 문단을 여러 줄로
        # 접어 쓰는데, 줄마다 문단을 만들면 hwpx 에서 들여쓰기와 줄간격이 깨진다.
        elif cur and cur[-1].kind in ("l1", "l2", "l3", "note", "plain"):
            cur[-1].text = cur[-1].text.rstrip() + " " + s
        else:
            cur.append(Block("plain", s))
    if skipped:
        print(f"  ※ 코드블록(도식) {len(skipped)}개를 건너뛰었다 — 한글에서 그림·표로 넣을 것")
    return meta, docs


# ---------------------------------------------------------------- XML 조립

def esc(t: str) -> str:
    return html.escape(t, quote=False)


def runs(text: str, base: int, bold: int) -> str:
    """`**굵게**` 를 굵은 run 으로 쪼갠다."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            body, cp = part[2:-2], bold
        else:
            body, cp = part, base
        body = re.sub(r"[`*_]", "", body)
        if body:
            out.append(f'<hp:run charPrIDRef="{cp}"><hp:t>{esc(body)}</hp:t></hp:run>')
    return "".join(out) or f'<hp:run charPrIDRef="{base}"/>'


def top_paras(sec: str) -> list[tuple[int, int]]:
    """최상위 `<hp:p>` 의 (시작, 끝) 위치. 표 셀 안의 문단은 중첩이므로 건너뛴다.

    정규식 non-greedy 로 뽑으면 표를 담은 문단이 셀 문단의 닫는 태그에서 잘려
    XML 이 깨진다. 깊이를 세어 최상위만 고른다.
    """
    spans, depth, start = [], 0, 0
    for m in re.finditer(r"<hp:p\b[^>]*?(/?)>|</hp:p>", sec):
        if m.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                spans.append((start, m.end()))
        elif m.group(1) == "/":          # 자기닫힘 문단
            if depth == 0:
                spans.append((m.start(), m.end()))
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return spans


class Template:
    """양식에서 뽑은 문단 뼈대. run 부분만 갈아끼우고 나머지는 원본을 유지한다."""

    def __init__(self, xml: str):
        self.xml = xml

    def with_runs(self, runs_xml: str) -> str:
        """첫 run 부터 마지막 run 까지를 통째로 교체. linesegarray 등은 보존한다.

        빈 문단의 run 은 `<hp:run .../>` 자기닫힘이라 닫는 태그가 없다. 둘 다 받는다.
        """
        ms = list(re.finditer(r"<hp:run\b[^>]*/>|<hp:run\b[^>]*>.*?</hp:run>", self.xml, re.S))
        if not ms:  # run 이 아예 없으면 문단 여는 태그 뒤에 넣는다
            i = self.xml.index(">") + 1
            return self.xml[:i] + runs_xml + self.xml[i:]
        return self.xml[: ms[0].start()] + runs_xml + self.xml[ms[-1].end():]


def make_table(rows: list[list[str]], width: int = 47617) -> str:
    ncol = max(len(r) for r in rows)
    cw = width // ncol
    trs = []
    for ri, row in enumerate(rows):
        cells = []
        for ci in range(ncol):
            txt = row[ci] if ci < len(row) else ""
            bold = CP_BODY_B if ri == 0 else CP_BODY
            p = (f'<hp:p id="0" paraPrIDRef="1" styleIDRef="0" pageBreak="0" '
                 f'columnBreak="0" merged="0">{runs(txt, bold, CP_BODY_B)}</hp:p>')
            cells.append(
                f'<hp:tc name="" header="{1 if ri == 0 else 0}" hasMargin="0" protect="0" '
                f'editable="0" dirty="0" borderFillIDRef="9">'
                f'<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" '
                f'vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" '
                f'textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">{p}</hp:subList>'
                f'<hp:cellAddr colAddr="{ci}" rowAddr="{ri}"/>'
                f'<hp:cellSpan colSpan="1" rowSpan="1"/>'
                f'<hp:cellSz width="{cw}" height="2000"/>'
                f'<hp:cellMargin left="141" right="141" top="141" bottom="141"/></hp:tc>'
            )
        trs.append("<hp:tr>" + "".join(cells) + "</hp:tr>")
    tbl = (
        f'<hp:tbl id="0" zOrder="0" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" '
        f'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" pageBreak="CELL" '
        f'repeatHeader="1" rowCnt="{len(rows)}" colCnt="{ncol}" cellSpacing="0" '
        f'borderFillIDRef="3" noAdjust="0">'
        f'<hp:sz width="{width}" widthRelTo="ABSOLUTE" height="{2000 * len(rows)}" '
        f'heightRelTo="ABSOLUTE" protect="0"/>'
        f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" '
        f'holdAnchorAndSO="0" vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" '
        f'horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
        f'<hp:outMargin left="283" right="283" top="283" bottom="283"/>'
        f'<hp:inMargin left="141" right="141" top="141" bottom="141"/>'
        + "".join(trs) + "</hp:tbl>"
    )
    return (f'<hp:p id="0" paraPrIDRef="1" styleIDRef="0" pageBreak="0" columnBreak="0" '
            f'merged="0"><hp:run charPrIDRef="{CP_BODY}">{tbl}</hp:run></hp:p>')


# ---------------------------------------------------------------- header 확장

def extend_header(h: str) -> str:
    """본문용 검정 정체 charPr 4종을 추가한다 (양식 견본은 파란 기울임이라 못 쓴다)."""
    if f'<hh:charPr id="{CP_BODY}"' in h:
        return h

    def clone(src_id: int, new_id: int, bold: bool) -> str:
        src = re.search(rf'<hh:charPr\b[^>]*\bid="{src_id}"[^>]*?>.*?</hh:charPr>', h, re.S).group(0)
        x = src.replace(f'id="{src_id}"', f'id="{new_id}"', 1)
        x = x.replace('textColor="#0000FF"', 'textColor="#000000"')
        x = x.replace("<hh:italic/>", "")           # 견본의 기울임 제거
        if bold:                                     # HWPML 자식 순서: offset → bold → italic
            x = x.replace("<hh:underline", "<hh:bold/><hh:underline", 1)
        return x

    added = "".join([
        clone(17, CP_BODY, False), clone(17, CP_BODY_B, True),
        clone(25, CP_NOTE, False), clone(25, CP_NOTE_B, True),
    ])
    m = re.search(r'<hh:charProperties itemCnt="(\d+)">', h)
    cnt = int(m.group(1))
    h = h.replace(m.group(0), f'<hh:charProperties itemCnt="{cnt + 4}">', 1)
    return h.replace("</hh:charProperties>", added + "</hh:charProperties>", 1)


# ---------------------------------------------------------------- 표지 표 채우기

def fill_cover(sec: str, meta: dict[str, str]) -> str:
    """표지 표의 값 칸을 채운다. 라벨 셀 바로 다음 셀의 문단 텍스트를 교체한다."""
    # 선택분야: 고른 분야 칸에 ✔ 를 붙이고 굵게 (양식은 분야1~4 를 나열만 해 둔다)
    field = meta.get("선택분야", "")
    fm = re.search(r"분야\s*([1-4])", field)
    if fm:
        want = f"분야{fm.group(1)}"
        for m in re.finditer(r"<hp:p\b(?:(?!</hp:p>).)*?</hp:p>", sec, re.S):
            if re.sub(r"<[^>]+>", "", m.group(0)).strip() == want:
                newp = Template(m.group(0)).with_runs(
                    f'<hp:run charPrIDRef="{CP_BODY_B}"><hp:t>✔ {want}</hp:t></hp:run>')
                sec = sec[: m.start()] + newp + sec[m.end():]
                break

    pairs = [("팀명", "팀 명"), ("에이전트명", "에이전트명")]
    for key, label in pairs:
        val = meta.get(key)
        if not val:
            continue
        # 라벨이 들어 있는 tc 이후 첫 tc 의 첫 문단을 채운다
        m = re.search(rf'<hp:tc\b(?:(?!</hp:tc>).)*?<hp:t>{re.escape(label)}</hp:t>.*?</hp:tc>',
                      sec, re.S)
        if not m:
            continue
        rest = sec[m.end():]
        nxt = re.search(r'<hp:tc\b.*?</hp:tc>', rest, re.S)
        if not nxt:
            continue
        cell = nxt.group(0)
        pm = re.search(r'<hp:p\b.*?</hp:p>', cell, re.S)
        if not pm:
            continue
        newp = Template(pm.group(0)).with_runs(runs(val, CP_BODY, CP_BODY_B))
        sec = sec[:m.end()] + rest[:nxt.start()] + cell.replace(pm.group(0), newp, 1) + rest[nxt.end():]

    for key, label in [("키워드국문", "국문"), ("키워드영문", "영문")]:
        val = meta.get(key)
        if not val:
            continue
        m = re.search(rf'<hp:tc\b(?:(?!</hp:tc>).)*?<hp:t>{label}</hp:t>.*?</hp:tc>', sec, re.S)
        if not m:
            continue
        rest = sec[m.end():]
        nxt = re.search(r'<hp:tc\b.*?</hp:tc>', rest, re.S)
        if not nxt:
            continue
        cell = nxt.group(0)
        pm = re.search(r'<hp:p\b.*?</hp:p>', cell, re.S)
        if not pm:
            continue
        newp = Template(pm.group(0)).with_runs(runs(val, CP_BODY, CP_BODY_B))
        sec = sec[:m.end()] + rest[:nxt.start()] + cell.replace(pm.group(0), newp, 1) + rest[nxt.end():]
    return sec


# ---------------------------------------------------------------- 본문 생성

def build(md_path: Path, out_path: Path) -> None:
    form = resolve(FORM)
    md = md_path.read_text(encoding="utf-8")
    meta, docs = parse_md(md)

    with zipfile.ZipFile(form) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
        infos = {i.filename: i.compress_type for i in z.infolist()}

    sec = blobs["Contents/section0.xml"].decode("utf-8")
    header = extend_header(blobs["Contents/header.xml"].decode("utf-8"))

    spans = top_paras(sec)
    paras = [sec[a:b] for a, b in spans]

    # 견본 문단은 인덱스가 아니라 **내용**으로 찾는다. 양식이 개정돼도 견디게.
    def find(pred) -> str:
        for p in paras:
            t = re.sub(r"<[^>]+>", "", p)
            if pred(t, p):
                return p
        raise SystemExit("양식에서 서식 견본 문단을 찾지 못했다. 양식이 바뀌었는지 확인할 것.")

    sample = lambda t: "함초롬돋움" in t and "줄간격" in t
    tpl = {
        "l1": Template(find(lambda t, p: sample(t) and t.lstrip().startswith("□"))),
        "l2": Template(find(lambda t, p: sample(t) and t.lstrip().startswith("◦"))),
        "l3": Template(find(lambda t, p: sample(t) and t.lstrip().startswith("-"))),
        "note": Template(find(lambda t, p: sample(t) and t.lstrip().startswith("※"))),
        "empty": Template(find(lambda t, p: not t.strip() and "<hp:tbl" not in p)),
        "head": Template(find(lambda t, p: t.strip().startswith("1. 신약개발"))),
    }
    empty_para = tpl["empty"].xml
    prefix = {"l1": "□ ", "l2": " ◦", "l3": "   - ", "plain": ""}

    def render(blocks: list[Block]) -> str:
        out = []
        for b in blocks:
            if b.kind == "table":
                out.append(make_table(b.rows))
            elif b.kind == "note":
                r = (f'<hp:run charPrIDRef="{CP_BODY}"><hp:t>    </hp:t></hp:run>'
                     f'<hp:run charPrIDRef="{CP_NOTE}"><hp:t>※ </hp:t></hp:run>'
                     + runs(b.text, CP_NOTE, CP_NOTE_B))
                out.append(tpl["note"].with_runs(r))
            else:
                kind = b.kind if b.kind in prefix else "plain"
                pre = prefix[kind]
                r = ""
                if pre:
                    r = f'<hp:run charPrIDRef="{CP_BODY}"><hp:t>{esc(pre)}</hp:t></hp:run>'
                r += runs(b.text, CP_BODY, CP_BODY_B)
                out.append(tpl[kind if kind != "plain" else "l1"].with_runs(r))
        return "".join(out)

    # 양식 골격: 표지(표 + 그 앞뒤 문단)를 그대로 두고, <요약> 이후를 새로 쓴다.
    sum_head_idx = next(i for i, p in enumerate(paras) if "&lt;요약&gt;" in p)
    head_part = fill_cover(sec[: spans[sum_head_idx][0]], meta)

    # 표지의 참고사항 문단([참고사항 삭제 후 제출], ◈ …) 제거
    for p in paras[:sum_head_idx]:
        t = re.sub(r"<[^>]+>", "", p)
        if "[참고사항" in t or t.strip().startswith("◈"):
            head_part = head_part.replace(p, "", 1)

    body = [paras[sum_head_idx]]
    if "요약" in docs:
        body.append(render(docs["요약"]))
    body.append(empty_para)

    for key, title in SECTIONS:
        body.append(tpl["head"].with_runs(f'<hp:run charPrIDRef="20"><hp:t>{esc(title)}</hp:t></hp:run>'))
        if key in docs:
            body.append(render(docs[key]))
        body.append(empty_para)

    new_sec = head_part + "".join(body) + "</hs:sec>"

    blobs["Contents/section0.xml"] = new_sec.encode("utf-8")
    blobs["Contents/header.xml"] = header.encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w") as z:
        for n in names:  # mimetype·version.xml 은 STORED 로 유지해야 한다
            z.writestr(n, blobs[n], compress_type=infos.get(n, zipfile.ZIP_DEFLATED))

    miss = [k for k, _ in SECTIONS if k not in docs] + ([] if "요약" in docs else ["요약"])
    print(f"→ {out_path}")
    print(f"  항목 {len(docs)}개 반영" + (f" · 비어 있는 항목: {', '.join(miss)}" if miss else ""))
    check(out_path)


# ---------------------------------------------------------------- 검증

def check(path: Path) -> None:
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            print(f"  ✗ zip 손상: {bad}")
            return
        first = z.infolist()[0]
        ok_mime = first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
        for n in ("Contents/section0.xml", "Contents/header.xml"):
            try:
                ET.fromstring(z.read(n))
            except ET.ParseError as e:
                print(f"  ✗ {n} XML 오류: {e}")
                return
        sec = z.read("Contents/section0.xml").decode("utf-8")
    npara = len(re.findall(r"<hp:p\b", sec))
    ntbl = len(re.findall(r"<hp:tbl\b", sec))
    print(f"  ✓ XML 정상 · 문단 {npara} · 표 {ntbl} · mimetype {'STORED' if ok_mime else '⚠ 압축됨'}")

    texts = [html.unescape(t) for t in re.findall(r"<hp:t>(.*?)</hp:t>", sec, re.S)]
    left = [t for t in texts if "참고사항" in t or "함초롬돋움, 13pt" in t]
    if left:
        print(f"  ⚠ 양식 안내문이 {len(left)}곳 남아 있다")

    # 초안의 작업 표시가 제출본에 실리는 것이 가장 위험한 사고다. 지우지는 않고 —
    # 무엇을 쓸지는 사람이 정할 일이므로 — 위치를 짚어만 준다.
    notes = [t for t in texts if re.search(r"【[^】]*】|TODO|FIXME", t)]
    if notes:
        print(f"  ⚠ 작업 표시 {len(notes)}곳 — 제출 전 반드시 정리할 것")
        for t in notes[:6]:
            m = re.search(r"【[^】]*】|TODO|FIXME", t)
            s = max(0, m.start() - 30)
            print(f"      … {t[s:m.end() + 30].strip()} …")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "--check":
        check(Path(args[1]))
        return
    if not args:
        print(__doc__)
        sys.exit(1)
    src = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else src.with_suffix(".hwpx")
    build(src, out)


if __name__ == "__main__":
    main()
