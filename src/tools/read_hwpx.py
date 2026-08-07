"""hwpx 리더.

hwpx = ZIP 컨테이너 + OWPML(XML). 본문은 Contents/section*.xml 에 들어있고,
문단은 <hp:p>, 텍스트 런은 <hp:t> 이다. 표는 <hp:tbl> / <hp:tc> 로 표현된다.

용도: 대회 제안서 양식의 항목 구조를 뽑아 무엇을 채워야 하는지 파악한다.

사용법:
    python src/tools/read_hwpx.py <파일.hwpx>              # 본문 텍스트
    python src/tools/read_hwpx.py <파일.hwpx> --outline    # 번호 항목만 추려서 개요
    python src/tools/read_hwpx.py <파일.hwpx> --raw        # 내부 파일 목록
"""

from __future__ import annotations

import re
import sys
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def resolve(path: Path) -> Path:
    """한글 파일명의 NFC/NFD 불일치를 흡수한다.

    브라우저·macOS에서 받은 파일은 분해형(NFD)으로 저장되는 경우가 있어서,
    조합형(NFC)으로 입력한 경로와 문자열이 달라 exists()가 False가 된다.
    같은 디렉터리에서 정규화 후 일치하는 항목을 찾아 실제 경로를 돌려준다.
    """
    if path.exists():
        return path
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.is_dir():
        return path
    want = unicodedata.normalize("NFC", path.name)
    for entry in parent.iterdir():
        if unicodedata.normalize("NFC", entry.name) == want:
            return entry
    return path

# OWPML 네임스페이스는 버전에 따라 URI가 달라서 접두사에 의존하지 않고 로컬명으로 매칭한다.
def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def section_files(z: zipfile.ZipFile) -> list[str]:
    names = [n for n in z.namelist() if re.search(r"Contents/section\d+\.xml$", n)]
    if not names:
        # 일부 생성기는 다른 경로를 쓴다
        names = [n for n in z.namelist() if n.endswith(".xml") and "section" in n.lower()]
    return sorted(names)


def paragraphs(xml_bytes: bytes) -> list[str]:
    """<hp:p> 단위로 텍스트를 뽑는다. 표 셀도 문단을 포함하므로 자연히 함께 나온다."""
    root = ET.fromstring(xml_bytes)
    out: list[str] = []

    def walk(node, in_para: list[str] | None):
        for child in node:
            name = local(child.tag)
            if name == "p":
                buf: list[str] = []
                walk(child, buf)
                text = "".join(buf).strip()
                out.append(text)
                continue
            if name == "t":
                target = in_para if in_para is not None else None
                if target is not None:
                    target.append("".join(child.itertext()))
                else:
                    out.append("".join(child.itertext()))
                continue
            if name in ("lineBreak", "tab") and in_para is not None:
                in_para.append(" ")
            walk(child, in_para)

    walk(root, None)
    return out


def read_text(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        secs = section_files(z)
        if not secs:
            raise SystemExit(
                f"본문 section XML을 찾지 못했습니다. 내부 파일 목록:\n  "
                + "\n  ".join(z.namelist()[:40])
            )
        paras: list[str] = []
        for name in secs:
            paras.extend(paragraphs(z.read(name)))
    return paras


# 번호 항목 후보: "1.", "1)", "①", "가.", "<요약>", "[항목]" 등
NUMBERED = re.compile(
    r"^\s*("
    r"\d{1,2}\s*[.)]"
    r"|[①-⑳]"
    r"|[가-하]\s*[.)]"
    r"|[<\[(【]\s*\S+?\s*[>\])】]"
    r"|제?\s*\d{1,2}\s*(항목|장|절)"
    r")"
)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    path = resolve(Path(sys.argv[1]))
    flags = set(sys.argv[2:])

    if not path.exists():
        raise SystemExit(f"파일이 없습니다: {path}")

    if "--raw" in flags:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                print(f"  {info.file_size:>9,}  {info.filename}")
        return

    paras = [p for p in read_text(path) if p]

    if "--outline" in flags:
        print(f"# {path.name} — 번호 항목 개요\n")
        for p in paras:
            if NUMBERED.match(p):
                print(f"  {p[:160]}")
        return

    print(f"# {path.name} — 본문 {len(paras)}문단\n")
    for p in paras:
        marker = "§ " if NUMBERED.match(p) else "  "
        print(f"{marker}{p}")


if __name__ == "__main__":
    main()
