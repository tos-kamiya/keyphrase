import re


def split_markdown_paragraphs(md_text):
    lines = md_text.splitlines()
    paragraphs = []
    buffer = []
    in_code_block = False
    code_fence = None
    in_quote_block = False

    def flush():
        nonlocal buffer
        if buffer:
            para = "\n".join(buffer).strip()
            if para:
                paragraphs.append(para)
            buffer = []

    for line in lines:
        stripped = line.strip()
        # コードブロック開始・終了判定
        code_match = re.match(r"^(```|~~~)", stripped)
        if code_match:
            fence = code_match.group(1)
            if not in_code_block:
                flush()
                in_code_block = True
                code_fence = fence
                buffer.append(line)
            elif in_code_block and stripped.startswith(code_fence):
                buffer.append(line)
                in_code_block = False
                code_fence = None
                flush()
            else:
                buffer.append(line)
            continue
        if in_code_block:
            buffer.append(line)
            continue

        # 見出し判定
        if re.match(r"^#{1,6}\s", stripped):
            flush()
            paragraphs.append(line)
            continue

        # リストアイテム判定
        if re.match(r"^(\s*([-+*]|\d+\.)\s)", line):
            flush()
            paragraphs.append(line)
            continue

        # 引用判定
        if stripped.startswith(">"):
            if not in_quote_block:
                flush()
                in_quote_block = True
            buffer.append(line)
            continue
        else:
            if in_quote_block:
                flush()
                in_quote_block = False

        # 空行判定
        if not stripped:
            flush()
            continue

        # 通常行
        buffer.append(line)

    # バッファ残り
    flush()
    return paragraphs
