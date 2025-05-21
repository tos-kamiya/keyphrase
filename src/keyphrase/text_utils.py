import re
from typing import List, Optional

import blingfire


def split_markdown_paragraphs(md_text: str) -> List[str]:
    """
    Splits a Markdown text into paragraphs, treating headings, list items,
    code blocks, and quote blocks as separate paragraphs.

    - Headings (lines starting with #) always start a new paragraph.
    - List items (lines starting with '-', '+', '*', or numbered lists) are single-paragraph.
    - Code blocks (delimited by ``` or ~~~) are extracted as single paragraphs.
    - Quote blocks (lines starting with '>') are grouped as single paragraphs.
    - Normal paragraphs are split by empty lines.

    Args:
        md_text (str): The input Markdown text.

    Returns:
        List[str]: List of paragraphs as strings.
    """
    lines: List[str] = md_text.splitlines()
    paragraphs: List[str] = []
    buffer: List[str] = []
    in_code_block: bool = False
    code_fence: str | None = None
    in_quote_block: bool = False

    def flush() -> None:
        """Flush the current buffer as a paragraph if not empty."""
        nonlocal buffer
        if buffer:
            para = "\n".join(buffer).strip()
            if para:
                paragraphs.append(para)
            buffer = []

    for line in lines:
        stripped = line.strip()

        # Check for code block start/end (``` or ~~~)
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

        # Heading (e.g., "#", "##", etc.) always starts a new paragraph
        if re.match(r"^#{1,6}\s", stripped):
            flush()
            paragraphs.append(line)
            continue

        # List item (unordered or ordered) as a single-paragraph
        if re.match(r"^(\s*([-+*]|\d+\.)\s)", line):
            flush()
            paragraphs.append(line)
            continue

        # Quote block (lines starting with ">")
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

        # Empty line splits paragraphs
        if not stripped:
            flush()
            continue

        # Regular line, add to buffer
        buffer.append(line)

    # Flush any remaining lines in the buffer
    flush()
    return paragraphs


def split_by_punct_character(text: str, sentence_max_length: int) -> List[str]:
    words = []
    for text in re.split("( )", text):
        if text == " ":
            words.append(text)
        else:
            ws = blingfire.text_to_words(text).split(" ")
            words.extend(ws)

    phrases = [""]
    for w in words:
        phrases[-1] += w
        if len(w) == 1:  # is punct char
            phrases.append("")
    if phrases[-1] == "":
        phrases.pop()

    merged_phrases = [""]
    for p in phrases:
        if len(merged_phrases[-1] + p) <= sentence_max_length:
            merged_phrases[-1] += p
        else:
            merged_phrases.append(p)
    if merged_phrases[-1] == "":
        merged_phrases.pop()

    return merged_phrases


def extract_sentences(paragraph: str, sentence_max_length: Optional[int] = 100) -> List[str]:
    normalized = paragraph.replace("．", "。")
    sents_text = blingfire.text_to_sentences(normalized)

    sents = [s.strip() for s in sents_text.split("\n") if s.strip()]
    r = []
    for s in sents:
        if len(s) <= sentence_max_length:
            r.append(s)
        else:
            phrases = split_by_punct_character(s, sentence_max_length)
            r.extend(phrases)
    return r


# def extract_sentences(paragraph: str, sentence_max_length: Optional[int] = 100) -> List[str]:
#     lines = paragraph.split("\n")
#     r = []
#     for line in lines:
#         if len(line) <= sentence_max_length:
#             r.append(line)
#             continue
#
#         normalized = paragraph.replace("．", "。")
#         sents_text = blingfire.text_to_sentences(normalized)
#
#         sents = [s.strip() for s in sents_text.split("\n") if s.strip()]
#         for s in sents:
#             if len(s) <= sentence_max_length:
#                 r.append(s)
#                 continue
#
#             phrases = split_by_punct_character(s, sentence_max_length)
#             r.extend(phrases)
#     return r
