import re
from typing import Iterator, List, Optional
import unicodedata


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
    code_fence: Optional[str] = None
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


def is_punctuation(token: str) -> bool:
    return len(token) == 1 and unicodedata.category(token)[0] == "P"


def split_by_punct_character(text: str, sentence_max_length: int) -> List[str]:
    phrases = [""]
    for ch in text:
        if is_punctuation(ch):
            phrases.append(ch)
            phrases.append("")
        else:
            phrases[-1] += ch
    if phrases[-1] == "":
        phrases.pop()

    merged_phrases = [""]
    for p in phrases:
        if not merged_phrases[-1] or len(merged_phrases[-1] + p) <= sentence_max_length:
            merged_phrases[-1] += p
        else:
            merged_phrases.append(p)
    if merged_phrases and merged_phrases[-1] == "":
        merged_phrases.pop()

    # If the last paragraph is too short, merge it to the previous paragraph.
    if len(merged_phrases) >= 2 and len(merged_phrases[-1]) < sentence_max_length // 3:
        p = merged_phrases.pop()
        merged_phrases[-1] += p

    return merged_phrases


_sat_instance = None

def get_sat():
    global _sat_instance
    if _sat_instance is None:
        from wtpsplit import SaT
        _sat_instance = SaT("sat-3l")
    return _sat_instance


def unload_sentence_splitting_model():
    global _sat_instance
    _sat_instance = None


def remove_control_characters(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', s)


def extract_sentences(paragraph: str, sentence_max_length: int = 100) -> List[str]:
    p = remove_control_characters(paragraph).strip()
    if not p:
        return []

    sat = get_sat()
    sents = [s.strip() for s in sat.split(p) if s.strip()]
    r = []
    for s in sents:
        if len(s) <= sentence_max_length:
            r.append(s)
        else:
            phrases = split_by_punct_character(s, sentence_max_length)
            r.extend(phrases)
    return r


def extract_sentences_iter(paragraphs: List[str], sentence_max_length: int = 100) -> Iterator[List[str]]:
    cleaned_paragraphs = []
    for p in paragraphs:
        p = remove_control_characters(p).strip()
        if p:
            cleaned_paragraphs.append(p)

    sat = get_sat()
    split_results = sat.split(cleaned_paragraphs)
    for sr in split_results:
        sents = [s.strip() for s in sr if s.strip()]
        r = []
        for s in sents:
            if len(s) <= sentence_max_length:
                r.append(s)
            else:
                phrases = split_by_punct_character(s, sentence_max_length)
                r.extend(phrases)
        yield r

