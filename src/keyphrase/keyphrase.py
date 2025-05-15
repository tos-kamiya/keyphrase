import argparse
import os
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import fitz  # PyMuPDF
import blingfire
import ollama
from pydantic import BaseModel

CATEGORY_PRIORITY = ["threat", "experiment", "idea"]
COLOR_MAP = {
    "idea": (0, 0.5, 1),    # blue
    "experiment": (0, 1, 0),# green
    "threat": (1, 1, 0),    # yellow
}
MD_COLOR_MAP = {
    "idea": "#3498db",       # blue
    "experiment": "#27ae60", # green
    "threat": "#f7e158",     # yellow
}

# Buffer size threshold for batch processing
BUFFER_SENTENCE_COUNT = 50

class ImportantSentences(BaseModel):
    line_numbers: List[int]


def extract_sentences(paragraph: str) -> List[str]:
    normalized = paragraph.replace("．", "。")
    sents_text = blingfire.text_to_sentences(normalized)
    return [s.strip() for s in sents_text.split("\n") if s.strip()]


def make_category_prompt(
    numbered: List[str],
    category: str
) -> str:
    category_instructions = {
        "idea": (
            "Carefully select only the sentence(s) that most directly and specifically describe "
            "the main motivations, ideas, approaches of the paper."
        ),
        "experiment": (
            "Carefully select only the sentence(s) that most directly and specifically summarize "
            "experiments, evaluations, or main results."
        ),
        "threat": (
            "Carefully select only the sentence(s) that mention possible threats to validity, limitations, "
            "weaknesses, failure cases. "
            "This includes concerns about generalization, data quality, experimental design, "
            "and any risks or uncertainties in the effectiveness of the method."
        ),
    }
    if category not in category_instructions:
        raise ValueError(f"Unknown category: {category}")
    core_instruction = category_instructions[category]
    shared_tail = (
        "In most cases, there should be at most one, sometimes zero, such sentence(s).  "
        "If none apply, return an empty list.\n"
        "Return a JSON object with a key 'line_numbers', listing only the number(s) of the most directly relevant sentence(s). "
        "Do NOT select sentences about author names, URLs, publication info, acknowledgments, references, or general metadata. "
        "Return only the JSON object, nothing else.\n\n"
    )
    return (
        "Below are numbered sentences from a scientific paper paragraph. "
        f"{core_instruction}\n{shared_tail}"
        + "\n".join(numbered)
    )


def should_extract_headings(num_lines: int, num_heading_lines: int) -> bool:
    return not (num_lines >= 10 and num_heading_lines / num_lines >= 0.5)


def make_heading_prompt(numbered: List[str]) -> str:
    instruction = (
        "From the sentence with line numbers below, select only those that are likely to be section headings, "
        "titles, or subsection titles. If none apply, return an empty list. "
        "Return a JSON object with a key 'line_numbers', listing only the number(s) of heading-like sentences. "
        "Do NOT select sentences that are part of the body text or typical content. "
        "Return only the JSON object, nothing else.\n\n"
    )
    return instruction + "\n".join(numbered)


def detect_heading_indices_with_llm(
    sentences: List[str],
    model: str,
) -> List[int]:
    numbered = [f"{i+1}: {s}" for i, s in enumerate(sentences)]
    prompt = make_heading_prompt(numbered)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=ImportantSentences.model_json_schema(),
    )
    result = ImportantSentences.model_validate_json(response.message.content)
    return [i for i in result.line_numbers if 1 <= i <= len(sentences)]


def get_category_indices(
    sentences: List[str],
    category: str,
    model: str,
    retries: int = 1,
    majority_vote: bool = False
) -> List[int]:
    numbered = [f"{i+1}: {s}" for i, s in enumerate(sentences)]
    votes: Dict[int, int] = {}
    n_iter = retries if majority_vote else 1
    for _ in range(n_iter):
        prompt = make_category_prompt(numbered, category)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=ImportantSentences.model_json_schema(),
        )
        try:
            result = ImportantSentences.model_validate_json(response.message.content)
            for ln in result.line_numbers:
                if 1 <= ln <= len(sentences):
                    votes[ln] = votes.get(ln, 0) + 1
        except Exception as e:
            print(f"parse error for {category}: {e}", file=sys.stderr)
    if majority_vote:
        threshold = n_iter // 2 + 1
        return [idx for idx, count in votes.items() if count >= threshold]
    return list(votes.keys())


def label_sentences(
    sentences: List[str],
    model: str,
    majority_vote: bool = False
) -> List[Optional[str]]:
    cat_indices_map = {
        cat: get_category_indices(
            sentences, cat,
            model=model,
            retries=3 if majority_vote else 1,
            majority_vote=majority_vote
        ) for cat in CATEGORY_PRIORITY
    }
    labeled: List[Optional[str]] = [None] * len(sentences)
    for cat in CATEGORY_PRIORITY:
        for idx in cat_indices_map[cat]:
            orig_idx = idx - 1
            if labeled[orig_idx] is None:
                labeled[orig_idx] = cat
    return labeled


def process_buffered_pdf(
    doc,
    buffer: List[Tuple[int, int, str]],  # (page_idx, sent_idx, sentence)
    headings: List[str],
    model: str,
    majority_vote: bool,
    verbose: bool = False
) -> None:
    # Batch detect headings
    sentences = [item[2] for item in buffer]
    heading_idxs = detect_heading_indices_with_llm(sentences, model)
    if should_extract_headings(len(sentences), len(heading_idxs)):
        new_h = [sentences[i-1] for i in heading_idxs]
        if new_h:
            if verbose:
                for h in new_h:
                    print(f"[Heading] {h}", file=sys.stderr)
            headings.extend(new_h)
    # Batch label and annotate
    labels = label_sentences(sentences, model, majority_vote)
    for (page_idx, sent_idx, sent), cat in zip(buffer, labels):
        if not cat:
            continue
        page = doc[page_idx]
        text = sent.replace("。", "")
        for quads in page.search_for(text):
            highlight = page.add_highlight_annot(quads)
            r, g, b = COLOR_MAP[cat]
            highlight.set_colors({"stroke": (r, g, b)})
            highlight.set_opacity(0.5)
            highlight.update()


def highlight_sentences_in_pdf(
    pdf_path: str,
    output_pdf_path: str,
    model: str,
    majority_vote: bool = False,
    verbose: bool = False,
) -> None:
    doc = fitz.open(pdf_path)
    buffer: List[Tuple[int, int, str]] = []
    headings: List[str] = []

    for page_idx, page in enumerate(doc):
        page_text = page.get_text()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        for para in paragraphs:
            sentences = extract_sentences(para)
            for idx, sent in enumerate(sentences):
                buffer.append((page_idx, idx, sent))
            if len(buffer) >= BUFFER_SENTENCE_COUNT:
                process_buffered_pdf(doc, buffer, headings, model, majority_vote, verbose)
                buffer.clear()
    if buffer:
        process_buffered_pdf(doc, buffer, headings, model, majority_vote, verbose)
    doc.save(output_pdf_path, garbage=4)
    doc.close()
    print(f"Info: Save the highlighted pdf to: {output_pdf_path}", file=sys.stderr)


def process_buffered_md(
    buffer: List[Tuple[int, int, str]],  # (para_idx, sent_idx, sentence)
    headings: List[str],
    model: str,
    majority_vote: bool,
    verbose: bool = False
) -> Dict[Tuple[int, int], str]:
    # Batch detect headings
    sentences = [item[2] for item in buffer]
    heading_idxs = detect_heading_indices_with_llm(sentences, model)
    if should_extract_headings(len(sentences), len(heading_idxs)):
        new_h = [sentences[i-1] for i in heading_idxs]
        if new_h:
            if verbose:
                for h in new_h:
                    print(f"[Heading] {h}", file=sys.stderr)
            headings.extend(new_h)
    # Batch label and collect highlights
    labels = label_sentences(sentences, model, majority_vote)
    highlights: Dict[Tuple[int, int], str] = {}
    for (para_idx, sent_idx, sent), cat in zip(buffer, labels):
        if cat:
            highlights[(para_idx, sent_idx)] = f'<span style="background-color:{MD_COLOR_MAP[cat]}">{sent}</span>'
    return highlights


def highlight_sentences_in_md(
    md_path: str,
    output_path: Optional[str],
    model: str,
    majority_vote: bool = False,
    verbose: bool = False,
) -> None:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

    buffer: List[Tuple[int, int, str]] = []
    headings: List[str] = []
    highlighted: Dict[Tuple[int, int], str] = {}

    for p_idx, para in enumerate(paragraphs):
        sentences = extract_sentences(para)
        for s_idx, sent in enumerate(sentences):
            buffer.append((p_idx, s_idx, sent))
        if len(buffer) >= BUFFER_SENTENCE_COUNT:
            batch_highlights = process_buffered_md(buffer, headings, model, majority_vote, verbose)
            highlighted.update(batch_highlights)
            buffer.clear()
    if buffer:
        highlighted.update(process_buffered_md(buffer, headings, model, majority_vote, verbose))

    # Reconstruct with highlights
    highlighted_paragraphs: List[str] = []
    for p_idx, para in enumerate(paragraphs):
        sentences = extract_sentences(para)
        new_sents: List[str] = []
        for s_idx, sent in enumerate(sentences):
            new_sents.append(highlighted.get((p_idx, s_idx), sent))
        highlighted_paragraphs.append("".join(new_sents))

    out_text = "\n\n".join(highlighted_paragraphs)
    if output_path is None:
        print(out_text)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Info: Save the highlighted markdown to: {output_path}", file=sys.stderr)


def get_output_path(
    input_path: str,
    output: Optional[str],
    output_auto: bool,
    suffix: str = "-annotated",
    overwrite: bool = False
) -> Optional[str]:
    if output == "-":
        return None
    ext = os.path.splitext(input_path)[1].lower()
    if output:
        out_path = output
    elif output_auto:
        base, _ = os.path.splitext(input_path)
        out_path = f"{base}{suffix}{ext}"
    else:
        out_path = f"out{ext}"
    if not overwrite and os.path.exists(out_path):
        print(f"Error: output file already exists: {out_path}", file=sys.stderr)
        sys.exit(1)
    return out_path


def detect_filetype(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".md":
        return "md"
    raise ValueError("Unknown file type")


def main() -> None:
    parser = argparse.ArgumentParser(description="Highlight key sentences in PDF or Markdown (AI-based color coding, heading-aware)")
    parser.add_argument("input", help="input PDF or Markdown file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-o", "--output", help="Output file")
    group.add_argument("-O", "--output-auto", action="store_true", help="Output to INPUT-annotated.(pdf|md)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite when the output file exists.")
    parser.add_argument("-m", "--model", type=str, default="qwen3:30b", help="LLM for identify key sentences (default: 'qwen3:30b').")
    parser.add_argument("-3", "--majority-vote", action="store_true", help="Use majority voting (3 times per category per paragraph)")
    parser.add_argument("--verbose", action="store_true", help="Show progress bar with tqdm.")
    args = parser.parse_args()

    input_path: str = args.input
    filetype: str = detect_filetype(input_path)
    output_path: Optional[str] = get_output_path(
        input_path, args.output, args.output_auto,
        suffix="-annotated", overwrite=args.overwrite
    )

    if filetype == "pdf":
        if output_path is None:
            print("Error: Output to standard output ('-o -') is not supported for PDF files. Use '-o OUTPUT.pdf'.", file=sys.stderr)
            sys.exit(1)
        highlight_sentences_in_pdf(
            input_path,
            output_path,
            model=args.model,
            majority_vote=args.majority_vote,
            verbose=args.verbose
        )
    elif filetype == "md":
        highlight_sentences_in_md(
            input_path,
            output_path,
            model=args.model,
            majority_vote=args.majority_vote,
            verbose=args.verbose
        )
    else:
        print("Unknown file type", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
