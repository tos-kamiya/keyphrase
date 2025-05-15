import argparse
import os
import re
import sys
from typing import List, Optional, Dict, Any
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

class Headdings(BaseModel):
    line_numbers: List[int]

class ImportantSentences(BaseModel):
    line_numbers: List[int]

def extract_sentences(paragraph: str) -> List[str]:
    normalized = paragraph.replace("．", "。")
    sents_text = blingfire.text_to_sentences(normalized)
    return [s.strip() for s in sents_text.split("\n") if s.strip()]

def make_category_prompt(
    numbered: List[str],
    category: str,
    headings: Optional[List[str]] = None
) -> str:
    category_instructions = {
        "idea": (
            "Carefully select only the sentence(s) that most directly and specifically describe "
            "the main idea, method, or proposal of the paper in this paragraph."
        ),
        "experiment": (
            "Carefully select only the sentence(s) that most directly and specifically summarize "
            "experiments, evaluations, or main results."
        ),
        "threat": (
            "Carefully select only the sentence(s) that mention possible threats to validity, limitations, "
            "weaknesses, failure cases, or situations where the proposed method might not work well. "
            "This includes concerns about generalization, data quality, experimental design, "
            "and any risks or uncertainties in the effectiveness of the method."
        ),
    }
    if category not in category_instructions:
        raise ValueError(f"Unknown category: {category}")
    core_instruction = category_instructions[category]
    shared_tail = (
        " In most cases, there should be at most one, sometimes zero, such sentence(s)."
        " If none apply, return an empty list."
        " Return a JSON object with a key 'line_numbers', listing only the number(s) of the most directly relevant sentence(s)."
        " Do NOT select sentences about author names, URLs, publication info, acknowledgments, references, or general metadata."
        " Return only the JSON object, nothing else.\n\n"
    )
    heading_hint = ""
    if headings:
        heading_hint = "Context: Section headings so far:\n" + "\n".join(f"- {h}" for h in headings) + "\n\n"
    prompt = (
        heading_hint +
        "Below are numbered sentences from a scientific paper paragraph. "
        f"{core_instruction}{shared_tail}"
        + "\n".join(numbered)
    )
    return prompt

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
        format=Headdings.model_json_schema(),
    )
    result = Headdings.model_validate_json(response.message.content)
    return [i for i in result.line_numbers if 1 <= i <= len(sentences)]

def get_category_indices(
    sentences: List[str],
    category: str,
    model: str,
    retries: int = 1,
    majority_vote: bool = False,
    headings: Optional[List[str]] = None
) -> List[int]:
    numbered = [f"{i+1}: {s}" for i, s in enumerate(sentences)]
    votes: Dict[int, int] = {}
    n_iter = retries if majority_vote else 1
    for _ in range(n_iter):
        prompt = make_category_prompt(numbered, category, headings=headings)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=ImportantSentences.model_json_schema(),
        )
        try:
            result = ImportantSentences.model_validate_json(response.message.content)
            line_numbers = [i for i in result.line_numbers if 1 <= i <= len(sentences)]
            for ln in line_numbers:
                votes[ln] = votes.get(ln, 0) + 1
        except Exception as e:
            print(f"parse error for {category}: {e}", file=sys.stderr)
    if majority_vote:
        threshold = n_iter // 2 + 1
        return [idx for idx, count in votes.items() if count >= threshold]
    else:
        return list(votes.keys())

def highlight_sentences_in_pdf(
    pdf_path: str,
    output_pdf_path: str,
    model: str,
    majority_vote: bool = False,
    verbose: bool = False,
) -> None:
    """
    Highlight important sentences in a PDF using LLM judgments, with heading extraction and filtering.
    PDF is loaded only once.
    """
    doc = fitz.open(pdf_path)
    headings: List[str] = []

    for page_idx, page in enumerate(doc):
        # 段落抽出（ダブル改行で分割）
        page_text = page.get_text()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        for para in paragraphs:
            sentences = extract_sentences(para)
            if not sentences:
                continue

            # 見出し判定
            heading_line_numbers = detect_heading_indices_with_llm(sentences, model)
            num_lines = len(sentences)
            num_heading_lines = len(heading_line_numbers)

            # フィルタ条件
            if should_extract_headings(num_lines, num_heading_lines):
                new_headings = [sentences[i-1] for i in heading_line_numbers]
                if new_headings:
                    if verbose:
                        for h in new_headings:
                            print(f"[Heading] {h}", file=sys.stderr)
                    headings.extend(new_headings)

            # キーフレーズ抽出
            cat_indices_map = {
                cat: get_category_indices(
                    sentences, cat, model=model, retries=3 if majority_vote else 1,
                    majority_vote=majority_vote, headings=headings
                ) for cat in CATEGORY_PRIORITY
            }

            labeled: List[Optional[str]] = [None] * len(sentences)
            for cat in CATEGORY_PRIORITY:
                for idx in cat_indices_map[cat]:
                    orig_idx = idx - 1
                    if labeled[orig_idx] is None:
                        labeled[orig_idx] = cat

            # ハイライト
            for idx, cat in enumerate(labeled):
                if cat is None:
                    continue
                sent = sentences[idx].replace("。", "")
                for inst in page.search_for(sent):
                    annot = page.add_rect_annot(inst)
                    annot.set_colors(stroke=None, fill=COLOR_MAP[cat])
                    annot.set_opacity(0.5)
                    annot.update()
    doc.save(output_pdf_path, garbage=4)
    doc.close()
    print(f"Info: Save the highlighted pdf to: {output_pdf_path}", file=sys.stderr)


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

    headings: List[str] = []
    highlighted_paragraphs: List[str] = []
    for para in paragraphs:
        sentences = extract_sentences(para)
        if not sentences:
            highlighted_paragraphs.append(para)
            continue

        # 1. Heading detection
        heading_indices = detect_heading_indices_with_llm(sentences, model)
        num_lines = len(sentences)
        num_heading_lines = len(heading_indices)

        # 2. フィルタ
        if should_extract_headings(num_lines, num_heading_lines):
            new_headings = [sentences[i-1] for i in heading_indices]
            if new_headings:
                if verbose:
                    for h in new_headings:
                        print(f"[Heading] {h}", file=sys.stderr)
                headings.extend(new_headings)

        # 3. キーフレーズ抽出（heading_indicesは必ず除外）
        # heading_indicesは1-indexedなので、0-indexedのsetに変換
        skip_set = set(idx - 1 for idx in heading_indices)
        filtered_sentences = [s for i, s in enumerate(sentences) if i not in skip_set]

        # filtered_sentencesのインデックスを元のsentencesのインデックスに変換するマップを作る
        idx_map = [i for i in range(len(sentences)) if i not in skip_set]

        cat_indices_map = {
            cat: get_category_indices(
                filtered_sentences, cat, model=model, retries=3 if majority_vote else 1,
                majority_vote=majority_vote, headings=headings
            ) for cat in CATEGORY_PRIORITY
        }

        # ラベル付け
        labeled: List[Optional[str]] = [None] * len(sentences)
        for cat in CATEGORY_PRIORITY:
            for idx in cat_indices_map[cat]:
                orig_idx = idx_map[idx - 1]
                if labeled[orig_idx] is None:
                    labeled[orig_idx] = cat

        new_sents: List[str] = []
        for idx, sent in enumerate(sentences):
            cat = labeled[idx]
            if cat is None:
                new_sents.append(sent)
            else:
                color = MD_COLOR_MAP[cat]
                new_sents.append(f'<span style="background-color:{color}">{sent}</span>')
        para_highlighted = "".join(new_sents)
        highlighted_paragraphs.append(para_highlighted)

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
    parser.add_argument("--majority-vote", action="store_true", help="Use majority voting (3 times per category per paragraph)")
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
