import argparse
import os
import re
import sys
from typing import List, Set, Dict, Optional

import fitz
import blingfire
import ollama
from pydantic import BaseModel
from tqdm import tqdm

from .__about__ import __version__


class ImportantSentenceIndices(BaseModel):
    indices: List[int]


def extract_paragraphs_from_pdf(pdf_path: str) -> List[List[str]]:
    doc = fitz.open(pdf_path)
    all_page_paragraphs = []
    for page in doc:
        page_text = page.get_text()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        all_page_paragraphs.append(paragraphs)
    return all_page_paragraphs


def extract_sentences(paragraph: str) -> List[str]:
    normalized = paragraph.replace("．", "。")  # blingfire does not recognize "．" as Japanese punctuation
    sents_text = blingfire.text_to_sentences(normalized)
    sents = [s.strip() for s in sents_text.split("\n") if s.strip()]
    return sents


def make_category_prompt(numbered: list[str], category: str) -> str:
    """
    Generate an LLM prompt for extracting key sentences in a given category.

    Args:
        numbered: List of numbered sentences as strings.
        category: One of "idea", "experiment", "threat".

    Returns:
        A string prompt for the LLM.
    """
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
        " Return a JSON object with a key 'indices', listing only the number(s) of the most directly relevant sentence(s)."
        " Do NOT select sentences about author names, URLs, publication info, acknowledgments, references, or general metadata."
        " Return only the JSON object, nothing else.\n\n"
    )

    prompt = (
        "Below are numbered sentences from a scientific paper paragraph. "
        f"{core_instruction}{shared_tail}"
        + "\n".join(numbered)
    )

    return prompt


def get_category_indices(
    sentences: List[str],
    category: str,
    model: str,
    retries: int = 3,
    majority_vote: bool = False,
    pbar: Optional[tqdm] = None,
) -> Set[int]:
    """
    Query the LLM for important sentence indices for a category.
    If majority_vote is True, query multiple times and use majority vote.

    Args:
        sentences: List of sentences in the paragraph.
        category: Category label ("idea", "experiment", "threat").
        model: LLM model name.
        retries: Number of retries or voting rounds.
        majority_vote: If True, perform majority voting over multiple runs.
        pbar: tqdm progress bar (optional).

    Returns:
        Set of 1-based indices.
    """
    if not sentences:
        return set()
    numbered = [f"{i+1}. {s}" for i, s in enumerate(sentences)]
    prompt = make_category_prompt(numbered, category)

    votes: Dict[int, int] = {}
    n_iter = retries if majority_vote else 1
    for i in range(n_iter):
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=ImportantSentenceIndices.model_json_schema(),
        )
        try:
            c = response.message.content
            assert i is not None
            result = ImportantSentenceIndices.model_validate_json(c)
            indices = [i for i in result.indices if 1 <= i <= len(sentences)]
            for idx in indices:
                votes[idx] = votes.get(idx, 0) + 1
        except Exception as e:
            print(f"parse error for {category}: {e}", file=sys.stderr)
        if pbar:
            pbar.update(1)

    if majority_vote:
        # Keep indices that appeared more than half the time
        threshold = n_iter // 2 + 1
        return {idx for idx, count in votes.items() if count >= threshold}
    else:
        return set(votes.keys())


CATEGORY_PRIORITY = ["threat", "experiment", "idea"]
PDF_COLOR_MAP = {
    "idea": (0, 0.5, 1),  # Blue
    "experiment": (0, 1, 0),  # Green
    "threat": (1, 1, 0),  # Yellow
}
MD_COLOR_MAP = {
    "idea": "#3498db",  # Blue
    "experiment": "#27ae60",  # Green
    "threat": "#f7e158",  # Yellow
}


def highlight_sentences_in_pdf(
    pdf_path: str,
    output_pdf_path: str,
    all_page_paragraphs: List[List[str]],
    model: str,
    majority_vote: bool = False,
    verbose: bool = False,
) -> None:
    """
    Highlight important sentences in a PDF based on LLM judgments.

    Args:
        pdf_path: Path to input PDF.
        output_pdf_path: Path to output PDF.
        all_page_paragraphs: List of paragraphs per page.
        model: LLM model name.
        majority_vote: If True, use majority voting for LLM queries.
        verbose: If True, show progress bar with tqdm.
    """
    doc = fitz.open(pdf_path)
    total = sum(len(paras) for paras in all_page_paragraphs) * len(CATEGORY_PRIORITY) * (3 if majority_vote else 1)
    pbar = tqdm(total=total, desc="Highlighting", disable=not verbose)
    for page_idx, paragraphs in enumerate(all_page_paragraphs):
        page = doc[page_idx]
        for para in paragraphs:
            sentences = extract_sentences(para)
            if not sentences:
                pbar.update(len(CATEGORY_PRIORITY) * (3 if majority_vote else 1))
                continue

            # Query each category with majority vote if specified
            cat_indices = {
                cat: get_category_indices(
                    sentences,
                    cat,
                    model=model,
                    retries=3 if majority_vote else 1,
                    majority_vote=majority_vote,
                    pbar=pbar,
                )
                for cat in CATEGORY_PRIORITY
            }

            # Assign color by priority
            labeled: List[Optional[str]] = [None] * len(sentences)
            for cat in CATEGORY_PRIORITY:
                for idx in cat_indices[cat]:
                    if labeled[idx - 1] is None:
                        labeled[idx - 1] = cat

            for idx, cat in enumerate(labeled):
                if cat is None:
                    continue
                sent = sentences[idx].replace("。", "")  # blingfire does not recognize "．" as Japanese punctuation
                for inst in page.search_for(sent):
                    annot = page.add_rect_annot(inst)
                    annot.set_colors(stroke=None, fill=PDF_COLOR_MAP[cat])
                    annot.set_opacity(0.5)
                    annot.update()
    pbar.close()

    doc.save(output_pdf_path, garbage=4)
    doc.close()

    print(f"Info: Save the highlighted pdf to: {output_pdf_path}", file=sys.stderr)


def highlight_sentences_in_md(
    md_path: str, output_path: Optional[str], model: str, majority_vote: bool = False, verbose: bool = False
) -> None:
    """
    Highlight important sentences in a Markdown file using HTML span tags.

    Args:
        md_path: Path to input Markdown.
        output_path: Path to output Markdown. Specify `None` for the standard output.
        model: LLM model name.
        majority_vote: If True, use majority voting for LLM queries.
        verbose: If True, show progress bar with tqdm.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    total = len(paragraphs) * len(CATEGORY_PRIORITY) * (3 if majority_vote else 1)
    pbar = tqdm(total=total, desc="Highlighting", disable=not verbose)

    highlighted_paragraphs: List[str] = []
    for para in paragraphs:
        sentences = extract_sentences(para)
        if not sentences:
            highlighted_paragraphs.append(para)
            pbar.update(len(CATEGORY_PRIORITY) * (3 if majority_vote else 1))
            continue

        cat_indices = {
            cat: get_category_indices(
                sentences, cat, model=model, retries=3 if majority_vote else 1, majority_vote=majority_vote, pbar=pbar
            )
            for cat in CATEGORY_PRIORITY
        }

        labeled: List[Optional[str]] = [None] * len(sentences)
        for cat in CATEGORY_PRIORITY:
            for idx in cat_indices[cat]:
                if labeled[idx - 1] is None:
                    labeled[idx - 1] = cat

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

    pbar.close()

    out_text = "\n\n".join(highlighted_paragraphs)
    if output_path is None:
        print(out_text)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Info: Save the highlighted markdown to: {output_path}", file=sys.stderr)


def get_output_path(
    input_path: str, output: Optional[str], output_auto: bool, suffix: str = "-annotated",
    overwrite: bool = False
) -> Optional[str]:
    """
    Decide output path. If output == '-', return None (stdout).
    """
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


def detect_filetype(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pdf"]:
        return "pdf"
    elif ext in [".md", ".markdown"]:
        return "md"
    else:
        raise ValueError(f"Unknown file extension: {ext}")


def main() -> None:
    """
    Command-line entry point for AI-based key sentence highlighting.
    """
    parser = argparse.ArgumentParser(description="Highlight key sentences in PDF or Markdown (AI-based color coding)")
    parser.add_argument("input", help="input PDF or Markdown file")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-o", "--output",
        help="Output file name. Use '-' (a single hyphen) to write output to standard output (stdout)."
    )
    group.add_argument("-O", "--output-auto", action="store_true", help="Output to INPUT-annotated.(pdf|md)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite when the output file exists.")
    parser.add_argument(
        "-m", "--model", type=str, default="qwen3:30b", help="LLM for identify key sentences (default: 'qwen3:30b')."
    )
    parser.add_argument(
        "-3", "--majority-vote", action="store_true", help="Use majority voting (3 times per category per paragraph)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress bar with tqdm.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    args = parser.parse_args()

    input_path: str = args.input
    filetype: str = detect_filetype(input_path)
    output_path: Optional[str] = get_output_path(input_path, args.output, args.output_auto, overwrite=args.overwrite)

    if filetype == "pdf":
        if output_path is None:
            print(
                "Error: Output to standard output ('-o -') is not supported for PDF files. "
                "Please specify an output file name.",
                file=sys.stderr
            )
            sys.exit(1)
        all_page_paragraphs: List[List[str]] = extract_paragraphs_from_pdf(input_path)
        highlight_sentences_in_pdf(
            input_path,
            output_path,
            all_page_paragraphs,
            model=args.model,
            majority_vote=args.majority_vote,
            verbose=args.verbose,
        )
    elif filetype == "md":
        highlight_sentences_in_md(
            input_path, output_path, model=args.model, majority_vote=args.majority_vote, verbose=args.verbose
        )
    else:
        print("Unknown file type", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
