import argparse
import os
import re
import sys
from typing import List, Dict, Optional, Tuple

import fitz  # PyMuPDF
import ollama
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .color_utils import (
    DEFAULT_COLOR_MAP,
    InvalidColorMap,
    InvalidHexColorCode,
    color_legend_str,
    hex_to_float,
    make_color_map,
)
from .pdf_utils import extract_paragraphs_in_page, find_least_appearance
from .text_utils import extract_sentences_iter, split_markdown_paragraphs, unload_sentence_splitting_model


def make_sentence_category_prompt(numbered: List[str]) -> str:
    INSTRUCTIONS = (
        "Below are numbered sentences, presented in their original order as consecutive lines from a scientific paper paragraph, "
        "possibly from only one page or section (not necessarily the start of the paper). "
        "Each sentence should be interpreted in the context of the surrounding sentences.\n"
        "From the numbered list of sentences below, identify the key sentences for each of the following categories.\n"
        "\n"
        "Categories:\n"
        "  - 'approach': The most important idea, main novelty, or core contribution of the paper. Include concise descriptions or overviews of the proposed method or system as a whole. Exclude lengthy, step-by-step details or technical minutiae.\n"
        "  - 'experiment': Experimental setup, major observations and experimental results of the study. Exclude minor results or general statements.\n"
        "  - 'threat': Threats to validity, limitations, weaknesses, or potential problems with the approach or experimental results.\n"
        "  - 'reference': Reference or bibliography sentences, typically listing prior work or sources cited in the paper.\n"
        "\n"
        "Return a JSON object with exactly these four keys: 'approach', 'experiment', 'threat', 'reference'. "
        "Each key should have a list of 0-based indices for the sentences that belong to that category.\n"
        "Example:\n"
        "{\n"
        '  \"approach\": [2],\n'
        '  \"experiment\": [4],\n'
        '  \"threat\": [7],\n'
        '  \"reference\": [10]\n'
        "}\n"
        "\n"
        "Numbered sentences:\n"
    )

    return INSTRUCTIONS + "\n".join(numbered)


class SentenceCategory(BaseModel):
    approach: List[int]
    experiment: List[int]
    threat: List[int]
    reference: List[int]


def make_skim_prompt(numbered: List[str]) -> str:
    INSTRUCTIONS = (
        "Below are numbered sentences, presented in their original order as consecutive lines from a scientific paper paragraph, "
        "possibly from only one page or section (not necessarily the start of the paper). "
        "Each sentence should be interpreted in the context of the surrounding sentences.\n"
        "From the numbered list of sentences below, identify the key sentences.\n"
        "Also list reference or bibliography sentences, typically listing prior work or sources cited in the paper.\n"
        "Example:\n"
        '{\n  "skim": [0, 3, 5],\n  "reference": [2, 8]\n}\n'
        "Numbered sentences:\n"
    )
    return INSTRUCTIONS + "\n".join(numbered)


class Skim(BaseModel):
    skim: List[int]
    reference: List[int]


def cleanup_response_content(content: str) -> str:
    while re.match(r"\s*<think>.*?</think>\s*", content, re.DOTALL):
        content = re.sub(r"\s*<think>.*?</think>\s*", "", content, flags=re.DOTALL)

    if re.match(r"^```json[ \t]*\n", content, re.DOTALL) and re.match(r".*```[ \t}]*\s*$", content, re.DOTALL):
        content = re.sub(r"^```json[ \t]*\n", "", content, flags=re.DOTALL)
        content = re.sub(r"```[ \t}]*\s*$", "", content, flags=re.DOTALL)

    if re.match(r"^\s*\{\s*\{*", content, re.DOTALL) and re.match(r".*\}\s*\}\s*$", content, re.DOTALL):
        content = re.sub(r"^\s*\{\s*", "", content, flags=re.DOTALL)
        content = re.sub(r"\s*\}\s*$", "", content, flags=re.DOTALL)

    return content


def label_sentences(
    sentences: List[str], model: str, extraction: str = "aetr", debug: bool = False
) -> List[Optional[str]]:
    """
    Assign a category label to each sentence using LLM.
    The strategy is to try LLM multiple times, feeding previous responses back to refine the result.
    The final result from the last successful trial is used.

    Args:
        sentences (List[str]): Sentences to label.
        extraction: 'aetr' for approach/experiment/threat/reference, 's' for skim mode.
        model (str): LLM model name.
        debug (bool): Enable debug prints.

    Returns:
        List[Optional[str]]: List of category label of each sentence or None.
    """

    def extract_labels(result):
        labels: List[Optional[str]] = [None] * len(sentences)
        category_data = result.model_dump()
        for cat, idxs in category_data.items():
            if cat == "reference":
                continue
            for idx in idxs:
                if 0 <= idx < len(sentences) and labels[idx] is None:
                    labels[idx] = cat

        # Remove the items categorized as references
        for cat, idxs in category_data.items():
            if cat == "reference":
                for idx in idxs:
                    if 0 <= idx < len(sentences):
                        labels[idx] = None
        return labels

    numbered = [f"{idx}: {s}" for idx, s in enumerate(sentences)]

    if extraction == "aetr":
        prompt_text = make_sentence_category_prompt(numbered)
        validator = SentenceCategory.model_validate_json
    elif extraction == "s":
        prompt_text = make_skim_prompt(numbered)
        validator = Skim.model_validate_json
    else:
        raise ValueError(f"Unknown extraction mode: {extraction}")

    last_valid_labels: List[Optional[str]] = []
    num_trials = 3
    messages = [{"role": "user", "content": prompt_text}]

    for i in range(num_trials):
        if debug:
            print(f"--- Attempt {i + 1}/{num_trials} ---", file=sys.stderr)
            print("[Debug] Sending messages:", file=sys.stderr)
            for msg in messages:
                print(f"  Role: {msg['role']}", file=sys.stderr)
                content_preview = (msg['content'][:150] + '...') if len(msg['content']) > 150 else msg['content']
                print(f"  Content: {content_preview.strip()}", file=sys.stderr)

        try:
            response = ollama.chat(
                model=model,
                messages=messages,
            )
            content = response.get("message", {}).get("content", "")
            if not content:
                print(f"Warning: Empty response content on attempt {i + 1}", file=sys.stderr)
                continue

            if debug:
                print("[Debug] Raw LLM response:", file=sys.stderr)
                print(f"> {content}", file=sys.stderr)

            content = cleanup_response_content(content)

            try:
                result = validator(content)
            except ValidationError:
                content = content.rstrip() + "}"  # Add missing closing brace when JSON validation fails due to incomplete structure
                result = validator(content)
            last_valid_labels = extract_labels(result)  # Overwrite with the latest valid labels0

            messages.append({"role": "assistant", "content": content})
            if i < num_trials - 1:
                messages.append({
                    "role": "user",
                    "content": "That was a good attempt. Please review your previous answer and provide a new, improved one. Focus on accuracy and conciseness."
                })

        except ValidationError as e:
            print(f"Warning: LLM returned invalid JSON on attempt {i + 1}: {e}", file=sys.stderr)
            if 'content' in locals():
                content = content.rstrip()
                print(f"LLM response content: {content}", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred during LLM call on attempt {i + 1}: {e}", file=sys.stderr)

    if not last_valid_labels:
        print("Warning: LLM returned no valid responses; skipping highlights.", file=sys.stderr)
        return [None] * len(sentences)

    # Use the result from the last successful trial.
    return last_valid_labels


def buffer_len_chars(buffer: List[Tuple[int, int, str]]) -> int:
    """
    Calculate the total character length of all sentences in the buffer.

    Args:
        buffer (List[Tuple[int, int, str]]): Buffer of (para/page idx, sent idx, sentence).

    Returns:
        int: Total character count.
    """
    return sum(len(sent) for (_, _, sent) in buffer)


def process_buffered_pdf(
    doc: fitz.Document,
    buffer: List[Tuple[int, int, str]],  # (page_idx, sent_idx, sentence)
    pdf_color_map: Dict[str, Tuple[float, float, float, float]],
    model: str,
    extraction: str = "aetr",  # 'aetr' or 's'
    debug: bool = False,
) -> None:
    """
    Process a batch (buffer) of sentences for a PDF, highlighting sentences according to category.

    Args:
        doc (fitz.Document): PyMuPDF document object.
        buffer (List[Tuple[int, int, str]]): Sentences to process, with page indices.
        extraction: 'aetr' for approach/experiment/threat/reference, 's' for skim mode.
        model (str): LLM model name.
    """
    sentences = [item[2] for item in buffer]

    # Batch label and annotate
    labels = label_sentences(sentences, model, extraction=extraction, debug=debug)

    for (page_idx, sent_idx, sent), cat in zip(buffer, labels):
        if not cat or cat not in pdf_color_map:  # Ensure category is valid for highlighting
            continue
        page = doc[page_idx]

        search_text = sent.strip()
        if not search_text:  # Avoid searching for empty strings
            continue

        found_text, count = find_least_appearance(search_text, page)
        if found_text and count == 1:
            quads = page.search_for(found_text)
            highlight = page.add_highlight_annot(quads[0])
            r, g, b, a = pdf_color_map[cat]
            highlight.set_colors({"stroke": (r, g, b)})
            highlight.set_opacity(a)
            highlight.update()
        else:
            if count == 0:
                print(f"Warning: '{search_text}' not found on page {page_idx+1}.", file=sys.stderr)


def highlight_sentences_in_pdf(
    pdf_path: str,
    output_pdf_path: str,
    color_map: Dict[str, str],
    model: str,
    buffer_size: int,
    max_sentence_length: int,
    extraction: str = "aetr",  # 'aetr' or 's'
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """
    Highlight important sentences in a PDF file using the LLM.

    Args:
        pdf_path (str): Input PDF path.
        output_pdf_path (str): Output (highlighted) PDF path.
        model (str): LLM model name.
        buffer_size (int): Buffer size (character count).
        max_sentence_length (int): Maximum length of each sentence.
        extraction: 'aetr' for approach/experiment/threat/reference, 's' for skim mode.
        verbose (bool): If True, print progress.
    """
    pdf_color_map = {k: hex_to_float(v) for k, v in color_map.items()}
    doc = fitz.open(pdf_path)
    buffer: List[Tuple[int, int, str]] = []

    if verbose:
        print(f"Split sentences ...", file=sys.stderr)
    page_paragraph_sentences_data: Dict[int, List[Tuple[int, List[str]]]] = dict()
    for page_idx, page in enumerate(doc):
        paragraphs = extract_paragraphs_in_page(page)
        paragraph_sentences_data: List[Tuple[int, List[str]]] = list(
            enumerate(extract_sentences_iter(paragraphs, max_sentence_length))
        )
        page_paragraph_sentences_data[page_idx] = paragraph_sentences_data
    unload_sentence_splitting_model()

    pages = list(doc)
    if verbose:
        it = tqdm(pages, unit="page", desc="Key Extraction")
    else:
        it = pages
    for page_idx, page in enumerate(it):
        paragraph_sentences_data = page_paragraph_sentences_data[page_idx]

        # Ensure that the last batch in a page is also processed
        for p_idx, sentences in paragraph_sentences_data:
            for idx, sent in enumerate(sentences):
                if len(sent) >= 5:
                    buffer.append((page_idx, idx, sent))
            if buffer_len_chars(buffer) >= buffer_size:
                process_buffered_pdf(doc, buffer, pdf_color_map, model, extraction=extraction, debug=debug)
                buffer.clear()

        # Process any remaining sentences in the buffer at the end of the page
        if buffer:
            process_buffered_pdf(doc, buffer, pdf_color_map, model, extraction=extraction, debug=debug)
            buffer.clear()

    if buffer:  # This check is redundant due to page-level processing, but harmless.
        process_buffered_pdf(doc, buffer, pdf_color_map, model, extraction=extraction, debug=debug)
        buffer.clear()

    doc.save(output_pdf_path, garbage=4)
    doc.close()
    print(f"Info: Save the highlighted pdf to: {output_pdf_path}", file=sys.stderr)


def process_buffered_md(
    buffer: List[Tuple[int, int, str]],  # (para_idx, sent_idx, sentence)
    color_map: Dict[str, str],
    model: str,
    extraction: str = "aetr",  # 'aetr' or 's'
    debug: bool = False,
) -> Dict[Tuple[int, int], str]:
    """
    Process a batch (buffer) of sentences for a Markdown file, returning highlights as HTML spans.

    Args:
        buffer (List[Tuple[int, int, str]]): Sentences to process, with paragraph indices.
        model (str): LLM model name.
        extraction: 'aetr' for approach/experiment/threat/reference, 's' for skim mode.

    Returns:
        Dict[Tuple[int, int], str]: Mapping (para_idx, sent_idx) to highlighted HTML.
    """
    sentences = [item[2] for item in buffer]

    # Batch label and collect highlights
    labels = label_sentences(sentences, model, extraction=extraction, debug=debug)
    highlights: Dict[Tuple[int, int], str] = {}
    for (para_idx, sent_idx, sent), cat in zip(buffer, labels):
        # Ensure category is valid and has a corresponding color in color_map
        if cat and cat in color_map:
            css_color = color_map[cat]
            highlights[(para_idx, sent_idx)] = f'<span style="background-color:{css_color}">{sent}</span>'
    return highlights


def has_base64_image(paragraph: str) -> bool:
    return re.search(r"!\[.*?\]\(\s*data:image/[^)]+\)", paragraph, re.DOTALL) is not None


def highlight_sentences_in_md(
    md_path: str,
    output_path: Optional[str],
    color_map: Dict[str, str],
    model: str,
    buffer_size: int,
    max_sentence_length: int,
    extraction: str = "aetr",  # 'aetr' or 's'
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """
    Highlight important sentences in a Markdown file using the LLM.

    Args:
        md_path (str): Input Markdown path.
        output_path (Optional[str]): Output file path (if None, print to stdout).
        model (str): LLM model name.
        buffer_size (int): Buffer size (character count).
        max_sentence_length (int): Maximum length of each sentence.
        extraction: 'aetr' for approach/experiment/threat/reference, 's' for skim mode.
        verbose (bool): If True, print progress.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    if has_base64_image(content):
        print("Error: the Markdown file contains base64-encoded images.", file=sys.stderr)
        sys.exit(1)

    paragraphs = split_markdown_paragraphs(content)

    buffer: List[Tuple[int, int, str]] = []
    highlighted: Dict[Tuple[int, int], str] = {}

    # Prepare paragraph -> sentences table
    if verbose:
        print(f"Split sentences ...", file=sys.stderr)
    paragraph_sentences_data: List[Tuple[int, List[str]]] = list(
        enumerate(extract_sentences_iter(paragraphs, max_sentence_length))
    )
    unload_sentence_splitting_model()

    if verbose:
        it = tqdm(paragraph_sentences_data, unit="paragraph", desc="Key Extraction")
    else:
        it = paragraph_sentences_data

    # Process paragraphs and buffer sentences
    for p_idx, sentences in it:
        for s_idx, sent in enumerate(sentences):
            buffer.append((p_idx, s_idx, sent))

        # If buffer size threshold is reached, process the batch
        if buffer_len_chars(buffer) >= buffer_size:
            batch_highlights = process_buffered_md(buffer, color_map, model, extraction=extraction, debug=debug)
            highlighted.update(batch_highlights)
            buffer.clear()

    # Process any remaining sentences in the buffer after all paragraphs are done
    if buffer:
        highlighted.update(process_buffered_md(buffer, color_map, model, extraction=extraction, debug=debug))
        buffer.clear()

    # Reconstruct Markdown with highlighted sentences
    highlighted_paragraphs: List[str] = []
    for p_idx, sentences in paragraph_sentences_data:
        new_sents: List[str] = []
        for s_idx, sent in enumerate(sentences):
            # If a sentence is highlighted, use the highlighted HTML; otherwise, use the original sentence
            new_sents.append(highlighted.get((p_idx, s_idx), sent))
        # Join sentences to form the paragraph. Assuming extract_sentences preserves original spacing/punctuation
        highlighted_paragraphs.append("".join(new_sents))

    out_text = "\n\n".join(highlighted_paragraphs)
    if output_path is None:
        print(out_text)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Info: Save the highlighted markdown to: {output_path}", file=sys.stderr)


def get_output_path(
    input_path: str, output: Optional[str], output_auto: bool, suffix: str = "-annotated", overwrite: bool = False
) -> Optional[str]:
    """
    Compute the output path based on the input and options.

    Args:
        input_path (str): Input file path.
        output (Optional[str]): Output path specified by the user.
        output_auto (bool): Whether to automatically generate output name.
        suffix (str): Suffix to append for auto-generated outputs.
        overwrite (bool): If False, do not overwrite existing files.

    Returns:
        Optional[str]: Output file path or None (if output to stdout).

    Raises:
        SystemExit: If the output file exists and overwrite is not allowed.
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
        # Default output name if neither -o nor -O is specified
        # This will create "out.pdf" or "out.md" in the current directory
        out_path = f"out{ext}"

    always_overwrite = {"out.md", "out.pdf"}
    filename_only = os.path.basename(out_path).lower()

    if not overwrite and os.path.exists(out_path) and filename_only not in always_overwrite:
        print(f"Error: output file already exists: {out_path}", file=sys.stderr)
        sys.exit(1)
    return out_path


def detect_filetype(path: str) -> str:
    """
    Detect the file type ("pdf" or "md") from extension.

    Args:
        path (str): File path.

    Returns:
        str: "pdf" or "md".

    Raises:
        ValueError: If the extension is not recognized.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".md":
        return "md"
    raise ValueError(f"Unknown file type for path: {path}. Supported types are .pdf and .md")


def build_parser(mode: str) -> argparse.ArgumentParser:
    assert mode in ("normal", "color_legend")
    parser = argparse.ArgumentParser(
        description="Highlight key sentences in PDF or Markdown (AI-based color coding, heading-aware)"
    )

    if mode != "color_legend":
        parser.add_argument("input", help="Input PDF or Markdown file")

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("-o", "--output", help="Output file. Use '-' for stdout (Markdown only).")
    group1.add_argument("-O", "--output-auto", action="store_true", help="Output to INPUT-annotated.(pdf|md).")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite when the output file exists.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1300,
        help="Buffer size threshold for batch processing (characters, default: 1300)",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=120,
        help="Maximum length of each sentence for analysis (default: 120)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="LLM for key sentence detection (default: 'gpt-oss:20b').",
    )
    parser.add_argument(
        "--skim",
        action="store_true",
        help="If set, extract key-phrases only, otherwise, extract approach/experiment/threat.",
    )
    parser.add_argument(
        "--color-map",
        type=str,
        action="append",
        help="Customize marker colors. Format: 'name:#rgba' or 'name:#rrrggbbaa'.",
    )
    parser.add_argument(
        "--color-legend",
        choices=["text", "ansi", "html"],
        help="Output color legend in the specified format (text|ansi|html).",
    )

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output (disable verbose).")
    group2.add_argument("--debug", action="store_true", help="Enable debug output (prompt/response) and progress bar.")
    group2.add_argument("--verbose", action="store_true", help="Enable progress bar (default).")

    return parser


def main() -> None:
    # Check if --color-legend specified
    legend_mode = any(arg.startswith("--color-legend") for arg in sys.argv)

    # Color legend mode
    if legend_mode:
        parser = build_parser("color_legend")
        args = parser.parse_args()
        try:
            color_map = make_color_map(DEFAULT_COLOR_MAP, args.color_map)
        except InvalidColorMap | InvalidHexColorCode as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(color_legend_str(color_map, args.color_legend))
        sys.exit(0)

    # Normal mode
    parser = build_parser("normal")
    args = parser.parse_args()

    verbose = not args.quiet or args.debug or args.verbose
    debug = args.debug

    try:
        color_map = make_color_map(DEFAULT_COLOR_MAP, args.color_map)
    except InvalidColorMap | InvalidHexColorCode as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    input_path: str = args.input
    try:
        filetype: str = detect_filetype(input_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_path: Optional[str] = get_output_path(
        input_path, args.output, args.output_auto, suffix="-annotated", overwrite=args.overwrite
    )

    if filetype == "pdf":
        if output_path is None:
            print(
                "Error: Output to standard output ('-o -') is not supported for PDF files. Use '-o OUTPUT.pdf'.",
                file=sys.stderr,
            )
            sys.exit(1)
        highlight_sentences_in_pdf(
            input_path,
            output_path,
            color_map=color_map,
            model=args.model,
            buffer_size=args.buffer_size,
            extraction="s" if args.skim else "aetr",
            max_sentence_length=args.max_sentence_length,
            verbose=verbose,
            debug=debug,
        )
    elif filetype == "md":
        highlight_sentences_in_md(
            input_path,
            output_path,
            color_map=color_map,
            model=args.model,
            buffer_size=args.buffer_size,
            extraction="s" if args.skim else "aetr",
            max_sentence_length=args.max_sentence_length,
            verbose=verbose,
            debug=debug,
        )
    else:
        print("Unknown file type", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
