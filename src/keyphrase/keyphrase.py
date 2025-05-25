import argparse
import os
import re
import sys
from typing import List, Dict, Optional, Tuple

import fitz  # PyMuPDF
import ollama
from pydantic import BaseModel, ValidationError  # ValidationError をインポート
from tqdm import tqdm

from .text_utils import extract_sentences_iter, split_markdown_paragraphs, unload_sentence_splitting_model
from .color_utils import (
    DEFAULT_COLOR_MAP,
    InvalidColorMap,
    InvalidHexColorCode,
    color_legend_str,
    hex_to_float,
    make_color_map,
)


def make_joint_category_prompt(numbered: List[str]) -> str:
    INSTRUCTIONS = (
        "Below are numbered sentences, presented in their original order as consecutive lines from a scientific paper paragraph. "
        "Each sentence should be interpreted in the context of the surrounding sentences.\n"
        "For each of the following categories, select a small number of sentences that, when combined, would best summarize that category's key points in the paragraph. "
        "When judging which category a sentence belongs to, consider the context provided by the surrounding sentences.\n"
        "\n"
        "Categories:\n"
        "  - 'approach': The most important idea, main novelty, or core contribution of the paper. Include concise descriptions or overviews of the proposed method or system as a whole. Exclude lengthy, step-by-step details or technical minutiae.\n"
        "  - 'experiment': Experimental setup, major observations and experimental results of the study. Exclude minor results or general statements.\n"
        "  - 'threat': Threats to validity, limitations, weaknesses, or potential problems with the approach or experimental results.\n"
        "  - 'reference': Reference or bibliography sentences, typically listing prior work or sources cited in the paper.\n"
        "\n"
        "Try to choose sentences that cover different aspects of the category, rather than multiple sentences expressing the same or similar ideas. "
        "Avoid selecting long blocks of consecutive sentences within a single category—if several consecutive sentences are candidates, select only the most essential one(s).\n"
        "Return a JSON object with exactly these four keys: 'approach', 'experiment', 'threat', 'reference'. "
        "Each key should have a list of 0-based indices for the sentences that belong to that category.\n"
        "If no sentence fits a category, use an empty list for that category.\n"
        "Example:\n"
        "{\n"
        '  "approach": [2],\n'
        '  "experiment": [4],\n'
        '  "threat": [7],\n'
        '  "reference": [10]\n'
        "}\n"
        "\n"
        "Numbered sentences:\n"
    )
    return INSTRUCTIONS + "\n".join(numbered)


class JointImportantPhrases(BaseModel):
    approach: List[int]
    experiment: List[int]
    threat: List[int]
    reference: List[int]


def label_sentences(sentences: List[str], model: str) -> List[Optional[str]]:
    """
    Assign a category label to each sentence using LLM.
    The strategy is to try LLM multiple times and select the response
    that classifies the fewest sentences (excluding 'reference' for this size metric),
    aiming for a more concise selection.

    Args:
        sentences (List[str]): Sentences to label.
        model (str): LLM model name.

    Returns:
        List[Optional[str]]: List of category labels ("approach", "experiment", "threat", "reference") or None.
    """
    numbered = [f"{idx}: {s}" for idx, s in enumerate(sentences)]

    maps = []
    num_retries = 3
    for i in range(num_retries):
        prompt = make_joint_category_prompt(numbered)
        response = None
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format=JointImportantPhrases.model_json_schema(),
            )
            result = JointImportantPhrases.model_validate_json(response.message.content)

            size = len(result.approach) + len(result.experiment) + len(result.threat)
            cat_indices_map = {
                "approach": result.approach,
                "experiment": result.experiment,
                "threat": result.threat,
            }
            maps.append((size, cat_indices_map))
        except ValidationError as e:
            assert response is not None
            print(f"Warning: LLM returned invalid JSON schema on attempt {i + 1}: {e}", file=sys.stderr)
            print(f"LLM response content: {response.message.content}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: LLM chat call failed on attempt {i + 1}: {e}", file=sys.stderr)
            if "response" in locals() and response and hasattr(response, "message"):
                print(f"LLM response content (if available): {response.message.content}", file=sys.stderr)

    if not maps:
        print(
            f"Error: No valid LLM responses received after {num_retries} attempts. Returning all sentences as unclassified.",
            file=sys.stderr,
        )
        return [None] * len(sentences)

    # Select the map with the smallest size (fewest classified sentences in core categories)
    maps.sort(key=lambda sc: sc[0])
    cat_indices_map = maps[0][1]

    labeled: List[Optional[str]] = [None] * len(sentences)

    for cat_name in ["approach", "experiment", "threat"]:
        for idx in cat_indices_map[cat_name]:
            if 0 <= idx < len(sentences):  # Validate index is within bounds
                if labeled[idx] is None:  # Only label if not already labeled by a previous category
                    labeled[idx] = cat_name
            else:
                print(
                    f"Warning: LLM returned out-of-bounds index {idx} for category '{cat_name}'. Skipping.",
                    file=sys.stderr,
                )
    return labeled


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
) -> None:
    """
    Process a batch (buffer) of sentences for a PDF, highlighting sentences according to category.

    Args:
        doc (fitz.Document): PyMuPDF document object.
        buffer (List[Tuple[int, int, str]]): Sentences to process, with page indices.
        model (str): LLM model name.
    """
    sentences = [item[2] for item in buffer]

    # Batch label and annotate
    labels = label_sentences(sentences, model)

    for (page_idx, sent_idx, sent), cat in zip(buffer, labels):
        if not cat or cat not in pdf_color_map:  # Ensure category is valid for highlighting
            continue
        page = doc[page_idx]

        search_text = sent.strip()
        if not search_text:  # Avoid searching for empty strings
            continue

        quads = page.search_for(search_text)
        if len(quads) == 1:
            highlight = page.add_highlight_annot(quads[0])
            r, g, b, a = pdf_color_map[cat]
            highlight.set_colors({"stroke": (r, g, b)})
            highlight.set_opacity(a)
            highlight.update()
        elif len(quads) == 0:
            if sys.stderr.isatty():
                print(f"Warning: '{search_text}' not found on page {page_idx+1}.", file=sys.stderr)


def highlight_sentences_in_pdf(
    pdf_path: str,
    output_pdf_path: str,
    color_map: Dict[str, str],
    model: str,
    buffer_size: int,
    max_sentence_length: int,
    verbose: bool = False,
) -> None:
    """
    Highlight important sentences in a PDF file using the LLM.

    Args:
        pdf_path (str): Input PDF path.
        output_pdf_path (str): Output (highlighted) PDF path.
        model (str): LLM model name.
        buffer_size (int): Buffer size (character count).
        max_sentence_length (int): Maximum length of each sentence.
        verbose (bool): If True, print progress.
    """
    pdf_color_map = {k: hex_to_float(v) for k, v in color_map.items()}
    doc = fitz.open(pdf_path)
    buffer: List[Tuple[int, int, str]] = []

    if verbose:
        print(f"Split sentences ...", file=sys.stderr)
    page_paragraph_sentences_data: Dict[int, List[Tuple[int, List[str]]]] = dict()
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        paragraphs = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
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
                process_buffered_pdf(doc, buffer, pdf_color_map, model)
                buffer.clear()

        # Process any remaining sentences in the buffer at the end of the page
        if buffer:
            process_buffered_pdf(doc, buffer, pdf_color_map, model)
            buffer.clear()

    if buffer:  # This check is redundant due to page-level processing, but harmless.
        process_buffered_pdf(doc, buffer, pdf_color_map, model)
        buffer.clear()

    doc.save(output_pdf_path, garbage=4)
    doc.close()
    print(f"Info: Save the highlighted pdf to: {output_pdf_path}", file=sys.stderr)


def process_buffered_md(
    buffer: List[Tuple[int, int, str]],  # (para_idx, sent_idx, sentence)
    color_map: Dict[str, str],
    model: str,
) -> Dict[Tuple[int, int], str]:
    """
    Process a batch (buffer) of sentences for a Markdown file, returning highlights as HTML spans.

    Args:
        buffer (List[Tuple[int, int, str]]): Sentences to process, with paragraph indices.
        model (str): LLM model name.

    Returns:
        Dict[Tuple[int, int], str]: Mapping (para_idx, sent_idx) to highlighted HTML.
    """
    sentences = [item[2] for item in buffer]

    # Batch label and collect highlights
    labels = label_sentences(sentences, model)
    highlights: Dict[Tuple[int, int], str] = {}
    for (para_idx, sent_idx, sent), cat in zip(buffer, labels):
        # Ensure category is valid and has a corresponding color in MD_COLOR_MAP
        if cat and cat in color_map:
            highlights[(para_idx, sent_idx)] = f'<span style="background-color:{DEFAULT_COLOR_MAP[cat]}">{sent}</span>'
    return highlights


def has_base64_image(paragraph: str) -> bool:
    return re.search(r'!\[.*?\]\(\s*data:image/[^)]+\)', paragraph, re.DOTALL) is not None


def highlight_sentences_in_md(
    md_path: str,
    output_path: Optional[str],
    color_map: Dict[str, str],
    model: str,
    buffer_size: int,
    max_sentence_length: int,
    verbose: bool = False,
) -> None:
    """
    Highlight important sentences in a Markdown file using the LLM.

    Args:
        md_path (str): Input Markdown path.
        output_path (Optional[str]): Output file path (if None, print to stdout).
        model (str): LLM model name.
        buffer_size (int): Buffer size (character count).
        max_sentence_length (int): Maximum length of each sentence.
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
            batch_highlights = process_buffered_md(buffer, color_map, model)
            highlighted.update(batch_highlights)
            buffer.clear()

    # Process any remaining sentences in the buffer after all paragraphs are done
    if buffer:
        highlighted.update(process_buffered_md(buffer, color_map, model))
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
    """
    Build and return argparse.ArgumentParser for 'normal' or 'color_legend'.
    """
    assert mode in ("normal", "color_legend")

    parser = argparse.ArgumentParser(
        description="Highlight key sentences in PDF or Markdown (AI-based color coding, heading-aware)"
    )

    if mode != "color_legend":  # the input argument is required in normal mode, but not required in "legend" mode
        parser.add_argument("input", help="Input PDF or Markdown file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-o", "--output", help="Output file. Use '-' for stdout (Markdown only).")
    group.add_argument("-O", "--output-auto", action="store_true", help="Output to INPUT-annotated.(pdf|md).")
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
        default="qwen3:30b-a3b",
        help="LLM for key sentence detection (default: 'qwen3:30b-a3b').",
    )
    parser.add_argument(
        "--color-map",
        type=str,
        action="append",
        help=(
            "Customize marker colors. "
            "Format: 'name:#rgba' or 'name:#rrrggbbaa' (e.g., approach:#8edefbb0). "
            "Supported names: approach, experiment, threat. "
            "To disable a marker, use 'name:0'."
        ),
    )
    parser.add_argument(
        "--color-legend",
        choices=["text", "ansi", "html"],
        help="Output color legend in the specified format (text|ansi|html).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show progress bar with tqdm.")
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
            max_sentence_length=args.max_sentence_length,
            verbose=args.verbose,
        )
    elif filetype == "md":
        highlight_sentences_in_md(
            input_path,
            output_path,
            color_map=color_map,
            model=args.model,
            buffer_size=args.buffer_size,
            max_sentence_length=args.max_sentence_length,
            verbose=args.verbose,
        )
    else:
        print("Unknown file type", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
