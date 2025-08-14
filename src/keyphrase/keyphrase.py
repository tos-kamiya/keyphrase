import argparse
from collections import Counter
import os
import re
import sys
from typing import List, Dict, Optional, Tuple

import fitz  # PyMuPDF
try:
    import ollama
except ImportError:
    ollama = None

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
from .harmony_ollama import HarmonyOllamaClient

# --- Prompts and Pydantic Models ---

def make_sentence_category_prompt(numbered: List[str]) -> str:
    INSTRUCTIONS = (
        "Below are numbered sentences from a scientific paper. For each category ('approach', 'experiment', 'threat', 'reference'), "
        "select the sentences that best fit. Return a JSON object with four keys, each containing a list of 0-based indices.\n"
        "Example: {\"approach\": [2], \"experiment\": [4], \"threat\": [7], \"reference\": [10]}\n"
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
        "From the numbered list of sentences below, identify key sentences (max 10) and reference sentences. "
        "Return a JSON object with 'skim' and 'reference' keys, each with a list of 0-based indices.\n"
        "Example: {\"skim\": [0, 3, 5], \"reference\": [2, 8]}\n"
        "Numbered sentences:\n"
    )
    return INSTRUCTIONS + "\n".join(numbered)


class Skim(BaseModel):
    skim: List[int]
    reference: List[int]

# --- LLM Interaction ---

def label_sentences_ollama_native(
    sentences: List[str], model: str, extraction: str, debug: bool
) -> List[Optional[str]]:
    if ollama is None:
        raise ImportError("The 'ollama' package is required for the 'ollama' backend but is not installed.")

    numbered = [f"{idx}: {s}" for idx, s in enumerate(sentences)]
    if extraction == "aetr":
        prompt_text = make_sentence_category_prompt(numbered)
        validator = SentenceCategory.model_validate_json
    else:  # 's'
        prompt_text = make_skim_prompt(numbered)
        validator = Skim.model_validate_json

    messages = [{"role": "user", "content": prompt_text}]
    try:
        response = ollama.chat(model=model, messages=messages, format="json")
        content = response.get("message", {}).get("content", "")
        if not content:
            print("Warning: Empty response from Ollama.", file=sys.stderr)
            return [None] * len(sentences)
        
        result = validator(content)
        
        labels: List[Optional[str]] = [None] * len(sentences)
        category_data = result.model_dump()
        for cat, idxs in category_data.items():
            if cat != "reference":
                for idx in idxs:
                    if 0 <= idx < len(labels):
                        labels[idx] = cat
        return labels

    except (ValidationError, Exception) as e:
        print(f"Warning: Ollama native call failed: {e}", file=sys.stderr)
        return [None] * len(sentences)


def label_sentences_harmony(
    sentences: List[str], client: HarmonyOllamaClient, extraction: str, debug: bool
) -> List[Optional[str]]:
    numbered = [f"{idx}: {s}" for idx, s in enumerate(sentences)]
    if extraction == "aetr":
        prompt_text = make_sentence_category_prompt(numbered)
        pydantic_model = SentenceCategory
    else:  # 's'
        prompt_text = make_skim_prompt(numbered)
        pydantic_model = Skim

    try:
        result = client.generate_json(prompt_text, pydantic_model, debug=debug)
        
        labels: List[Optional[str]] = [None] * len(sentences)
        category_data = result.model_dump()
        for cat, idxs in category_data.items():
            # Do not highlight references
            if cat != "reference":
                for idx in idxs:
                    if 0 <= idx < len(labels):
                        labels[idx] = cat
        return labels

    except RuntimeError as e:
        print(f"Warning: Harmony client failed to get a valid response: {e}", file=sys.stderr)
        return [None] * len(sentences)


def label_sentences(
    sentences: List[str],
    llm_backend: str,
    model: str,
    extraction: str,
    ollama_base_url: str,
    debug: bool,
) -> List[Optional[str]]:
    if llm_backend == "ollama":
        return label_sentences_ollama_native(sentences, model, extraction, debug)
    elif llm_backend == "harmony":
        client = HarmonyOllamaClient(base_url=ollama_base_url, model=model)
        return label_sentences_harmony(sentences, client, extraction, debug)
    else:
        raise ValueError(f"Unknown LLM backend: {llm_backend}")


# --- Core Processing Logic ---

def buffer_len_chars(buffer: List[Tuple[int, int, str]]) -> int:
    return sum(len(sent) for (_, _, sent) in buffer)


def process_buffered_pdf(
    doc: fitz.Document,
    buffer: List[Tuple[int, int, str]],
    pdf_color_map: Dict[str, Tuple[float, float, float, float]],
    llm_backend: str,
    model: str,
    extraction: str,
    ollama_base_url: str,
    debug: bool,
) -> None:
    sentences = [item[2] for item in buffer]
    labels = label_sentences(sentences, llm_backend, model, extraction, ollama_base_url, debug)

    for (page_idx, _, sent), cat in zip(buffer, labels):
        if not cat or cat not in pdf_color_map:
            continue
        
        page = doc[page_idx]
        search_text = sent.strip()
        if not search_text:
            continue

        found_text, count = find_least_appearance(search_text, page)
        if found_text and count == 1:
            quads = page.search_for(found_text)
            if quads:
                highlight = page.add_highlight_annot(quads[0])
                r, g, b, a = pdf_color_map[cat]
                highlight.set_colors({"stroke": (r, g, b)})
                highlight.set_opacity(a)
                highlight.update()
        elif count > 1:
             print(f"Warning: Skipping ambiguous text '{search_text}' (found {count} times) on page {page_idx+1}.", file=sys.stderr)
        else:
            print(f"Warning: Text '{search_text}' not found on page {page_idx+1}.", file=sys.stderr)


def highlight_sentences_in_pdf(
    pdf_path: str,
    output_pdf_path: str,
    color_map: Dict[str, str],
    llm_backend: str,
    model: str,
    ollama_base_url: str,
    buffer_size: int,
    max_sentence_length: int,
    extraction: str,
    verbose: bool,
    debug: bool,
) -> None:
    pdf_color_map = {k: hex_to_float(v) for k, v in color_map.items()}
    doc = fitz.open(pdf_path)
    buffer: List[Tuple[int, int, str]] = []

    if verbose:
        print("Splitting sentences...", file=sys.stderr)
    
    page_paragraph_sentences_data = {
        page_idx: list(enumerate(extract_sentences_iter(extract_paragraphs_in_page(page), max_sentence_length)))
        for page_idx, page in enumerate(doc)
    }
    unload_sentence_splitting_model()

    it = tqdm(doc, unit="page", desc="Key Extraction") if verbose else doc
    for page_idx, page in enumerate(it):
        for p_idx, sentences in page_paragraph_sentences_data.get(page_idx, []):
            for s_idx, sent in enumerate(sentences):
                if len(sent.strip()) > 5:
                    buffer.append((page_idx, s_idx, sent))
            
            if buffer_len_chars(buffer) >= buffer_size:
                process_buffered_pdf(doc, buffer, pdf_color_map, llm_backend, model, extraction, ollama_base_url, debug)
                buffer.clear()

        if buffer: # Process remaining buffer at the end of each page
            process_buffered_pdf(doc, buffer, pdf_color_map, llm_backend, model, extraction, ollama_base_url, debug)
            buffer.clear()

    doc.save(output_pdf_path, garbage=4)
    doc.close()
    print(f"Info: Saved the highlighted PDF to: {output_pdf_path}", file=sys.stderr)


def process_buffered_md(
    buffer: List[Tuple[int, int, str]],
    color_map: Dict[str, str],
    llm_backend: str,
    model: str,
    extraction: str,
    ollama_base_url: str,
    debug: bool,
) -> Dict[Tuple[int, int], str]:
    sentences = [item[2] for item in buffer]
    labels = label_sentences(sentences, llm_backend, model, extraction, ollama_base_url, debug)
    
    highlights: Dict[Tuple[int, int], str] = {}
    for (para_idx, sent_idx, sent), cat in zip(buffer, labels):
        if cat and cat in color_map:
            css_color = color_map[cat]
            highlights[(para_idx, sent_idx)] = f'<span style="background-color:{css_color}">{sent}</span>'
    return highlights


def highlight_sentences_in_md(
    md_path: str,
    output_path: Optional[str],
    color_map: Dict[str, str],
    llm_backend: str,
    model: str,
    ollama_base_url: str,
    buffer_size: int,
    max_sentence_length: int,
    extraction: str,
    verbose: bool,
    debug: bool,
) -> None:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "data:image/" in content:
        print("Error: Markdown file appears to contain base64-encoded images, which is not supported.", file=sys.stderr)
        sys.exit(1)

    paragraphs = split_markdown_paragraphs(content)
    if verbose:
        print("Splitting sentences...", file=sys.stderr)
    
    paragraph_sentences_data = list(enumerate(extract_sentences_iter(paragraphs, max_sentence_length)))
    unload_sentence_splitting_model()

    buffer: List[Tuple[int, int, str]] = []
    highlighted: Dict[Tuple[int, int], str] = {}
    
    it = tqdm(paragraph_sentences_data, unit="paragraph", desc="Key Extraction") if verbose else paragraph_sentences_data
    for p_idx, sentences in it:
        for s_idx, sent in enumerate(sentences):
            if len(sent.strip()) > 5:
                buffer.append((p_idx, s_idx, sent))
        
        if buffer_len_chars(buffer) >= buffer_size:
            batch_highlights = process_buffered_md(buffer, color_map, llm_backend, model, extraction, ollama_base_url, debug)
            highlighted.update(batch_highlights)
            buffer.clear()

    if buffer:
        batch_highlights = process_buffered_md(buffer, color_map, llm_backend, model, extraction, ollama_base_url, debug)
        highlighted.update(batch_highlights)
        buffer.clear()

    reconstructed_paragraphs = [
        "".join(highlighted.get((p_idx, s_idx), sent) for s_idx, sent in enumerate(sentences))
        for p_idx, sentences in paragraph_sentences_data
    ]
    
    out_text = "\n\n".join(reconstructed_paragraphs)
    if output_path is None:
        print(out_text)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Info: Saved the highlighted markdown to: {output_path}", file=sys.stderr)


# --- CLI and Main Entry ---

def get_output_path(
    input_path: str, output: Optional[str], output_auto: bool, suffix: str = "-annotated", overwrite: bool = False
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

    if not overwrite and os.path.exists(out_path) and os.path.basename(out_path).lower() not in {"out.md", "out.pdf"}:
        print(f"Error: Output file already exists: {out_path}", file=sys.stderr)
        sys.exit(1)
    return out_path


def detect_filetype(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".md":
        return "md"
    raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .md are supported.")


def build_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Highlight key sentences in scientific documents using LLMs.")
    
    if mode != "color_legend":
        parser.add_argument("input", help="Input PDF or Markdown file")

    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-o", "--output", help="Output file path. Use '-' for stdout (Markdown only).")
    output_group.add_argument("-O", "--output-auto", action="store_true", help="Automatically generate output name as INPUT-annotated.ext.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")

    # LLM and processing options
    llm_group = parser.add_argument_group("LLM and Processing")
    llm_group.add_argument(
        "--llm-backend",
        type=str,
        default="harmony",
        choices=["harmony", "ollama"],
        help="The LLM backend to use. 'harmony' for raw Harmony format, 'ollama' for native JSON mode. Default: harmony.",
    )
    llm_group.add_argument(
        "-m", "--model", type=str, default="gpt-oss:20b", help="LLM model name (e.g., 'gpt-oss:20b')."
    )
    llm_group.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama API server.",
    )
    llm_group.add_argument(
        "--skim", action="store_true", help="Use 'skim' mode for extraction instead of detailed categories."
    )
    llm_group.add_argument(
        "--buffer-size", type=int, default=1300, help="Character buffer size for batch processing (default: 1300)."
    )
    llm_group.add_argument(
        "--max-sentence-length", type=int, default=120, help="Maximum sentence length for analysis (default: 120)."
    )

    # Color and display options
    color_group = parser.add_argument_group("Color and Display")
    color_group.add_argument(
        "--color-map",
        type=str,
        action="append",
        help="Customize colors, e.g., 'approach:#RRGGBBAA'. Can be specified multiple times.",
    )
    color_group.add_argument(
        "--color-legend",
        choices=["text", "ansi", "html"],
        nargs='?', 
        const='ansi',
        help="Show color legend and exit. Optionally specify format (text, ansi, html).",
    )

    # Verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("-q", "--quiet", action="store_true", help="Suppress all non-error output.")
    verbosity_group.add_argument("--debug", action="store_true", help="Enable detailed debug output.")
    verbosity_group.add_argument("-v", "--verbose", action="store_true", help="Enable progress bar (default).")
    
    return parser


def main() -> None:
    # Special handling for color-legend as it doesn't require other args
    if "--color-legend" in sys.argv:
        parser = build_parser("color_legend")
        args = parser.parse_args()
        try:
            color_map = make_color_map(DEFAULT_COLOR_MAP, args.color_map)
            print(color_legend_str(color_map, args.color_legend))
        except (InvalidColorMap, InvalidHexColorCode) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    parser = build_parser("normal")
    args = parser.parse_args()

    verbose = (not args.quiet and not args.debug) or args.verbose
    debug = args.debug

    try:
        color_map = make_color_map(DEFAULT_COLOR_MAP, args.color_map)
    except (InvalidColorMap, InvalidHexColorCode) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        filetype = detect_filetype(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = get_output_path(args.input, args.output, args.output_auto, overwrite=args.overwrite)

    extraction_mode = "s" if args.skim else "aetr"

    if filetype == "pdf":
        if output_path is None:
            print("Error: Output to stdout ('-o -') is not supported for PDF files.", file=sys.stderr)
            sys.exit(1)
        highlight_sentences_in_pdf(
            args.input,
            output_path,
            color_map=color_map,
            llm_backend=args.llm_backend,
            model=args.model,
            ollama_base_url=args.ollama_base_url,
            buffer_size=args.buffer_size,
            max_sentence_length=args.max_sentence_length,
            extraction=extraction_mode,
            verbose=verbose,
            debug=debug,
        )
    elif filetype == "md":
        highlight_sentences_in_md(
            args.input,
            output_path,
            color_map=color_map,
            llm_backend=args.llm_backend,
            model=args.model,
            ollama_base_url=args.ollama_base_url,
            buffer_size=args.buffer_size,
            max_sentence_length=args.max_sentence_length,
            extraction=extraction_mode,
            verbose=verbose,
            debug=debug,
        )

if __name__ == "__main__":
    main()
