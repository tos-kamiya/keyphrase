# Keyphrase

**keyphrase** is a command-line tool that automatically detects key phrases and important sentences in PDF or Markdown files using an LLM (Large Language Model) and annotates them with color highlights. It is designed for academic papers, technical documents, and any text where understanding the main points at a glance is helpful.

## Features

* Supports both **PDF** and **Markdown** (`.md`) files.
* AI-based detection and color-coding of:

  * **Proposed ideas/methods** (blue)
  * **Experimental results/summary** (green)
  * **Threats to validity / limitations** (yellow)
* Section headings can also be automatically detected and highlighted.
* Output is a new, annotated file with color-coded highlights.
* Flexible output filename options, with overwrite safety.
* Fast and private: all inference is done locally via Ollama.

## Installation

### 1. Install via pipx (recommended):

```bash
pipx install https://github.com/tos-kamiya/keyphrase
```

If you don't have `pipx`, you can install it with:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2. Install and set up Ollama

keyphrase uses [Ollama](https://ollama.com/) for local LLM inference.

Follow the instructions for your platform on the [official Ollama site](https://ollama.com/download).

### 3. Download the Qwen3:30b model for Ollama

You need to install the `qwen3:30b` model in your local Ollama server:

```bash
ollama pull qwen3:30b
```

## Usage

### Basic usage

```bash
keyphrase input.pdf
```

* For PDFs: creates `out.pdf` (if not present) with color highlights.

For Markdown files:

```bash
keyphrase input.md
```

* For Markdown: creates `out.md` with color highlights using `<span>` tags.

### Output filename options

* `-o OUTPUT`, `--output OUTPUT`: Specify the output file name.
  Use `-o -` to write output to standard output (stdout) (Markdown only).
* `-O`, `--output-auto`: Output to `INPUT-annotated.pdf` or `INPUT-annotated.md`.
* By default, the output will be `out.pdf` or `out.md`.
  If the file exists, an error will be raised unless `--overwrite` is specified.

### Batch/Buffering options

* `--buffer-size N`: Buffer size for batch LLM queries (in characters, default: 2000).
  Sentences are processed in batches for efficiency.

### Category voting options

* `-i N`, `--intersection-vote N`:
  For each buffer, query the LLM N times for each category and only keep sentences detected in **all** runs (intersection vote).
  (Default: `3`, which improves reliability but may reduce the number of detected key sentences.)

### Other options

* `-m MODEL`, `--model MODEL`: Specify the Ollama model to use (default: `qwen3:30b`).
* `--overwrite`: Overwrite output file if it already exists.

### Example

```bash
keyphrase paper.pdf -O
```

* Annotates `paper.pdf`, outputs as `paper-annotated.pdf`.

```bash
keyphrase notes.md -o highlights.md -i 5 --buffer-size 5000
```

* Annotates `notes.md`, outputs to `highlights.md`, using 5-vote intersection and a larger buffer size.

## Requirements

* Python 3.10 or newer
* [Ollama](https://ollama.com/) running locally
* Qwen3:30b model installed in Ollama (`ollama pull qwen3:30b`)

## License

MIT

## Notes

* This tool does **not** send any data to third-party APIs: all processing is local via Ollama.
* For best results on scientific papers, use high-quality, clean PDF or Markdown sources.
* Markdown output uses HTML `<span style="background-color:...">...</span>` for color highlights.
