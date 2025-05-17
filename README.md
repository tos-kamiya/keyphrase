# Keyphrase

**keyphrase** is a command-line tool that automatically detects key phrases and important sentences in PDF or Markdown files using an LLM (Large Language Model) and annotates them with color highlights. It is designed for academic papers, technical documents, and any text where understanding the main points at a glance is helpful.

**Example Outputs**

* [docs/icpc-2022-zhu-annotated.pdf](docs/icpc-2022-zhu-annotated.pdf)

* [docs/kbase-202405-kamiya-annotated.pdf](docs/kbase-202405-kamiya-annotated.pdf) (in Japanese)

## Features

* Supports both **PDF** and **Markdown** (`.md`) files
* AI-based detection and color-coding of:
  * **Approach/methodology** (blue): The main novelty or core contribution of the paper
  * **Experimental results** (green): Key observations and experimental outcomes
  * **Threats to validity / limitations** (yellow): Weaknesses or potential problems with the approach
* Output is a new, annotated file with color-coded highlights
* Flexible output filename options, with overwrite protection
* All inference is done locally via Ollama

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

### 3. Download the Qwen3 model for Ollama

You need to install the `qwen3:30b-a3b` model in your local Ollama server:

```bash
ollama pull qwen3:30b-a3b
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

### Other options

* `-m MODEL`, `--model MODEL`: Specify the Ollama model to use (default: `qwen3:30b-a3b`).
* `--max-sentence-length N`: Maximum length of each sentence for analysis (default: 80).
* `--overwrite`: Overwrite output file if it already exists.
* `--verbose`: Show progress bar with tqdm.

### Example

```bash
keyphrase paper.pdf -O
```

* Annotates `paper.pdf`, outputs as `paper-annotated.pdf`.

```bash
keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
```

* Annotates `notes.md`, outputs to `highlights.md`, using a larger buffer size, longer maximum sentence length, and showing progress.

## Requirements

* Python 3.10 or newer
* [Ollama](https://ollama.com/) running locally
* Qwen3:30b-a3b model installed in Ollama (`ollama pull qwen3:30b-a3b`)
* Required dependencies: blingfire, numpy, pymupdf, ollama, tqdm, pydantic

## License

MIT

## Notes

* This tool does **not** send any data to third-party APIs: all processing is local via Ollama.
* For best results on scientific papers, use high-quality, clean PDF or Markdown sources.
* Markdown output uses HTML `<span style="background-color:...">...</span>` for color highlights.
