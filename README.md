# Keyphrase

**keyphrase** is a command-line tool that automatically detects key phrases and important sentences in PDF or Markdown files using an LLM (Large Language Model) and annotates them with color highlights. It is designed for academic papers, technical documents, and any text where understanding the main points at a glance is helpful.

## Features

* Supports both **PDF** and **Markdown** (`.md`) files
* AI-based detection of:
  * Proposed ideas/methods (blue)
  * Experimental results/summary (green)
  * Threats to validity / limitations (yellow)
* Output is a new, annotated file with color-coded highlights
* Progress bar with `tqdm` (`--verbose` option)
* Flexible output filename options

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

### 2. **Install and set up Ollama**

keyphrase uses [Ollama](https://ollama.com/) for local LLM inference.

Install Ollama by following the instructions for your platform on the [official site](https://ollama.com/download).

### 3. **Download the Qwen3:30b model for Ollama**

You will need to install the `qwen3:30b` model in your local Ollama server:

```bash
ollama pull qwen3:30b
```

## Usage

### Basic usage

```bash
keyphrase input.pdf
```

* The tool will create `out.pdf` (if not present) with color highlights.

For Markdown files:

```bash
keyphrase input.md
```

* The tool will create `out.md` with color highlights using `<span>` tags.

### Output filename options

* `-o OUTPUT` / `--output OUTPUT`: Specify the output file name. Use `-o -` to write output to standard output (stdout).
* `-O`: Output to `INPUT-annotated.pdf` or `INPUT-annotated.md`.
* By default, the output will be `out.pdf` or `out.md`. If the file exists, it will raise an error unless `--overwrite` is specified.

### Other options

* `--majority-vote`: For each paragraph and category, the LLM is queried three times and the results are decided by majority vote (for improved stability, but note that this may slightly reduce the number of detected ideas).
* `--verbose`: Show progress bar using tqdm.
* `-m MODEL`, `--model MODEL`: Specify the Ollama model to use (default: `qwen3:30b`).
* `--overwrite`: Overwrite output file if it already exists.

### Example

```bash
keyphrase paper.pdf -O --verbose
```

* This will annotate `paper.pdf`, output as `paper-annotated.pdf`, with a progress bar.

## Requirements

* Python 3.10+
* [Ollama](https://ollama.com/) running locally
* Qwen3:30b model installed in Ollama (`ollama pull qwen3:30b`)

## License

MIT

## Notes

* This tool does not send any data to third-party APIs: all processing is local via Ollama.
* For best results on scientific papers, use high-quality, clean PDF or Markdown sources.
