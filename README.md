# Keyphrase

**keyphrase** is a command-line tool that automatically detects key phrases and important sentences in PDF or Markdown files using an LLM (Large Language Model) and annotates them with color highlights. It is designed for academic papers, technical documents, and any text where understanding the main points at a glance is helpful.

**Example Outputs**

* [docs/icpc-2022-zhu-annotated.pdf](docs/icpc-2022-zhu-annotated.pdf)
* [docs/kbase-202405-kamiya-annotated.pdf](docs/kbase-202405-kamiya-annotated.pdf) (in Japanese)

## Features

* Supports both **PDF** and **Markdown** (`.md`) files
* AI-based detection and color-coding of key concepts:

  * <span style="display:inline-block;width:40px;height:20px;background:#8edefbb0;"></span> **Approach/methodology** (blue): The main novelty or core contribution of the paper
  * <span style="display:inline-block;width:40px;height:20px;background:#d0fbb1b0;"></span> **Experimental results** (green): Key observations and experimental outcomes
  * <span style="display:inline-block;width:40px;height:20px;background:#fec6afb0;"></span> **Threats to validity** (pink): Weaknesses or potential problems with the approach
* Generates a new, annotated file with color-coded highlights
* Flexible output filename options, with overwrite protection
* All LLM inference is done locally via Ollama
* **Customizable highlight colors** for each category via command-line options

## Installation

### 1. Install via pipx (recommended)

```bash
pipx install git+https://github.com/tos-kamiya/keyphrase.git
```

If you don't have `pipx`:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2. Install and set up Ollama

Keyphrase uses [Ollama](https://ollama.com/) for local LLM inference.
Follow the instructions for your platform on the [official Ollama site](https://ollama.com/download).

### 3. Download the Qwen3 model for Ollama

Install the required model in your local Ollama server:

```bash
ollama pull qwen3:30b-a3b
```

## Usage

### Basic usage

For PDF:

```bash
keyphrase input.pdf
```

* Annotates `input.pdf`, outputs as `out.pdf` (if not present).

For Markdown:

```bash
keyphrase input.md
```

* Annotates `input.md`, outputs as `out.md` using HTML `<span>` tags for highlights.

### Output options

* `-o OUTPUT`, `--output OUTPUT`: Specify output file name.
  Use `-o -` to write output to standard output (Markdown only).
* `-O`, `--output-auto`: Output to `INPUT-annotated.pdf` or `INPUT-annotated.md`.
* By default, output will be `out.pdf` or `out.md`.
  If the file exists, an error is raised unless `--overwrite` is specified.
* `--overwrite`: Overwrite output file if it already exists

### Color options

You can fully customize and preview highlight colors for each category using the options below.

#### Customizing highlight colors

* Use `--color-map` to specify colors for each category.
* **Format:** `name:#rgba` or `name:#rrrggbbaa` (e.g., `approach:#8edefbb0`)
* **Available category names:** `approach`, `experiment`, `threat`
* To disable a specific marker, specify `name:0` (e.g., `threat:0`)
* This option can be used multiple times.

**Example:**

```bash
# Change 'approach' to yellow, 'experiment' to teal, and disable 'threat'
keyphrase input.pdf --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff --color-map threat:0
```

#### Checking your current color settings (legend output)

You can check the currently active highlight colors as a legend in your terminal.
This is especially useful when adjusting colors with `--color-map`.

```bash
keyphrase --color-legend text   # Show legend as plain text
keyphrase --color-legend ansi   # Show legend with 24-bit color blocks (background + black text)
keyphrase --color-legend html   # Show legend as a compact HTML table snippet
```

You can combine this with `--color-map` to preview your custom color settings:

```bash
keyphrase --color-legend ansi --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff
```

* **ANSI output** uses a background color block and black text for visibility (works best in 24-bit color terminals).
* **HTML output** can be copy-pasted into documentation.

### Skim mode (experimental)

* `--skim`: Enable skim mode, a simplified highlighting mode intended for survey papers
  (i.e., papers not following the typical problem → approach → experiment structure).
  Instead of categorizing sentences by type, this mode highlights only
  important sentences using a single highlight color.

### Logging and verbosity options

* `-q`, `--quiet`: Suppress all progress output and messages.

* `--debug`: Enable debug output (show prompts/responses) and progress bar.

* `--verbose`: Show progress bar (default behavior if no --quiet).

### Other options

* `-m MODEL`, `--model MODEL`: Specify the Ollama model to use (default: `qwen3:30b-a3b`)
* `--max-sentence-length N`: Maximum sentence length for analysis (default: 80)
* `--buffer-size N`: Buffer size for batch LLM queries (in characters, default: 2000).
  Sentences are processed in batches for efficiency.

### More usage examples

```bash
keyphrase paper.pdf -O
# -> Annotates 'paper.pdf', outputs as 'paper-annotated.pdf'

keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
# -> Annotates 'notes.md', outputs to 'highlights.md', using a larger buffer, longer sentences, and showing progress.
```

## Requirements

* Python 3.10 or newer
* [Ollama](https://ollama.com/) running locally
* Qwen3:30b-a3b model installed in Ollama (`ollama pull qwen3:30b-a3b`)

## License

MIT

## Notes

* No data is sent to any third-party APIs: all processing is local via Ollama.
* For best results on scientific papers, use high-quality, clean PDF or Markdown sources.
* Markdown output uses HTML `<span style="background-color:...">...</span>` for color highlights.
