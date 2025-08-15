# Keyphrase

**Keyphrase** is a command-line tool that finds key sentences in PDF or Markdown files with the help of a local LLM and highlights them by category. It’s designed for academic papers, technical documents, and any text where you want the main points to pop at a glance.

**Example outputs**

* [docs/icpc-2022-zhu-annotated.pdf](docs/icpc-2022-zhu-annotated.pdf)
* [docs/kbase-202405-kamiya-annotated.pdf](docs/kbase-202405-kamiya-annotated.pdf) (Japanese)

## What’s new

* **Two-pass extraction with refinement**: we run the LLM twice. The second pass receives the first JSON result and refines it (dedup, fix mislabels, prefer fewer/stronger sentences). If validation fails, we safely fall back to pass-1.
* **Two backends**:

  * `harmony` (default): talks to an Ollama server using the “Harmony” JSON pattern with strict Pydantic validation.
  * `ollama`: uses Ollama’s native JSON mode.
* **Color legend generator**: `--color-legend [text|ansi|html]` prints a legend and exits—handy when tweaking colors.
* **Markdown safety checks**: base64-embedded images in `.md` are rejected with a clear error.

## Features

* Works with **PDF** and **Markdown** (`.md`)
* AI-based detection and color-coding of key concepts (defaults shown):

  * <span style="display:inline-block;width:40px;height:20px;background:#8edefbb0;"></span> **Approach / methodology** (blue)
  * <span style="display:inline-block;width:40px;height:20px;background:#d0fbb1b0;"></span> **Experimental results** (green)
  * <span style="display:inline-block;width:40px;height:20px;background:#fec6afb0;"></span> **Threats to validity** (pink)
* Optional **skim mode** for survey-style documents (single highlight class)
* **Customizable highlight colors** per category
* **Automatic output naming** with overwrite protection
* **Local-only inference** via your own Ollama server

> Note: the model also predicts a **`reference`** set (useful sentences that act like citations/anchors). These are **not highlighted** in the output; they inform the selection only.

## Installation

### 1) Install with pipx (recommended)

```bash
pipx install git+https://github.com/tos-kamiya/keyphrase.git
```

If you don’t have `pipx` yet:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2) Install and run Ollama

Keyphrase uses a **local** Ollama server.

### 3. Download the gpt-oss model for Ollama

### 3) Pull a model

Default examples use `gpt-oss:20b`:

```bash
ollama pull gpt-oss:20b
```

You can choose a different model with `-m/--model`.

## Usage

### Quick start

#### PDF

```bash
keyphrase input.pdf
```

* Produces `out.pdf` (unless it exists). Use `--overwrite` to replace.

#### Markdown

```bash
keyphrase input.md
```

* Produces `out.md` with `<span style="background-color:...">...</span>`.

* You may also write Markdown to stdout:

```bash
keyphrase input.md -o -
```

*(PDF → stdout is not supported.)*

### Backends and models

* `--llm-backend {harmony,ollama}` (default: `harmony`)

  * `harmony`: strict JSON validation (Pydantic) with a Harmony-style prompt.
  * `ollama`: Ollama’s native `format=json`.

* `-m, --model MODEL` (default: `gpt-oss:20b`)

* `--ollama-base-url URL` (default: `http://localhost:11434`)

**Examples**

```bash
# Harmony backend (default) with 20B model
keyphrase paper.pdf --llm-backend harmony -m gpt-oss:20b

# Ollama-native JSON backend
keyphrase notes.md --llm-backend ollama -m gpt-oss:20b -o highlights.md
```

### Modes

* **Default (categorized)**: picks sentences into `approach`, `experiment`, `threat` (and `reference` internally).
* **Skim mode**: a single class of important sentences.

```bash
keyphrase survey.pdf --skim -O
```

### Output options

* `-o, --output PATH` : specify output file. Use `-o -` for **Markdown to stdout**.
* `-O, --output-auto` : write to `INPUT-annotated.ext`.
* Default names are `out.pdf` / `out.md`.
* `--overwrite` : allow overwrite if the file exists.

### Color options

Fine-tune and preview your colors.

* `--color-map "name:#RRGGBBAA"` (or `#RGBA`). Multiple allowed.
* Categories: `approach`, `experiment`, `threat`
* Disable a category with `name:0`

**Examples**

```bash
# Change colors
keyphrase paper.pdf \
  --color-map approach:#ffcc00ff \
  --color-map experiment:#44cc99ff \
  --color-map threat:#cc66cc88

# Disable threat highlighting
keyphrase paper.pdf --color-map threat:0
```

#### Legends

```bash
keyphrase --color-legend           # same as --color-legend=ansi
keyphrase --color-legend text
keyphrase --color-legend ansi
keyphrase --color-legend html

# Preview with your tweaks
keyphrase --color-legend ansi \
  --color-map approach:#ffcc00ff \
  --color-map experiment:#44cc99ff
```

### Performance & limits

* `--buffer-size N` (default: 3000 chars):
  sentences are batched until the character budget is reached, then sent to the LLM.
* `--max-sentence-length N` (default: 120):
  very long sentences are truncated for analysis speed.
* `--timeout SEC` (default: 300):
  increase if your model is slow/timeouts occur.

### Verbosity

* `-q, --quiet` : suppress non-error output.
* `--debug` : detailed debug logs; shows warnings and refine fallbacks.
* `-v, --verbose` : progress bars (this is the default unless `--quiet`/`--debug`).

## CLI reference

```
keyphrase INPUT
  [--llm-backend {harmony,ollama}]
  [-m MODEL]
  [--ollama-base-url URL]
  [--skim]
  [--buffer-size N]
  [--max-sentence-length N]
  [--timeout SEC]
  [-o OUTPUT | -O]
  [--overwrite]
  [--color-map name:#RRGGBBAA]...
  [--color-legend [text|ansi|html]]
  [-q | --debug | -v]
```

**Notes**

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

* `-m MODEL`, `--model MODEL`: Specify the Ollama model to use (default: `gpt-oss:20b`)
* `--max-sentence-length N`: Maximum sentence length for analysis (default: 120)
* `--buffer-size N`: Buffer size for batch LLM queries (in characters, default: 3000).
  Sentences are processed in batches for efficiency.
* `--timeout N`: Timeout in seconds for the LLM response (default: 300).  
  Increase this value if timeouts occur frequently.

### More usage examples

```bash
keyphrase paper.pdf -O
# -> Annotates 'paper.pdf', outputs as 'paper-annotated.pdf'

keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
# -> Annotates 'notes.md', outputs to 'highlights.md', using a larger buffer, longer sentences, and showing progress.
```

## Requirements

* Python 3.10+
* [Ollama](https://ollama.com/) running locally
* gpt-oss:20b model installed in Ollama (`ollama pull gpt-oss:20b`)

## License

MIT

## Privacy

All LLM inference is performed **locally** via your Ollama server. No content is sent to third-party APIs.

## Tips

* For dense scientific PDFs, higher `--buffer-size` can reduce LLM round-trips.
* If you only need a quick gist, `--skim` often produces a compact set of highlights.
* `--color-legend html` is convenient for pasting into docs/wiki pages.
