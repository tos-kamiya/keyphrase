# Keyphrase

**Keyphrase** は、PDFやMarkdownファイルから、LLM（大規模言語モデル）を用いて重要文を自動検出し、カテゴリごとに色分けハイライトを付けるコマンドラインツールです。
学術論文や技術文書など、主要なポイントを一目で把握したい場面に最適です。

**出力例**

* [docs/icpc-2022-zhu-annotated.pdf](docs/icpc-2022-zhu-annotated.pdf)（英文）
* [docs/kbse-202405-kamiya-annotated.pdf](docs/kbse-202405-kamiya-annotated.pdf)（日本語）

## 新機能

* **二段階抽出（refine付き）**：
  1回目の結果をJSONとして取得し、2回目でその結果を踏まえて修正・洗練します（重複除去や誤分類修正、要点の絞り込み）。もし検証に失敗した場合は1回目の結果にフォールバックします。
* **2種類のバックエンド**：

  * `harmony`（デフォルト）：Harmony形式のプロンプト＋厳密なJSONバリデーション（Pydantic）
  * `ollama`：OllamaネイティブのJSONモード
* **カラー凡例出力**： `--color-legend [text|ansi|html]` で凡例を表示して終了。色調整の確認に便利です。
* **Markdown安全チェック**：base64埋め込み画像を含むMarkdownはサポート外とし、明示的にエラーを返します。

## 特徴

* **PDF** および **Markdown** (`.md`) ファイルに対応
* LLMによる自動判別と色分けハイライト（デフォルト設定）:

  * <span style="display:inline-block;width:40px;height:20px;background:#8edefbb0;"></span> **提案手法・主要アイデア**（青）
  * <span style="display:inline-block;width:40px;height:20px;background:#d0fbb1b0;"></span> **実験・評価結果**（緑）
  * <span style="display:inline-block;width:40px;height:20px;background:#fec6afb0;"></span> **妥当性の脅威**（ピンク）
* **スキムモード**（--skim）：サーベイ論文などに有効。カテゴリ分けせず重要文のみを単色で強調
* **カテゴリごとの色指定**が可能
* **自動出力ファイル名生成**（既存ファイルは保護）
* **ローカル処理のみ**：Ollamaサーバーを利用（外部送信なし）

> 補足：モデルは `reference`（引用的な文脈文）も推定しますが、これはハイライトには使われず、抽出精度向上のために内部利用されます。

## インストール

### 1. pipxによるインストール（推奨）

```bash
pipx install git+https://github.com/tos-kamiya/keyphrase.git
```

`pipx`が未導入の場合：

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2. Ollamaの導入

Keyphrase はローカルの Ollama サーバーを利用します。
[公式サイト](https://ollama.com/download) の案内に従ってインストールしてください。

### 3. モデルの取得

デフォルトは `gpt-oss:20b` を利用します：

```bash
ollama pull gpt-oss:20b
```

他のモデルを指定する場合は `-m` オプションを使用します。

## 使い方

### 基本例

#### PDF

```bash
keyphrase input.pdf
```

* `input.pdf` を注釈し、`out.pdf` を出力（既存ファイルがある場合はエラー）

#### Markdown

```bash
keyphrase input.md
```

* `input.md` を注釈し、`out.md` として `<span style="background-color:...">...</span>` を挿入
* `-o -` で標準出力へ出力可能（PDFは非対応）

### バックエンドとモデル指定

* `--llm-backend {harmony,ollama}` （デフォルト：`harmony`）
* `-m MODEL`（デフォルト：`gpt-oss:20b`）
* `--ollama-base-url URL`（デフォルト：`http://localhost:11434`）

```bash
# Harmonyバックエンド（デフォルト）
keyphrase paper.pdf --llm-backend harmony -m gpt-oss:20b

# OllamaネイティブJSONバックエンド
keyphrase notes.md --llm-backend ollama -m gpt-oss:20b -o highlights.md
```

### 出力オプション

* `-o OUTPUT`, `--output OUTPUT`：出力ファイル指定
  `-o -` は **Markdownのみ** 標準出力へ
* `-O`, `--output-auto`： `INPUT-annotated.pdf` / `INPUT-annotated.md` として出力
* デフォルト： `out.pdf` / `out.md`
* `--overwrite`：既存ファイルを上書き

### 色指定オプション

* `--color-map name:#RRGGBBAA` 形式で指定（複数回可）
* カテゴリ：`approach`, `experiment`, `threat`
* 無効化：`name:0`

**例**

```bash
keyphrase input.pdf \
  --color-map approach:#ffcc00ff \
  --color-map experiment:#44cc99ff \
  --color-map threat:0
```

#### 色の凡例表示

```bash
keyphrase --color-legend           # デフォルトは ansi
keyphrase --color-legend text
keyphrase --color-legend ansi
keyphrase --color-legend html
```

カスタム指定のプレビューも可能：

```bash
keyphrase --color-legend ansi --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff
```

### パフォーマンス調整

* `--buffer-size N`（デフォルト 3000文字）：このサイズを超えたらLLMにバッチ送信
* `--max-sentence-length N`（デフォルト 120）：長い文は解析対象外に
* `--timeout 秒数`（デフォルト 300）：遅いモデルの場合は延長推奨

### ログ・詳細表示

* `-q`, `--quiet`：すべての出力や進捗表示を抑制します
* `--debug`：デバッグ出力を有効に（プロンプトやレスポンス表示、進捗バーも含む）
* `--verbose`：進捗バーを表示（デフォルトの動作）

### モデル・バッチ処理オプション

* `-m MODEL`, `--model MODEL`：使用する Ollama モデル（デフォルト：`gpt-oss:20b`）
* `--max-sentence-length N`：解析対象となる1文あたりの最大文字数（デフォルト：120）
* `--buffer-size N`：バッチ処理時の最大文字数（デフォルト：3000）
* `--timeout N`：LLMのレスポンスのタイムアウトの秒数（デフォルト：300）。タイムアウトが頻繁に起きるようなら大きくしてください

### 使用例

```bash
keyphrase paper.pdf -O
# → paper-annotated.pdf を出力

keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
```

## 必要要件

* Python 3.10 以上
* [Ollama](https://ollama.com/)（ローカルで稼働）
* Ollamaで`gpt-oss:20b`モデルインストール済み（`ollama pull gpt-oss:20b`）

## ライセンス

MIT

## 注意事項

* 外部API等への送信は一切なく、すべてローカルOllamaで処理されます。
* 学術論文の場合は、できるだけレイアウトが整った高品質なPDF/Markdownをご利用ください。
* Markdown出力は `<span style="background-color:...">...</span>` で装飾されます。
* Markdownに **base64埋め込み画像** (`data:image/...`) が含まれる場合はサポート外です。
