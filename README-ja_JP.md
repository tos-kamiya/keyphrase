# Keyphrase

**keyphrase**は、PDFやMarkdownファイルから、LLM（大規模言語モデル）を用いてキーフレーズや重要文を自動検出し、色分けハイライト付きで注釈を行うコマンドラインツールです。学術論文や技術文書など、主要なポイントを一目で把握したいシーンに最適です。

## 特徴

* **PDF**・**Markdown**（`.md`）ファイルの両方に対応
* AIによる自動判別と色分け
  * **提案手法・主要アイデア**（青色でハイライト）：論文の主要な新規性や核となる貢献
  * **実験・評価結果**（緑色でハイライト）：主要な観察結果と実験的成果
  * **妥当性の脅威・制約**（黄色でハイライト）：アプローチや結果に関する弱点や潜在的問題点
* 色分けされたハイライト付きの新しいファイルを出力
* 柔軟な出力ファイル名指定、既存ファイルの上書き防止
* すべてローカル推論：Ollamaを利用

## インストール

### 1. pipxでのインストール（推奨）

```bash
pipx install https://github.com/tos-kamiya/keyphrase
```

`pipx`が未導入の場合は以下でインストールできます：

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2. Ollamaの導入

keyphraseはローカルLLM推論のため[Ollama](https://ollama.com/)を利用します。

お使いのOSごとに[公式サイト](https://ollama.com/download)の手順でOllamaをセットアップしてください。

### 3. Qwen3モデルのダウンロード

Ollamaで`qwen3:30b-a3b`モデルをインストールしてください：

```bash
ollama pull qwen3:30b-a3b
```

## 使い方

### 基本的な使い方

```bash
keyphrase input.pdf
```

* PDFの場合：色分けハイライト付きの`out.pdf`（未作成なら）を出力します。

Markdownの場合：

```bash
keyphrase input.md
```

* Markdownの場合：`out.md`を出力し、各文に`<span>`タグで色付きハイライトを付与します。

### 出力ファイル名のオプション

* `-o OUTPUT`, `--output OUTPUT`：出力ファイル名を指定。`-o -`とすると標準出力（stdout）に出力（Markdownのみ対応）。
* `-O`, `--output-auto`：`INPUT-annotated.pdf`や`INPUT-annotated.md`として自動命名出力
* デフォルトは`out.pdf`または`out.md`。同名ファイルが既にある場合は`--overwrite`指定がないとエラー

### バッチ・バッファ処理

* `--buffer-size N`：LLMへまとめて問い合わせる文バッファの最大文字数（デフォルト：2000）。バッファ単位で処理することで効率向上。

### その他のオプション

* `-m MODEL`, `--model MODEL`：使用するOllamaモデルを指定（デフォルトは`qwen3:30b-a3b`）
* `--max-sentence-length N`：分析対象の文の最大長を指定（デフォルト：80）
* `--overwrite`：既存の出力ファイルを上書き
* `--verbose`：tqdmを使用して進捗バーを表示

### 使用例

```bash
keyphrase paper.pdf -O
```

* `paper.pdf`を色分け注釈し、`paper-annotated.pdf`として出力

```bash
keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
```

* `notes.md`を`highlights.md`として出力し、バッファ5000文字、文の最大長100文字で処理、進捗バーを表示

## 必要要件

* Python 3.10 以上
* ローカルで動作する [Ollama](https://ollama.com/)
* Ollamaで`qwen3:30b-a3b`モデルインストール済み（`ollama pull qwen3:30b-a3b`）
* 必要な依存関係: blingfire, numpy, pymupdf, ollama, tqdm, pydantic

## ライセンス

MIT

## 注意

* 本ツールは外部API等へのデータ送信を一切行わず、すべてローカルOllamaで処理されます。
* 学術論文などでの利用時は、なるべくレイアウトが整理された高品質なPDF・Markdownをご利用ください。
* Markdown出力は各文を`<span style="background-color:...">...</span>`で色付けします。
