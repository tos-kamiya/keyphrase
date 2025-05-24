# Keyphrase

**keyphrase**は、PDFやMarkdownファイルから、LLM（大規模言語モデル）を用いてキーフレーズや重要文を自動検出し、色分けハイライト付きで注釈を行うコマンドラインツールです。
学術論文や技術文書など、主要なポイントを一目で把握したい場面に最適です。

**出力例**

* [docs/icpc-2022-zhu-annotated.pdf](docs/icpc-2022-zhu-annotated.pdf)（英文）
* [docs/kbase-202405-kamiya-annotated.pdf](docs/kbase-202405-kamiya-annotated.pdf)（日本語）

**※** 出力例は最新バージョンのものではないので、配色が異なっています。

## 特徴

* **PDF**および**Markdown**（`.md`）ファイルに対応
* AI（LLM）による自動判別と色分けハイライト

  * <span style="display:inline-block;width:40px;height:20px;background:#8edefbb0;"></span> **提案手法・主要アイデア**（青）：論文の新規性や主要な貢献
  * <span style="display:inline-block;width:40px;height:20px;background:#d0fbb1b0;"></span> **実験・評価結果**（緑）：主要な観察結果や実験的成果
  * <span style="display:inline-block;width:40px;height:20px;background:#fec6afb0;"></span> **妥当性の脅威**（ピンク）：弱点や潜在的な問題点
* 色分けハイライト付きの新規ファイルを自動生成
* 出力ファイル名の柔軟な指定や既存ファイルの上書き防止
* すべてローカルで推論（Ollamaを使用）
* **カテゴリごとにハイライト色をカスタマイズ可能**

## インストール

### 1. pipxによるインストール（推奨）

```bash
pipx install https://github.com/tos-kamiya/keyphrase
```

`pipx`が未導入の場合：

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2. Ollamaの導入

keyphraseはローカル推論のため[Ollama](https://ollama.com/)を利用します。
公式サイトの[ダウンロードページ](https://ollama.com/download)に従いセットアップしてください。

### 3. Qwen3モデルのインストール

Ollamaで次のコマンドを実行し、必要なモデルを取得してください：

```bash
ollama pull qwen3:30b-a3b
```

## 使い方

### 基本的な使い方

PDFの場合：

```bash
keyphrase input.pdf
```

* `input.pdf`を注釈し、`out.pdf`（未作成なら）として出力

Markdownの場合：

```bash
keyphrase input.md
```

* `input.md`を注釈し、`out.md`としてHTML `<span>`タグでハイライト出力

### 出力ファイル名のオプション

* `-o OUTPUT`, `--output OUTPUT`：出力ファイル名を指定
  `-o -`を指定すると標準出力（Markdownのみ対応）
* `-O`, `--output-auto`：`INPUT-annotated.pdf`または`INPUT-annotated.md`として自動命名
* デフォルトは`out.pdf`または`out.md`
  既に同名ファイルが存在する場合は、`--overwrite`指定がないとエラー

### 色オプション

各カテゴリのハイライト色は自由にカスタマイズ・プレビューできます。

#### ハイライト色のカスタマイズ

* `--color-map` オプションで各カテゴリの色を指定できます。
* **書式**：`name:#rgba` または `name:#rrrggbbaa`（例：`approach:#8edefbb0`）
* **指定可能なカテゴリ名**：`approach`, `experiment`, `threat`
* 特定のマーカーを無効化したい場合は `name:0`（例：`threat:0`）
* このオプションは複数回指定できます。

**使用例：**

```bash
# approachを黄色、experimentをティール色、threatを無効化
keyphrase input.pdf --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff --color-map threat:0
```

#### 現在の色設定の確認（凡例出力）

`--color-map` で設定した色をターミナル上で確認できます。
色調整時に便利です。

```bash
keyphrase --color-legend text   # 凡例をプレーンテキストで表示
keyphrase --color-legend ansi   # 24ビットカラー（背景＋黒文字）で表示
keyphrase --color-legend html   # HTMLテーブル形式の凡例を表示（ドキュメント貼り付け用）
```

`--color-map` と組み合わせると、カスタム設定をプレビューできます：

```bash
keyphrase --color-legend ansi --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff
```

* **ANSI出力**は背景色ブロック＋黒文字（24ビットカラー端末で推奨）
* **HTML出力**はドキュメント等に貼り付け可能です

### バッチ／バッファ処理オプション

* `--buffer-size N`：LLMへ一括処理する文バッファの最大文字数（デフォルト：2000）

### その他のオプション

* `-m MODEL`, `--model MODEL`：利用するOllamaモデル名（デフォルト：`qwen3:30b-a3b`）
* `--max-sentence-length N`：分析する1文あたりの最大文字数（デフォルト：80）
* `--overwrite`：既存出力ファイルの上書き許可
* `--verbose`：tqdmによる進捗バー表示

### 使用例

```bash
keyphrase paper.pdf -O
# → paper.pdfを注釈し、paper-annotated.pdfとして出力

keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
# → notes.mdをhighlights.mdとして出力し、バッファ5000文字、最大文長100、進捗表示付きで処理
```

## 必要要件

* Python 3.10 以上
* [Ollama](https://ollama.com/)（ローカルで稼働）
* Ollamaで`qwen3:30b-a3b`モデルインストール済み（`ollama pull qwen3:30b-a3b`）
* 必要な依存ライブラリ: blingfire, numpy, pymupdf, ollama, tqdm, pydantic

## ライセンス

MIT

## 注意事項

* 本ツールは外部API等へのデータ送信を一切行わず、すべてローカルOllamaで処理されます。
* 学術論文等では、できるだけレイアウトが整った高品質なPDFやMarkdownをご利用ください。
* Markdown出力では、各文が`<span style="background-color:...">...</span>`で色付けされます。
