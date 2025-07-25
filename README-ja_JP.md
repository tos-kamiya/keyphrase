# Keyphrase

**keyphrase**は、PDFやMarkdownファイルから、LLM（大規模言語モデル）を用いてキーフレーズや重要文を自動検出し、色分けハイライト付きで注釈を行うコマンドラインツールです。
学術論文や技術文書など、主要なポイントを一目で把握したい場面に最適です。

**出力例**

* [docs/icpc-2022-zhu-annotated.pdf](docs/icpc-2022-zhu-annotated.pdf)（英文）
* [docs/kbase-202405-kamiya-annotated.pdf](docs/kbase-202405-kamiya-annotated.pdf)（日本語）

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

* `input.pdf` を注釈し、`out.pdf`（未作成なら）として出力します。

Markdownの場合：

```bash
keyphrase input.md
```

* `input.md` を注釈し、`out.md` として HTML の `<span>` タグでハイライトを付けて出力します。

### 出力オプション

* `-o OUTPUT`, `--output OUTPUT`：出力ファイル名を指定
  `-o -` を指定すると標準出力へ出力（Markdownのみ対応）

* `-O`, `--output-auto`：`INPUT-annotated.pdf` または `INPUT-annotated.md` として自動命名

* デフォルトは `out.pdf` または `out.md`
  同名ファイルがすでに存在する場合、`--overwrite` がないとエラーになります。

* `--overwrite`：出力ファイルが既に存在していても上書きします。

### 色に関するオプション

カテゴリごとのハイライト色は自由にカスタマイズ可能で、プレビューも行えます。

#### ハイライト色のカスタマイズ

* `--color-map` オプションで各カテゴリに色を割り当てます。
* **形式**：`name:#rgba` または `name:#rrrggbbaa`（例：`approach:#8edefbb0`）
* **指定可能なカテゴリ名**：`approach`, `experiment`, `threat`
* 特定のマーカーを無効化するには `name:0` を指定（例：`threat:0`）
* このオプションは複数回指定できます。

**使用例：**

```bash
# approach を黄色に、experiment をティールに、threat を無効化
keyphrase input.pdf --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff --color-map threat:0
```

#### 色の凡例（レジェンド）表示

現在有効な色の設定をターミナルで確認できます。

```bash
keyphrase --color-legend text   # プレーンテキストで表示
keyphrase --color-legend ansi   # ANSIカラーで背景＋黒文字表示（24ビット端末推奨）
keyphrase --color-legend html   # HTMLテーブル形式で表示（ドキュメント貼り付け用）
```

`--color-map` と組み合わせてカスタム設定のプレビューが可能です：

```bash
keyphrase --color-legend ansi --color-map approach:#ffcc00ff --color-map experiment:#44cc99ff
```

### スキムモード（要点抽出； 実験中）

* `--skim`：サーベイ論文など、問題→手法→実験という典型的な構成でないペーパー向けの簡易強調モード。
  文をカテゴリ別に色分けせず、重要な文のみを単色でハイライトします。

### ログ・詳細表示に関するオプション

* `-q`, `--quiet`：すべての出力や進捗表示を抑制します
* `--debug`：デバッグ出力を有効に（プロンプトやレスポンス表示、進捗バーも含む）
* `--verbose`：進捗バーを表示（デフォルトの動作）

### モデル・バッチ処理オプション

* `-m MODEL`, `--model MODEL`：使用する Ollama モデル（デフォルト：`qwen3:30b-a3b`）
* `--max-sentence-length N`：解析対象となる1文あたりの最大文字数（デフォルト：80）
* `--buffer-size N`：バッチ処理時の最大文字数（デフォルト：2000）

### 使用例

```bash
keyphrase paper.pdf -O
# → paper.pdf を注釈し、paper-annotated.pdf として出力

keyphrase notes.md -o highlights.md --buffer-size 5000 --max-sentence-length 100 --verbose
# → notes.md を highlights.md として出力し、
# バッファを5000文字、最大文長100に設定、進捗表示付きで処理
```

## 必要要件

* Python 3.10 以上
* [Ollama](https://ollama.com/)（ローカルで稼働）
* Ollamaで`qwen3:30b-a3b`モデルインストール済み（`ollama pull qwen3:30b-a3b`）

## ライセンス

MIT

## 注意事項

* 本ツールは外部API等へのデータ送信を一切行わず、すべてローカルOllamaで処理されます。
* 学術論文等では、できるだけレイアウトが整った高品質なPDFやMarkdownをご利用ください。
* Markdown出力では、各文が`<span style="background-color:...">...</span>`で色付けされます。
