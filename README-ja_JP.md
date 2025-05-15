# Keyphrase

**keyphrase**は、PDFやMarkdownファイルから、LLM（大規模言語モデル）を用いて自動的にキーフレーズや重要文を検出し、色分けされたハイライトで注釈を付与するコマンドラインツールです。学術論文や技術文書から、主要なポイントを素早く抽出できるように設計されています。

## 特徴

* **PDF**および**Markdown**（`.md`）ファイルの両方に対応
* AIによる自動判別で以下の内容を検出
  * 提案手法や新しいアイデア（青色でハイライト）
  * 実験や結果の要約（緑色でハイライト）
  * 妥当性の脅威や制約事項（黄色でハイライト）
* 色分けされたハイライト付きの新しいファイルを出力
* `tqdm`によるプログレスバー表示（`--verbose`オプション）
* 柔軟な出力ファイル名の指定が可能

## インストール方法

### 1. pipxでのインストール（推奨）

```bash
pipx install https://github.com/tos-kamiya/keyphrase
```

`pipx`が未インストールの場合は、以下で導入してください：

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

### 2. **Ollamaのインストールとセットアップ**

keyphraseはローカルLLM推論のため[Ollama](https://ollama.com/)を利用します。

お使いのプラットフォームに合わせて[公式サイト](https://ollama.com/download)の手順でOllamaを導入してください。

### 3. **OllamaにQwen3:30bモデルをダウンロード**

ローカルOllamaサーバーで`qwen3:30b`モデルのインストールが必要です：

```bash
ollama pull qwen3:30b
```

## 使い方

### 基本的な使い方

```bash
keyphrase input.pdf
```

* `out.pdf`（未作成なら）に色付きハイライトを付与して出力します。

Markdownファイルの場合：

```bash
keyphrase input.md
```

* `<span>`タグを使って色付きハイライトを付与した`out.md`が出力されます。

### 出力ファイル名のオプション

* `-o OUTPUT` / `--output OUTPUT`：出力ファイル名を指定します。`-o -`（ハイフン1つ）とすると標準出力（stdout）に書き出します。
* `-O`：`INPUT-annotated.pdf`または`INPUT-annotated.md`として出力
* デフォルトでは`out.pdf`や`out.md`となります。同名のファイルが既に存在する場合は`--overwrite`を指定しない限りエラーになります。

### その他のオプション

* `--majority-vote`：各段落・各カテゴリごとに3回LLMで判定し、多数決で決定します（安定性向上用、ただしアイデア検出数がやや少なくなる傾向あり）。
* `--verbose`：tqdmによるプログレスバーを表示
* `-m MODEL`, `--model MODEL`：使用するOllamaモデルの指定（デフォルトは`qwen3:30b`）
* `--overwrite`：出力ファイルが既に存在する場合に上書き

### 使用例

```bash
keyphrase paper.pdf -O --verbose
```

* `paper.pdf`を色分けで注釈し、`paper-annotated.pdf`として出力、進捗バー付きで実行します。

## 必要要件

* Python 3.10以上
* ローカルで稼働する [Ollama](https://ollama.com/)
* Ollamaで`qwen3:30b`モデルをインストール済み（`ollama pull qwen3:30b`）

## ライセンス

MIT

## 注意

* このツールは外部APIにデータを送信せず、すべてローカルのOllama上で処理します。
* 学術論文などの解析には、なるべく高品質で整理されたPDFやMarkdownファイルをお使いください。


