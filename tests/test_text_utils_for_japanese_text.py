import pytest

from keyphrase.text_utils import extract_sentences, split_by_punct_character


@pytest.mark.parametrize(
    "text",
    [
        "これは（テスト）です。",
        "次の記号：、。『』「」【】（）；？！―…",
        "括弧（やカギカッコ『』も）正しく扱いたい。",
        "複数の区切り。カンマ、ピリオド。",
        "日本語：英語も混在。",
        "（カッコのみ）",
        "「引用符」や“ダブルクォート”も。",
        "【特殊記号】《例》〈テスト〉",
        "",
        "…、、、？！",
    ],
)
def test_split_by_punct_character_japanese(text):
    result = split_by_punct_character(text, sentence_max_length=10)
    assert "".join(result) == text


def test_extract_sentences_simple_japanese():
    paragraph = "これは、１つ目の文です。これは別の文です。"
    result = extract_sentences(paragraph, sentence_max_length=100)
    assert result == ["これは、１つ目の文です。", "これは別の文です。"]


def test_extract_sentences_question_exclamation_japanese():
    text = "これは機能しますか？はい！"
    result = extract_sentences(text, sentence_max_length=100)
    # Expect split at ？ and ！
    assert result == ["これは機能しますか？", "はい！"]
