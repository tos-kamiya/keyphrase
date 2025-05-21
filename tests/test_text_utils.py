import pytest

import sys
from unittest.mock import patch

from keyphrase.text_utils import extract_sentences, split_by_punct_character

def test_extract_sentences_simple():
    dummy_paragraph = "This is a sentence. This is another sentence."
    def fake_text_to_sentences(text):
        return text.replace(".", ".\n")
    with patch("blingfire.text_to_sentences", side_effect=fake_text_to_sentences):
        result = extract_sentences(dummy_paragraph, 100)
        assert result == ["This is a sentence.", "This is another sentence."]

def test_split_by_punct_character():
    result = split_by_punct_character("ab.cd!ef", 2)
    assert result == ["ab", ".", "cd", "!", "ef"]

def test_split_by_punct_character_english():
    test_cases = [
        "This is a (test).",
        "Punctuation: , . ; : ! ? - … ( ) [ ] { } \" '",
        "Multiple brackets (like these), and {curly} ones.",
        "Quotes: 'single', \"double\", `backtick`.",
        "Ellipsis... and dash—yes!",
        "Math: 3.14 * (2 + 5) = 21.98",
        "(Only parentheses)",
        "\"Just quotes.\"",
        "",
        "...?!"
    ]
    for text in test_cases:
        result = split_by_punct_character(text, sentence_max_length=5)
        assert "".join(result) == text

def test_split_by_punct_character_japanese():
    test_cases = [
        "これは（テスト）です。",
        "次の記号：、。『』「」【】（）；？！―…",
        "括弧（やカギカッコ『』も）正しく扱いたい。",
        "複数の区切り。カンマ、ピリオド。",
        "日本語：英語も混在。",
        "（カッコのみ）",
        "「引用符」や“ダブルクォート”も。",
        "【特殊記号】《例》〈テスト〉",
        "",
        "…、、、？！"
    ]
    for text in test_cases:
        result = split_by_punct_character(text, sentence_max_length=5)
        assert "".join(result) == text

def test_split_by_punct_character_mixed():
    test_cases = [
        "日本語とEnglish (mixed) テスト。",
        "（これは a “mixed” test！）",
        "符号(English)、カンマ, ピリオド.",
        "AIによる“自動生成” (automatic generation)."
    ]
    for text in test_cases:
        result = split_by_punct_character(text, sentence_max_length=5)
        assert "".join(result) == text
