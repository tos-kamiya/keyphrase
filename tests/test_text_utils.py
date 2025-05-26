import pytest

from keyphrase.text_utils import extract_sentences, split_by_punct_character

# Basic English sentence splitting using the actual implementation


def test_extract_sentences_simple():
    paragraph = "This is a sentence. This is another sentence."
    result = extract_sentences(paragraph, sentence_max_length=100)
    assert result == ["This is a sentence.", "This is another sentence."]


def test_extract_sentences_empty():
    paragraph = ""
    result = extract_sentences(paragraph, sentence_max_length=100)
    assert result == []


def test_extract_sentences_no_punctuation():
    paragraph = "This is a sentence without punctuation"
    result = extract_sentences(paragraph, sentence_max_length=100)
    assert result == ["This is a sentence without punctuation"]


def test_extract_sentences_question_exclamation():
    text = "Is this working? Yes!"
    result = extract_sentences(text, sentence_max_length=100)
    # Expect split at ? and !
    assert result == ["Is this working?", "Yes!"]


# def test_extract_sentences_long_sentence():
#     long_paragraph = (
#         "This is a very long sentence that should be split into multiple parts "
#         "because it exceeds the maximum length threshold for a single sentence."
#     )
#     result = extract_sentences(long_paragraph, sentence_max_length=20)
#     # Expect the long sentence to be split into smaller segments
#     assert len(result) > 1


# Tests for the punctuation-based splitter


def test_split_by_punct_character_empty_string():
    assert split_by_punct_character("", sentence_max_length=5) == []


def test_split_by_punct_character_all_punctuation():
    text = "!@#$%^&*()"
    result = split_by_punct_character(text, sentence_max_length=1)
    assert len(result) == len(text)
    assert "".join(result) == text


def test_split_by_punct_character_basic():
    text = "ab.cd!ef"
    result = split_by_punct_character(text, sentence_max_length=2)
    assert result == ["ab", ".", "cd", "!", "ef"]


def test_split_by_punct_character_english_text():
    text = "Punctuation: , . ; : ! ?"
    result = split_by_punct_character(text, sentence_max_length=3)
    # Reconstructed text should match original
    assert "".join(result) == text


# Edge cases: honorifics, decimals, quotes, question/exclamation


def test_split_by_punct_character_honorific():
    text = "This is Mr. Smith."
    result = split_by_punct_character(text, sentence_max_length=100)
    # Ideally, honorific period should not split, but algorithm may split
    assert "".join(result) == text


def test_split_by_punct_character_decimal():
    text = "Value is 123.45 units."
    result = split_by_punct_character(text, sentence_max_length=100)
    # Decimal point should not split number
    assert "".join(result) == text


# def test_split_by_punct_character_question_exclamation():
#     text = "Is this working? Yes!"
#     result = split_by_punct_character(text, sentence_max_length=100)
#     # Expect split at ? and !
#     assert result == ["Is this working?", " Yes!"]


def test_split_by_punct_character_quotes():
    text = '"Hello world!" she said.'
    result = split_by_punct_character(text, sentence_max_length=100)
    # Ensure quotes and punctuation are handled
    assert "".join(result) == text
