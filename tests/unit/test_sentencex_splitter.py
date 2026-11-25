"""``SentencexSplitter``: the sentences it finds, the spans it returns, the extra.

These run the real sentencex, which is in the dev install, and skip without it.
Only the guard on the extra and the fallback for pieces that do not tile the
document stand something in for the library.
"""

import sys
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from semantic_chunkers import ConsecutiveChunker, SentencexSplitter


@pytest.fixture
def sentencex():
    return pytest.importorskip("sentencex")


@pytest.fixture
def no_sentencex(monkeypatch):
    """Make ``import sentencex`` fail, as it does without the extra."""
    monkeypatch.setitem(sys.modules, "sentencex", None)


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            "Note: This only applies to version 2.",
            ["Note: This only applies to version 2."],
        ),
        (
            "Dr. Smith arrived at 3 p.m. He was late.",
            ["Dr. Smith arrived at 3 p.m.", "He was late."],
        ),
        (
            "Prof. Jones and Mr. Lee met Mrs. Hall.",
            ["Prof. Jones and Mr. Lee met Mrs. Hall."],
        ),
    ],
)
def test_colons_and_abbreviations_stay_inside_their_sentence(sentencex, doc, expected):
    assert SentencexSplitter()(doc) == expected


@pytest.mark.parametrize(
    "language, doc, expected",
    [
        (
            "fr",
            "Il a quitté Paris. Élise est restée. Ça va mieux.",
            ["Il a quitté Paris.", "Élise est restée.", "Ça va mieux."],
        ),
        (
            "de",
            "Er wohnt in München. Österreich ist nah. Über Nacht schneite es.",
            ["Er wohnt in München.", "Österreich ist nah.", "Über Nacht schneite es."],
        ),
        (
            "ru",
            "Москва большая. Это столица России. Мы были там.",
            ["Москва большая.", "Это столица России.", "Мы были там."],
        ),
        (
            "ja",
            "今日は晴れです。明日は雨でしょう。傘を持っていきます。",
            ["今日は晴れです。", "明日は雨でしょう。", "傘を持っていきます。"],
        ),
    ],
)
def test_sentences_that_do_not_start_with_an_ascii_capital_are_split(
    sentencex, language, doc, expected
):
    assert SentencexSplitter(language=language)(doc) == expected


def test_the_language_decides_which_abbreviations_apply(sentencex):
    # "3." is an ordinal in German and the end of a sentence in English.
    doc = "Er kam am 3. Oktober an. Dann ging er."

    assert SentencexSplitter(language="de")(doc) == [
        "Er kam am 3. Oktober an.",
        "Dann ging er.",
    ]
    assert SentencexSplitter(language="en")(doc) == [
        "Er kam am 3.",
        "Oktober an.",
        "Dann ging er.",
    ]


def test_the_language_field_is_what_sentencex_is_given(sentencex, monkeypatch):
    segment = Mock(wraps=sentencex.segment)
    monkeypatch.setattr(sentencex, "segment", segment)

    SentencexSplitter(language="pt-br")("O Sr. Silva chegou. Ele saiu.")

    segment.assert_called_once_with("pt-br", "O Sr. Silva chegou. Ele saiu.")


def test_splits_are_verbatim_substrings_and_spans_round_trip(sentencex):
    doc = "First one.  Second one!\n\n\tThird one?"
    splitter = SentencexSplitter()

    splits = splitter(doc)

    assert splits == ["First one.", "Second one!", "Third one?"]
    assert [doc[start:end] for start, end in splitter.spans(doc)] == splits


def test_offsets_count_characters_not_bytes(sentencex):
    # Accents, Cyrillic, kana and an emoji all take more than one UTF-8 byte.
    doc = "Ça marche 😀. Это тоже. 今日は晴れです。Done."
    splitter = SentencexSplitter()

    spans = splitter.spans(doc)

    assert spans == [(0, 12), (13, 22), (23, 31), (31, 36)]
    assert [doc[start:end] for start, end in spans] == [
        "Ça marche 😀.",
        "Это тоже.",
        "今日は晴れです。",
        "Done.",
    ]
    assert [doc[start:end] for start, end in spans] == splitter(doc)


def test_whitespace_around_sentences_is_trimmed(sentencex):
    assert SentencexSplitter().spans("  Leading.   Trailing.  ") == [(2, 10), (13, 22)]


def test_a_paragraph_break_ends_a_sentence_and_blank_lines_are_dropped(sentencex):
    doc = "Title\n\n\n  Body text. More.\n"

    assert SentencexSplitter()(doc) == ["Title", "Body text.", "More."]


def test_a_single_newline_without_punctuation_does_not_end_a_sentence(sentencex):
    doc = "A heading\nSome text here. More text."

    assert SentencexSplitter()(doc) == ["A heading\nSome text here.", "More text."]


@pytest.mark.parametrize("doc", ["", "   ", "\n\t \n"])
def test_empty_or_whitespace_input_gives_no_splits(sentencex, doc):
    assert SentencexSplitter()(doc) == []
    assert SentencexSplitter().spans(doc) == []


def test_a_misspelled_parameter_is_an_error_not_a_silent_english_split():
    with pytest.raises(ValidationError, match="lang"):
        SentencexSplitter(lang="ru")


def test_a_missing_install_names_the_extra(no_sentencex):
    with pytest.raises(ImportError, match=r"semantic-chunkers\[sentencex\]"):
        SentencexSplitter()("A sentence.")


def test_pieces_that_do_not_tile_the_document_are_found_in_it(sentencex, monkeypatch):
    # A sentencex that dropped the whitespace between sentences.
    monkeypatch.setattr(sentencex, "segment", Mock(return_value=["One.", "Two."]))
    doc = "One.\n\n Two."
    splitter = SentencexSplitter()

    assert splitter.spans(doc) == [(0, 4), (7, 11)]
    assert splitter(doc) == ["One.", "Two."]


def test_pieces_that_rewrite_the_document_give_splits_without_spans(
    sentencex, monkeypatch
):
    # A sentencex that normalised the text: nothing can be located, so there
    # are no spans, as `BaseSplitter.spans` promises, but the splits remain.
    monkeypatch.setattr(sentencex, "segment", Mock(return_value=["ONE. ", "TWO."]))
    splitter = SentencexSplitter()

    assert splitter.spans("One. Two.") == []
    assert splitter("One. Two.") == ["ONE.", "TWO."]


def test_it_plugs_into_a_chunker_as_the_splitter(sentencex):
    doc = "Les chats ronronnent. Élise dort.\n\nLa bourse a chuté."
    encoder = Mock(return_value=np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]]))
    chunker = ConsecutiveChunker(
        encoder=encoder, splitter=SentencexSplitter(language="fr"), score_threshold=0.9
    )

    chunks = chunker([doc])[0]

    encoder.assert_called_once_with(
        ["Les chats ronronnent.", "Élise dort.", "La bourse a chuté."]
    )
    assert [chunk.splits for chunk in chunks] == [
        ["Les chats ronronnent.", "Élise dort."],
        ["La bourse a chuté."],
    ]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 35), (35, 53)]
    assert "".join(chunk.content for chunk in chunks) == doc
