
import pytest
import gensim.downloader as api
import numpy as np

from Assignment2 import get_word_vector, build_tag_matrix, TAGS


# Load the model once
model = api.load("glove-wiki-gigaword-50")


def test_emptyTags():
    res = get_word_vector(model, [])
    assert res == ([], [])


def test_valid_tag():
    dropped, vectors = get_word_vector(model, ["research"])

    assert dropped == []
    assert len(vectors) == 1
    assert len(vectors[0]) == 50


def test_missing_tag():
    dropped, vectors = get_word_vector(model, ["zzzxxyy"])

    assert "zzzxxyy" in dropped
    assert len(vectors) == 0


def test_number_of_tags():
    dropped, vectors = get_word_vector(model, TAGS)

    assert len(dropped) == 0
    assert len(vectors) == 20


def test_tag_matrix_shape():
    T = build_tag_matrix(model, TAGS)

    assert T.shape == (20, 50)


def test_invalid_tag_raises_error():
    tags = TAGS.copy()
    tags[0] = "zzzxxyy"

    with pytest.raises(ValueError):
        build_tag_matrix(model, tags)


def test_wrong_number_of_tags():
    with pytest.raises(ValueError):
        build_tag_matrix(model, ["research"])


def test_empty_matrix_tags():
    with pytest.raises(ValueError):
        build_tag_matrix(model, [])

