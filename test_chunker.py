import pytest
from chunker import chunk_text, clean_lines, chunk_by_lines


def test_chunks_overlap():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_clean_lines_removes_junk():
    text = "Hello\n•\n17.\n\nWorld"
    assert clean_lines(text) == ["Hello", "World"]


def test_chunk_by_lines_has_no_empty_chunks():
    chunks = chunk_by_lines("line one\nline two\nline three", max_chars=15)
    assert chunks
    assert all(c.strip() for c in chunks)
