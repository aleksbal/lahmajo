# tests/test_processing.py
"""Unit tests for adaptive chunking document-type detection."""
import unittest

from langchain_core.documents import Document

from lahmajo.ingestion.processing import _detect_document_type


class TestDetectDocumentType(unittest.TestCase):
    """Test the structured-vs-long-form heuristic used to pick chunk size/overlap."""

    def test_empty_docs_is_not_structured(self):
        is_structured, stats = _detect_document_type([])
        self.assertFalse(is_structured)
        self.assertEqual(stats, {})

    def test_resume_with_bullets_is_structured(self):
        resume = Document(page_content=(
            "Jane Doe\nSoftware Engineer\n\n"
            "Experience:\n"
            "- Built scalable backend services\n"
            "- Led a team of 4 engineers\n"
            "- Migrated infra to Kubernetes\n\n"
            "Skills:\n"
            "- Python\n- Go\n- AWS\n"
        ))

        is_structured, stats = _detect_document_type([resume])

        self.assertTrue(is_structured)
        self.assertTrue(stats["is_short"])
        self.assertTrue(stats["is_list_heavy"] or stats["is_line_dense"])

    def test_long_form_prose_with_hyphens_is_not_structured(self):
        # Regression case for the old heuristic (`'-' in text[:500]`), which
        # misclassified ordinary hyphenated prose as a structured/bulleted doc.
        # This paragraph has hyphenated words and an em-dash-style "-" but no
        # actual list items and reads as flowing paragraphs.
        paragraph = (
            "The well-known result in state-of-the-art research - published "
            "over a decade ago - established a long-standing baseline for "
            "how self-supervised models generalize across domains. "
        )
        long_form = Document(page_content=paragraph * 20)

        is_structured, stats = _detect_document_type([long_form])

        self.assertFalse(is_structured)
        self.assertLess(stats["bullet_ratio"], 0.15)

    def test_large_document_is_not_structured_regardless_of_bullets(self):
        # Even a bullet-heavy document is treated as long-form once it's large
        # enough that granular CV-style chunking no longer makes sense.
        big_bulleted = Document(page_content="- A short bullet point line.\n" * 2000)

        is_structured, stats = _detect_document_type([big_bulleted])

        self.assertFalse(is_structured)
        self.assertFalse(stats["is_short"])

    def test_short_prose_paragraph_is_not_structured(self):
        prose = Document(page_content=(
            "This is a short piece of ordinary flowing prose that spans a "
            "couple of sentences and reads like a normal paragraph, not a "
            "list of bullet points or short structured lines at all. "
        ) * 3)

        is_structured, stats = _detect_document_type([prose])

        self.assertFalse(is_structured)


if __name__ == '__main__':
    unittest.main()
