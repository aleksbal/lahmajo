"""Tests for the cross-encoder rerank provider.

sentence-transformers is an optional extra and is NOT installed in CI, so every
test here injects a fake module into sys.modules rather than importing the real
one. That is deliberate: the suite must pass without the extra present.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from lahmajo.search import rerank_provider
from lahmajo.search.rerank_provider import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderRerankProvider,
    LLMRerankProvider,
    get_rerank_provider,
)


def _doc(content: str, source: str = "doc.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _fake_sentence_transformers(predict_return=None, predict_side_effect=None):
    """A stand-in sentence_transformers module whose CrossEncoder is a MagicMock."""
    model = MagicMock()
    if predict_side_effect is not None:
        model.predict.side_effect = predict_side_effect
    else:
        model.predict.return_value = predict_return

    module = MagicMock()
    module.CrossEncoder.return_value = model
    return module, model


class CrossEncoderTestCase(unittest.TestCase):
    def setUp(self):
        # The provider memoises loaded models by name; clear it so tests don't leak
        # a previously injected fake into each other.
        rerank_provider._cross_encoder_cache.clear()

    tearDown = setUp


class TestCrossEncoderRerank(CrossEncoderTestCase):
    def test_orders_by_score_descending(self):
        module, _model = _fake_sentence_transformers(predict_return=[0.1, 0.9, 0.5])
        docs = [_doc("a"), _doc("b"), _doc("c")]

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            ranked = CrossEncoderRerankProvider().rerank("q", docs, top_k=3)

        self.assertEqual([d.page_content for d, _ in ranked], ["b", "c", "a"])
        self.assertEqual([s for _, s in ranked], [0.9, 0.5, 0.1])

    def test_returns_the_models_real_scores_not_rank_positions(self):
        module, _model = _fake_sentence_transformers(predict_return=[7.25, -3.5])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            ranked = CrossEncoderRerankProvider().rerank("q", [_doc("a"), _doc("b")], top_k=2)

        # LLMRerankProvider synthesises 1.0, 0.5, ...; these must be the model's own.
        self.assertEqual([s for _, s in ranked], [7.25, -3.5])

    def test_scores_all_pairs_in_one_batched_call(self):
        module, model = _fake_sentence_transformers(predict_return=[0.1, 0.2, 0.3])
        docs = [_doc("a"), _doc("b"), _doc("c")]

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            CrossEncoderRerankProvider().rerank("my query", docs, top_k=3)

        model.predict.assert_called_once()
        pairs = model.predict.call_args[0][0]
        self.assertEqual(pairs, [("my query", "a"), ("my query", "b"), ("my query", "c")])

    def test_truncates_to_top_k(self):
        module, _model = _fake_sentence_transformers(predict_return=[0.1, 0.9, 0.5])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            ranked = CrossEncoderRerankProvider().rerank(
                "q", [_doc("a"), _doc("b"), _doc("c")], top_k=2
            )

        self.assertEqual(len(ranked), 2)
        self.assertEqual([d.page_content for d, _ in ranked], ["b", "c"])

    def test_empty_candidates_short_circuits(self):
        module, model = _fake_sentence_transformers(predict_return=[])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            ranked = CrossEncoderRerankProvider().rerank("q", [], top_k=5)

        self.assertEqual(ranked, [])
        model.predict.assert_not_called()

    def test_predict_failure_falls_back_to_original_order(self):
        module, _model = _fake_sentence_transformers(
            predict_side_effect=RuntimeError("CUDA out of memory")
        )
        docs = [_doc("a"), _doc("b"), _doc("c")]

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            ranked = CrossEncoderRerankProvider().rerank("q", docs, top_k=3)

        # Never raises out of rerank(), same posture as LLMRerankProvider.
        self.assertEqual([d.page_content for d, _ in ranked], ["a", "b", "c"])


class TestLazyModelLoading(CrossEncoderTestCase):
    """Weights load on the first rerank that has candidates, not at construction.

    get_rerank_provider() runs at the top of every retrieval, before it is known
    whether anything was found - so constructing the provider must not download or
    hold a model, and must not turn a no-results query into an error when the
    optional dependency or the network is missing.
    """

    def test_construction_does_not_load_the_model(self):
        module, _model = _fake_sentence_transformers(predict_return=[])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            CrossEncoderRerankProvider(model_name="m")

        module.CrossEncoder.assert_not_called()

    def test_reranking_nothing_does_not_load_the_model(self):
        module, _model = _fake_sentence_transformers(predict_return=[])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            ranked = CrossEncoderRerankProvider(model_name="m").rerank("q", [], top_k=5)

        self.assertEqual(ranked, [])
        module.CrossEncoder.assert_not_called()

    def test_model_loads_on_the_first_rerank_with_candidates(self):
        module, _model = _fake_sentence_transformers(predict_return=[0.5])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            CrossEncoderRerankProvider(model_name="m").rerank("q", [_doc("a")], top_k=1)

        module.CrossEncoder.assert_called_once_with("m")

    def test_missing_dependency_falls_back_instead_of_raising(self):
        # An unusable model must degrade to the incoming order, like every other
        # rerank failure - not propagate out of retrieve_context() as a 500.
        docs = [_doc("a"), _doc("b")]

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            ranked = CrossEncoderRerankProvider().rerank("q", docs, top_k=2)

        self.assertEqual([d.page_content for d, _ in ranked], ["a", "b"])


class TestModelSelection(CrossEncoderTestCase):
    def test_defaults_to_the_documented_model(self):
        module, _model = _fake_sentence_transformers(predict_return=[0.5])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("RERANK_MODEL", None)
                provider = CrossEncoderRerankProvider()
                provider.rerank("q", [_doc("a")], top_k=1)

        self.assertEqual(provider.model_name, DEFAULT_CROSS_ENCODER_MODEL)
        module.CrossEncoder.assert_called_once_with(DEFAULT_CROSS_ENCODER_MODEL)

    def test_rerank_model_env_var_overrides_the_default(self):
        module, _model = _fake_sentence_transformers(predict_return=[])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with patch.dict(os.environ, {"RERANK_MODEL": "BAAI/bge-reranker-base"}):
                provider = CrossEncoderRerankProvider()

        self.assertEqual(provider.model_name, "BAAI/bge-reranker-base")

    def test_model_is_loaded_once_and_reused_across_instances(self):
        module, _model = _fake_sentence_transformers(predict_return=[0.5])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            CrossEncoderRerankProvider(model_name="m").rerank("q", [_doc("a")], top_k=1)
            CrossEncoderRerankProvider(model_name="m").rerank("q", [_doc("a")], top_k=1)

        # get_rerank_provider() runs per retrieval; reloading weights per query
        # would make this provider unusable.
        module.CrossEncoder.assert_called_once_with("m")

    def test_missing_dependency_gives_an_actionable_error(self):
        # Simulate sentence-transformers not being installed. The message is what
        # reaches the log when reranking falls back, so it has to say what to do.
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                rerank_provider._load_cross_encoder("m")

        self.assertIn("pip install sentence-transformers", str(ctx.exception))


class TestFactory(CrossEncoderTestCase):
    def test_cross_encoder_provider_is_selectable(self):
        module, _model = _fake_sentence_transformers(predict_return=[])

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with patch.dict(os.environ, {"RERANK_PROVIDER": "cross_encoder"}):
                provider = get_rerank_provider()

        self.assertIsInstance(provider, CrossEncoderRerankProvider)

    def test_existing_providers_are_unchanged(self):
        with patch.dict(os.environ, {"RERANK_PROVIDER": "none"}):
            self.assertIsNone(get_rerank_provider())

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RERANK_PROVIDER", None)
            self.assertIsNone(get_rerank_provider())

        with patch.dict(os.environ, {"RERANK_PROVIDER": "llm"}):
            with patch("lahmajo.llm.get_llm", return_value=MagicMock()):
                self.assertIsInstance(get_rerank_provider(), LLMRerankProvider)

    def test_unknown_provider_lists_every_supported_name(self):
        with patch.dict(os.environ, {"RERANK_PROVIDER": "nonsense"}):
            with self.assertRaises(ValueError) as ctx:
                get_rerank_provider()

        message = str(ctx.exception)
        for name in ("none", "llm", "cross_encoder"):
            self.assertIn(name, message)


if __name__ == "__main__":
    unittest.main()
