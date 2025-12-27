import unittest
import tempfile
from pathlib import Path

from beyond_vector_search.evaluator import score_answer
from beyond_vector_search.router import AdaptiveRouter, RouterState


class TestEvaluator(unittest.TestCase):
    def test_score_answer_deterministic(self) -> None:
        a = "Hello   World"
        b = "hello world"
        self.assertEqual(score_answer(a, b), 1.0)
        self.assertEqual(score_answer("x", "y"), 0.0)


class TestRouter(unittest.TestCase):
    def test_router_prefers_keyword_for_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.sqlite")
            router = AdaptiveRouter.build(vocab={"hello", "world"}, rare_terms={"inc-49217"}, db_path=db_path)
            strat, feats, _meta = router.choose("Need details on INC-49217 now")
            self.assertEqual(strat, "keyword")
            self.assertGreaterEqual(feats.digit_ratio, 0.0)

    def test_router_adapts_when_keyword_wins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.sqlite")
            router = AdaptiveRouter.build(vocab={"a"}, rare_terms=set(), db_path=db_path)
            state = RouterState(weight_vector=0.0, weight_keyword=0.0, lr=0.5)
            router.save_state(state)

            # keyword consistently better
            for _ in range(4):
                router.update_from_pairwise(score_vector=0.0, score_keyword=1.0)

            st = router.load_state()
            self.assertGreater(st.weight_keyword, st.weight_vector)


if __name__ == "__main__":
    unittest.main()


