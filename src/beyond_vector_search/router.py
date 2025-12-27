from __future__ import annotations

from dataclasses import dataclass

from .telemetry import TelemetryStore, telemetry_from_env
from .text import QueryFeatures, featurize_query
from .types import Strategy


@dataclass
class RouterState:
    # learned additive biases per strategy
    weight_vector: float = 0.0
    weight_keyword: float = 0.0
    # how aggressively to adapt (used by evaluator feedback updates)
    lr: float = 0.25

    def to_json(self) -> dict:
        return {
            "weight_vector": self.weight_vector,
            "weight_keyword": self.weight_keyword,
            "lr": self.lr,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "RouterState":
        return cls(
            weight_vector=float(obj.get("weight_vector", 0.0)),
            weight_keyword=float(obj.get("weight_keyword", 0.0)),
            lr=float(obj.get("lr", 0.25)),
        )


@dataclass
class AdaptiveRouter:
    """
    A minimal "learned retrieval" router.

    - Uses query features as a decision signal.
    - Adds learned biases updated from evaluation outcomes.
    """

    vocab: set[str]
    rare_terms: set[str]
    store: TelemetryStore

    state_key: str = "router_state:v1"

    @classmethod
    def build(cls, *, vocab: set[str], rare_terms: set[str], db_path=None) -> "AdaptiveRouter":
        # db_path only applies to the default SQLite backend.
        store = telemetry_from_env(sqlite_path=db_path)
        return cls(vocab=vocab, rare_terms=rare_terms, store=store)

    def load_state(self) -> RouterState:
        return RouterState.from_json(self.store.get_state(self.state_key, RouterState().to_json()))

    def save_state(self, state: RouterState) -> None:
        self.store.set_state(self.state_key, state.to_json())

    def choose(self, query: str) -> tuple[Strategy, QueryFeatures, dict]:
        feats = featurize_query(query, vocab=self.vocab, rare_terms=self.rare_terms)
        state = self.load_state()

        # Heuristic preference score for keyword retrieval:
        # - IDs / digits: keyword tends to be robust
        # - high OOV: embeddings/TF-IDF can smear; keyword may still match substrings/rare terms
        # - rare terms: keyword may win on exact matching
        heuristic_keyword = (
            1.25 * feats.digit_ratio
            + 1.00 * feats.oov_ratio
            + 1.25 * feats.rare_ratio
            + (0.10 if feats.n_tokens <= 3 else 0.0)
        )
        heuristic_vector = 0.50 * (1.0 - min(1.0, feats.oov_ratio + feats.rare_ratio))

        score_keyword = heuristic_keyword + state.weight_keyword
        score_vector = heuristic_vector + state.weight_vector

        strategy: Strategy = "keyword" if score_keyword >= score_vector else "vector"

        meta = {
            "heuristic_keyword": heuristic_keyword,
            "heuristic_vector": heuristic_vector,
            "weight_keyword": state.weight_keyword,
            "weight_vector": state.weight_vector,
            "score_keyword": score_keyword,
            "score_vector": score_vector,
        }
        return strategy, feats, meta

    def update_from_pairwise(
        self,
        *,
        score_vector: float,
        score_keyword: float,
        state: RouterState | None = None,
    ) -> RouterState:
        """
        Bandit-style update from evaluation where both arms are scored.
        Push weight toward the better scoring strategy.
        """
        st = state or self.load_state()
        if score_keyword > score_vector:
            st.weight_keyword += st.lr
            st.weight_vector -= st.lr
        elif score_vector > score_keyword:
            st.weight_vector += st.lr
            st.weight_keyword -= st.lr
        # tie: no update
        self.save_state(st)
        return st


