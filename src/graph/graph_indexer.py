"""
GraphIndexer – high-level coordinator used by the pipeline.

Responsibilities
================
1. Accept a PairingRecord batch → EntityExtractor → GraphStore.
2. Optionally trigger community detection after each indexing run.
3. Optionally request LLM summaries for newly detected communities
   (best-effort; errors are swallowed so the main pipeline is never blocked).

Usage in pipeline.py
====================
    from src.graph.graph_indexer import GraphIndexer

    indexer = GraphIndexer()                      # singleton-friendly
    indexer.index_pairings(pairings, session_id)  # after _pair_and_store()
    context = indexer.get_historical_context(pairings)  # before report gen
"""

import logging
from typing import List, Optional

from src.database.models import PairingRecord
from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_retriever import GraphRetriever
from src.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphIndexer:
    """
    Orchestrates entity extraction, graph storage, community detection, and
    retrieval for the MSIS pipeline.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from src.config import GRAPH_DB_PATH
            db_path = GRAPH_DB_PATH

        from src.config import GRAPHRAG_COMMUNITY_INTERVAL, GRAPHRAG_CONTEXT_RADIUS_DEG
        self._auto_community_every = GRAPHRAG_COMMUNITY_INTERVAL
        self._context_radius_deg = GRAPHRAG_CONTEXT_RADIUS_DEG

        self._store = GraphStore(db_path)
        self._extractor = EntityExtractor(self._store)
        self._retriever = GraphRetriever(self._store)
        self._index_count = 0

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def index_pairings(
        self,
        pairings: List[PairingRecord],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Extract entities and relations from a pairing batch and write them to
        the graph store.  Optionally triggers community detection afterwards.

        This method is intentionally non-blocking: all exceptions are caught so
        a graph failure never halts the main intelligence pipeline.
        """
        if not pairings:
            return

        try:
            n_ent, n_rel = self._extractor.extract_from_pairings(pairings, session_id)
            self._index_count += 1
            logger.info(
                "[GraphIndexer] Indexed %d pairings → %d entity upserts, "
                "%d relation upserts  (run #%d).",
                len(pairings), n_ent, n_rel, self._index_count,
            )

            # Periodic community detection
            if (
                self._auto_community_every > 0
                and self._index_count % self._auto_community_every == 0
            ):
                self._run_community_detection()

        except Exception as exc:
            logger.warning("[GraphIndexer] Indexing error (non-fatal): %s", exc)

    def _run_community_detection(self) -> None:
        """Detect communities and persist them.  Errors are swallowed."""
        try:
            communities = self._store.detect_and_save_communities()
            logger.info("[GraphIndexer] Community detection: %d communities saved.",
                        len(communities))
        except Exception as exc:
            logger.warning("[GraphIndexer] Community detection error (non-fatal): %s", exc)

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def get_historical_context(
        self,
        pairings: List[PairingRecord],
        radius_deg: Optional[float] = None,
    ) -> str:
        """
        Return a formatted historical-context string for LLM prompt injection.
        Returns empty string on any error.
        """
        if radius_deg is None:
            radius_deg = self._context_radius_deg
        try:
            return self._retriever.get_historical_context(pairings, radius_deg)
        except Exception as exc:
            logger.warning("[GraphIndexer] Context retrieval error (non-fatal): %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # Direct store / retriever access                                      #
    # ------------------------------------------------------------------ #

    @property
    def store(self) -> GraphStore:
        return self._store

    @property
    def retriever(self) -> GraphRetriever:
        return self._retriever

    def clear(self) -> None:
        """그래프 DB의 모든 데이터를 삭제한다 (세션 시작 전 호출해 누적 방지)."""
        try:
            self._store.clear()
            self._index_count = 0
            logger.info("[GraphIndexer] Graph DB cleared.")
        except Exception as exc:
            logger.warning("[GraphIndexer] Clear error: %s", exc)

    def reindex_communities(self) -> int:
        """Force a community-detection run and return the number of communities."""
        try:
            communities = self._store.detect_and_save_communities()
            return len(communities)
        except Exception as exc:
            logger.warning("[GraphIndexer] Forced reindex error: %s", exc)
            return 0

    def stats(self) -> dict:
        """Return graph statistics (entity counts, relation counts, etc.)."""
        try:
            return self._store.stats()
        except Exception:
            return {}
