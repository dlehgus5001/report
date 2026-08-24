"""
EntityExtractor – converts structured PairingRecord lists into graph entities
and relations without requiring an LLM.

Extraction rules
================
For each PairingRecord:

  1. Upsert a *location* entity for the region centre (lat_center, lon_center).

  2. Upsert an *asset* entity for every detected object class at that location:
       - status="new"         → asset.new_count++
       - status="disappeared" → asset.disappeared_count++
       - status="matched"     → asset.matched_count++
       - status="moved"       → asset.moved_count++

  3. Add "found_at" relation  : asset → location
  4. Add "co_occurred_with"   : asset ↔ asset  (all pairs that share the same
                                 session × location cluster)
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.database.models import PairingRecord
from src.graph.graph_store import GraphStore, _asset_key, _loc_key

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts graph nodes and edges from a batch of PairingRecord objects."""

    def __init__(self, store: GraphStore):
        self._store = store

    def extract_from_pairings(
        self,
        pairings: List[PairingRecord],
        session_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Process a list of PairingRecords and write entities + relations to the
        GraphStore.

        Returns:
            (n_entities_upserted, n_relations_upserted)
        """
        if not pairings:
            return 0, 0

        n_entities = 0
        n_relations = 0

        # Group pairings by (session, location_cluster) to find co-occurrences.
        # key = (session_id, loc_key) → list of asset_entity_ids
        co_occurrence_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for p in pairings:
            lat = p.lat_center
            lon = p.lon_center

            # 1. Location entity
            loc_id = self._store.upsert_location(lat, lon)
            n_entities += 1

            # 2. Asset entity (current object for new/matched/moved; past for disappeared)
            if p.status in ("new", "matched", "moved"):
                obj_class = p.current_object_class or "unknown object"
                confidence = p.current_confidence or 0.0
                obj_lat = p.current_lat or lat
                obj_lon = p.current_lon or lon
                timestamp = p.current_capture_time
            else:  # "disappeared"
                obj_class = p.past_object_class or "unknown object"
                confidence = p.past_confidence or 0.0
                obj_lat = p.past_lat or lat
                obj_lon = p.past_lon or lon
                timestamp = p.past_capture_time

            asset_id = self._store.upsert_asset(
                object_class=obj_class,
                lat=lat,          # cluster by region centre, not exact detection lat
                lon=lon,
                status=p.status,
                confidence=confidence,
                timestamp=timestamp,
                session_id=session_id or p.session_id,
            )
            n_entities += 1

            # 3. "found_at" relation: asset → location
            self._store.upsert_relation(
                source_id=asset_id,
                target_id=loc_id,
                relation_type="found_at",
                extra_props={"confidence_sum": confidence},
            )
            n_relations += 1

            # Track for co-occurrence
            group_key = (session_id or p.session_id or "", _loc_key(lat, lon))
            co_occurrence_map[group_key].append(asset_id)

        # 4. "co_occurred_with" relations (within same session × location)
        for (sess, loc), asset_ids in co_occurrence_map.items():
            unique_ids = list(set(asset_ids))
            for i in range(len(unique_ids)):
                for j in range(i + 1, len(unique_ids)):
                    a, b = unique_ids[i], unique_ids[j]
                    # Store both directions so both nodes appear as neighbours
                    self._store.upsert_relation(a, b, "co_occurred_with",
                                                extra_props={"locations": [loc]})
                    self._store.upsert_relation(b, a, "co_occurred_with",
                                                extra_props={"locations": [loc]})
                    n_relations += 2

        logger.debug(
            "[EntityExtractor] Processed %d pairings → %d entity upserts, "
            "%d relation upserts.",
            len(pairings), n_entities, n_relations,
        )
        return n_entities, n_relations
