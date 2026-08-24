"""
GraphStore – SQLite-backed graph persistence with NetworkX in-memory graph.

Responsibilities
================
- Upsert entities and relations into SQLite.
- Maintain a NetworkX graph (rebuilt from SQLite on first access) for
  community-detection and neighbour traversal.
- Run Louvain (or fallback: connected-components) community detection and
  persist the resulting communities to the DB.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from src.graph.models import (
    CommunityRecord,
    EntityRecord,
    GraphBase,
    RelationRecord,
    create_graph_engine,
)

logger = logging.getLogger(__name__)

# Round lat/lon to this many decimal places to cluster nearby points.
# 2 dp  ≈ 1.1 km resolution at the equator.
_LOC_PRECISION = 2


def _loc_key(lat: float, lon: float) -> str:
    return f"loc:{lat:.{_LOC_PRECISION}f},{lon:.{_LOC_PRECISION}f}"


def _asset_key(object_class: str, lat: float, lon: float) -> str:
    slug = object_class.lower().replace(" ", "_")
    return f"asset:{slug}:{lat:.{_LOC_PRECISION}f},{lon:.{_LOC_PRECISION}f}"


class GraphStore:
    """
    Wraps the graph SQLite database and an in-memory NetworkX graph.

    The NetworkX graph is rebuilt lazily on first access and must be
    explicitly refreshed (``_load_networkx``) after bulk DB writes.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._engine = create_graph_engine(db_path)
        self._nx_graph = None  # lazy-loaded

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _session(self) -> Session:
        return Session(self._engine)

    def _get_nx(self):
        """Return (or build) the in-memory NetworkX graph."""
        if self._nx_graph is None:
            self._load_networkx()
        return self._nx_graph

    def _load_networkx(self):
        """Rebuild NetworkX graph from SQLite."""
        try:
            import networkx as nx
        except ImportError:
            logger.warning("[GraphStore] networkx not installed – community detection disabled.")
            self._nx_graph = None
            return

        G = nx.Graph()
        with self._session() as sess:
            entities = sess.query(EntityRecord).all()
            relations = sess.query(RelationRecord).all()

        for e in entities:
            G.add_node(e.id, entity_type=e.entity_type, name=e.name)

        for r in relations:
            if r.source_id in G and r.target_id in G:
                count = (r.properties or {}).get("count", 1)
                if G.has_edge(r.source_id, r.target_id):
                    G[r.source_id][r.target_id]["weight"] += count
                else:
                    G.add_edge(r.source_id, r.target_id,
                               relation_type=r.relation_type, weight=count)

        self._nx_graph = G
        logger.debug("[GraphStore] NetworkX graph: %d nodes, %d edges",
                     G.number_of_nodes(), G.number_of_edges())

    def invalidate_nx(self):
        """Force rebuild of in-memory graph on next access."""
        self._nx_graph = None

    # ------------------------------------------------------------------ #
    # Entity CRUD                                                          #
    # ------------------------------------------------------------------ #

    def upsert_location(
        self,
        lat: float,
        lon: float,
    ) -> str:
        """Insert or update a location entity. Returns the entity ID."""
        eid = _loc_key(lat, lon)
        name = f"Location ({lat:.{_LOC_PRECISION}f}, {lon:.{_LOC_PRECISION}f})"
        props = {"lat": round(lat, _LOC_PRECISION), "lon": round(lon, _LOC_PRECISION)}

        with self._session() as sess:
            rec = sess.get(EntityRecord, eid)
            if rec is None:
                rec = EntityRecord(
                    id=eid,
                    entity_type="location",
                    name=name,
                    properties=props,
                    observation_count=1,
                )
                sess.add(rec)
            else:
                rec.observation_count = (rec.observation_count or 0) + 1
            sess.commit()

        return eid

    def upsert_asset(
        self,
        object_class: str,
        lat: float,
        lon: float,
        status: str,
        confidence: float,
        timestamp: Optional[datetime],
        session_id: Optional[str],
    ) -> str:
        """
        Insert or update an asset entity for (object_class, location_cluster).
        Status counters (new / disappeared / matched / moved) are incremented
        on each call.

        Returns the entity ID.
        """
        eid = _asset_key(object_class, lat, lon)
        loc_key = _loc_key(lat, lon)
        name = f"{object_class} @ ({lat:.{_LOC_PRECISION}f},{lon:.{_LOC_PRECISION}f})"

        with self._session() as sess:
            rec = sess.get(EntityRecord, eid)
            if rec is None:
                props: Dict = {
                    "object_class": object_class,
                    "lat": round(lat, _LOC_PRECISION),
                    "lon": round(lon, _LOC_PRECISION),
                    "location_key": loc_key,
                    "new_count": 0,
                    "disappeared_count": 0,
                    "matched_count": 0,
                    "moved_count": 0,
                    "total_confidence": 0.0,
                    "sessions": [],
                }
                rec = EntityRecord(
                    id=eid,
                    entity_type="asset",
                    name=name,
                    properties=props,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    observation_count=0,
                )
                sess.add(rec)

            # Mutate properties
            props = dict(rec.properties or {})
            counter_key = f"{status}_count"
            if counter_key in props:
                props[counter_key] = props.get(counter_key, 0) + 1
            props["total_confidence"] = props.get("total_confidence", 0.0) + confidence
            sessions: list = props.get("sessions", [])
            if session_id and session_id not in sessions:
                sessions = sessions[-9:] + [session_id]   # keep last 10
            props["sessions"] = sessions

            rec.properties = props
            rec.observation_count = (rec.observation_count or 0) + 1
            if timestamp:
                if rec.first_seen is None or timestamp < rec.first_seen:
                    rec.first_seen = timestamp
                if rec.last_seen is None or timestamp > rec.last_seen:
                    rec.last_seen = timestamp

            sess.commit()

        return eid

    def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        with self._session() as sess:
            return sess.get(EntityRecord, entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[EntityRecord]:
        with self._session() as sess:
            return sess.query(EntityRecord).filter(
                EntityRecord.entity_type == entity_type
            ).all()

    def get_assets_near(
        self,
        lat: float,
        lon: float,
        radius_deg: float = 0.05,
    ) -> List[EntityRecord]:
        """Return asset entities whose cluster centre is within radius_deg."""
        with self._session() as sess:
            candidates = sess.query(EntityRecord).filter(
                EntityRecord.entity_type == "asset"
            ).all()

        results = []
        for rec in candidates:
            props = rec.properties or {}
            elat = props.get("lat", 0.0)
            elon = props.get("lon", 0.0)
            if abs(elat - lat) <= radius_deg and abs(elon - lon) <= radius_deg:
                results.append(rec)

        return results

    # ------------------------------------------------------------------ #
    # Relation CRUD                                                        #
    # ------------------------------------------------------------------ #

    def _relation_key(self, source_id: str, target_id: str, rel_type: str) -> Optional[RelationRecord]:
        """Return the existing relation record or None."""
        with self._session() as sess:
            return (
                sess.query(RelationRecord)
                .filter(
                    RelationRecord.source_id == source_id,
                    RelationRecord.target_id == target_id,
                    RelationRecord.relation_type == rel_type,
                )
                .first()
            )

    def upsert_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        extra_props: Optional[Dict] = None,
    ) -> str:
        """Increment count on an existing relation, or create it."""
        extra_props = extra_props or {}

        with self._session() as sess:
            rec = (
                sess.query(RelationRecord)
                .filter(
                    RelationRecord.source_id == source_id,
                    RelationRecord.target_id == target_id,
                    RelationRecord.relation_type == relation_type,
                )
                .first()
            )
            if rec is None:
                props = {"count": 1, **extra_props}
                rec = RelationRecord(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=props,
                )
                sess.add(rec)
            else:
                props = dict(rec.properties or {})
                props["count"] = props.get("count", 0) + 1
                for k, v in extra_props.items():
                    if isinstance(v, (int, float)):
                        props[k] = props.get(k, 0) + v
                    elif isinstance(v, list):
                        existing = props.get(k, [])
                        props[k] = existing + [x for x in v if x not in existing]
                    else:
                        props[k] = v
                rec.properties = props
                rec.updated_at = datetime.utcnow()
            rid = rec.id
            sess.commit()

        return rid

    def get_relations(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[RelationRecord]:
        with self._session() as sess:
            q = sess.query(RelationRecord)
            if source_id:
                q = q.filter(RelationRecord.source_id == source_id)
            if target_id:
                q = q.filter(RelationRecord.target_id == target_id)
            if relation_type:
                q = q.filter(RelationRecord.relation_type == relation_type)
            return q.all()

    # ------------------------------------------------------------------ #
    # Community detection                                                  #
    # ------------------------------------------------------------------ #

    def detect_and_save_communities(self) -> List[CommunityRecord]:
        """
        Run Louvain community detection on the in-memory NetworkX graph
        (asset nodes only, co_occurred_with edges) and persist results.

        Falls back to connected-components if Louvain is unavailable.
        Returns the list of saved CommunityRecord objects.
        """
        self.invalidate_nx()
        G = self._get_nx()
        if G is None:
            return []

        try:
            import networkx as nx
        except ImportError:
            return []

        # Build asset-only subgraph using co_occurred_with edges
        asset_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("entity_type") == "asset"
        ]
        asset_subgraph = G.subgraph(asset_nodes).copy()

        if asset_subgraph.number_of_nodes() == 0:
            return []

        # Community detection
        try:
            from networkx.algorithms.community import louvain_communities
            raw_communities = louvain_communities(asset_subgraph, weight="weight", seed=42)
        except Exception:
            try:
                from networkx.algorithms.components import connected_components
                raw_communities = list(connected_components(asset_subgraph))
            except Exception as exc:
                logger.warning("[GraphStore] Community detection failed: %s", exc)
                return []

        communities = [set(c) for c in raw_communities if len(c) >= 1]
        logger.info("[GraphStore] Detected %d communities from %d asset nodes.",
                    len(communities), asset_subgraph.number_of_nodes())

        # Build CommunityRecord objects
        saved = []
        with self._session() as sess:
            # Drop old communities before re-inserting
            sess.query(CommunityRecord).delete()

            for idx, member_set in enumerate(communities):
                member_ids = list(member_set)

                # Collect info about members
                member_recs = sess.query(EntityRecord).filter(
                    EntityRecord.id.in_(member_ids)
                ).all()

                class_counts: Dict[str, int] = {}
                for m in member_recs:
                    props = m.properties or {}
                    cls = props.get("object_class", m.name)
                    class_counts[cls] = class_counts.get(cls, 0) + (m.observation_count or 1)

                top_classes = sorted(class_counts.items(), key=lambda x: -x[1])
                label_parts = [f"{cls}({n})" for cls, n in top_classes[:4]]
                label = f"Cluster-{idx}: {', '.join(label_parts)}"

                # Build structured summary text
                location_keys = set()
                total_obs = 0
                total_new = 0
                total_disappeared = 0
                for m in member_recs:
                    props = m.properties or {}
                    if "location_key" in props:
                        location_keys.add(props["location_key"])
                    total_obs += m.observation_count or 0
                    total_new += props.get("new_count", 0)
                    total_disappeared += props.get("disappeared_count", 0)

                member_summary = (
                    f"Assets: {', '.join(label_parts)} | "
                    f"Locations: {len(location_keys)} cluster(s) | "
                    f"Observations: {total_obs} | "
                    f"New deployments: {total_new} | "
                    f"Disappearances: {total_disappeared}"
                )

                rec = CommunityRecord(
                    id=str(uuid.uuid4()),
                    community_index=idx,
                    label=label,
                    member_ids=member_ids,
                    member_summary=member_summary,
                )
                sess.add(rec)
                saved.append(rec)

            sess.commit()
            # Re-query after commit to avoid DetachedInstanceError on returned objects
            saved = sess.query(CommunityRecord).order_by(
                CommunityRecord.community_index
            ).all()
            sess.expunge_all()

        return saved

    def get_all_communities(self) -> List[CommunityRecord]:
        with self._session() as sess:
            return sess.query(CommunityRecord).order_by(CommunityRecord.community_index).all()

    def update_community_summary(self, community_id: str, summary: str):
        with self._session() as sess:
            rec = sess.get(CommunityRecord, community_id)
            if rec:
                rec.summary = summary
                sess.commit()

    # ------------------------------------------------------------------ #
    # Clear                                                                #
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """모든 엔티티·관계·커뮤니티를 삭제하고 NetworkX 캐시를 초기화한다."""
        with self._session() as sess:
            sess.query(CommunityRecord).delete()
            sess.query(RelationRecord).delete()
            sess.query(EntityRecord).delete()
            sess.commit()
        self._nx_graph = None

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        with self._session() as sess:
            n_entities = sess.query(EntityRecord).count()
            n_assets = sess.query(EntityRecord).filter(
                EntityRecord.entity_type == "asset"
            ).count()
            n_locations = sess.query(EntityRecord).filter(
                EntityRecord.entity_type == "location"
            ).count()
            n_relations = sess.query(RelationRecord).count()
            n_communities = sess.query(CommunityRecord).count()

        return {
            "entities": n_entities,
            "assets": n_assets,
            "locations": n_locations,
            "relations": n_relations,
            "communities": n_communities,
        }
