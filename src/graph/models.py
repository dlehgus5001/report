"""
SQLAlchemy ORM models for the MSIS GraphRAG knowledge graph.

Graph schema
============
Entities
  - location  : geographic cluster, key = "loc:{lat:.2f},{lon:.2f}"
  - asset     : unique (object_class, location_key), key = "asset:{class}:{loc_key}"

Relations
  - found_at          : asset → location
  - co_occurred_with  : asset ↔ asset  (symmetric, stored once each direction)

Communities
  - A cluster of assets detected by Louvain community detection.
    Each community has an auto-generated label and an optional LLM summary.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.orm import declarative_base

GraphBase = declarative_base()


class EntityRecord(GraphBase):
    """A node in the intelligence knowledge graph (location or asset class)."""

    __tablename__ = "graph_entities"

    # Stable, human-readable key – used for upsert deduplication.
    id = Column(String(256), primary_key=True)

    # "location" | "asset"
    entity_type = Column(String(32), nullable=False)

    # Human-readable label, e.g. "military tank @ (37.50,127.00)"
    name = Column(String(512), nullable=False)

    # Extra attributes as JSON dict:
    #   location : {"lat": 37.50, "lon": 127.00}
    #   asset    : {"object_class": "military tank", "lat": 37.50, "lon": 127.00,
    #               "new_count": 3, "disappeared_count": 1,
    #               "matched_count": 2, "moved_count": 0,
    #               "total_confidence": 4.2, "sessions": [...]}
    properties = Column(JSON, nullable=True)

    # Temporal bounds across all observations of this entity
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True)

    # Total number of pairing-record events referencing this entity
    observation_count = Column(Integer, default=0, nullable=False)


class RelationRecord(GraphBase):
    """A directed edge in the intelligence knowledge graph."""

    __tablename__ = "graph_relations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Entity IDs (must exist in graph_entities)
    source_id = Column(String(256), nullable=False)
    target_id = Column(String(256), nullable=False)

    # Relation semantics:
    #   "found_at"         : asset → location  (properties: count, first_seen, last_seen,
    #                                            new_count, disappeared_count, …)
    #   "co_occurred_with" : asset ↔ asset      (properties: count, locations)
    relation_type = Column(String(64), nullable=False)

    # Mutable metadata updated on each new observation
    properties = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CommunityRecord(GraphBase):
    """A detected community (cluster) of co-occurring military asset types."""

    __tablename__ = "graph_communities"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Index within the last community-detection run (0-based)
    community_index = Column(Integer, nullable=False)

    # Auto-generated short label, e.g. "Cluster-0: military tank, APC, artillery"
    label = Column(String(512), nullable=False)

    # JSON list of entity IDs belonging to this community
    member_ids = Column(JSON, nullable=False)

    # Structured text summary of the community members (auto-generated, no LLM)
    member_summary = Column(Text, nullable=True)

    # Optional LLM-generated intelligence summary (populated by graph_indexer)
    summary = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def create_graph_engine(db_path: str):
    """Create (or open) the graph SQLite database and ensure tables exist."""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    try:
        GraphBase.metadata.create_all(engine)
    except OperationalError as exc:
        if "already exists" not in str(exc).lower():
            raise
    return engine
