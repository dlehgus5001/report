"""
GraphRetriever – query interface for the MSIS GraphRAG knowledge graph.

Provides two retrieval modes (mirroring Microsoft GraphRAG's terminology):

  Local search  – entity-centric: returns historical asset activity within a
                  geographic area (radius-based entity lookup + relations).

  Global search – community-centric: returns community summaries that are
                  relevant to a given query location or text.

The main entry point for the report-generation pipeline is
``get_historical_context(pairings)``, which formats a text block ready to be
injected into the LLM prompt.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.database.models import PairingRecord
from src.graph.graph_store import GraphStore, _loc_key

logger = logging.getLogger(__name__)

# How many days of history to surface in the context block
_HISTORY_DAYS = 90


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class GraphRetriever:
    """Query interface built on top of GraphStore."""

    def __init__(self, store: GraphStore):
        self._store = store

    # ------------------------------------------------------------------ #
    # Local search                                                         #
    # ------------------------------------------------------------------ #

    def local_search(
        self,
        lat: float,
        lon: float,
        radius_deg: float = 0.05,
    ) -> Dict:
        """
        Return all asset entities within ``radius_deg`` of (lat, lon)
        together with their historical statistics.

        Returns a dict:
          {
            "location": {"lat": …, "lon": …, "radius_deg": …},
            "assets":   [{"object_class": …, "new_count": …, …}, …],
            "total_observations": int,
          }
        """
        asset_recs = self._store.get_assets_near(lat, lon, radius_deg)

        assets_out = []
        total_obs = 0
        for rec in asset_recs:
            props = rec.properties or {}
            obs = rec.observation_count or 0
            total_obs += obs
            assets_out.append({
                "entity_id":       rec.id,
                "object_class":    props.get("object_class", rec.name),
                "location_key":    props.get("location_key", ""),
                "new_count":       props.get("new_count", 0),
                "disappeared_count": props.get("disappeared_count", 0),
                "matched_count":   props.get("matched_count", 0),
                "moved_count":     props.get("moved_count", 0),
                "total_confidence": props.get("total_confidence", 0.0),
                "observation_count": obs,
                "first_seen":      _fmt_dt(rec.first_seen),
                "last_seen":       _fmt_dt(rec.last_seen),
            })

        # Sort: most-observed first
        assets_out.sort(key=lambda x: -x["observation_count"])

        return {
            "location": {"lat": lat, "lon": lon, "radius_deg": radius_deg},
            "assets": assets_out,
            "total_observations": total_obs,
        }

    # ------------------------------------------------------------------ #
    # Global search (community summaries)                                  #
    # ------------------------------------------------------------------ #

    def global_search(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_deg: float = 0.1,
        max_communities: int = 5,
    ) -> List[Dict]:
        """
        Return community summaries relevant to the given location.

        If lat/lon are provided, prefer communities whose members overlap
        with asset entities near that location; otherwise return all.

        Returns list of dicts:
          [{"label": …, "member_summary": …, "summary": …, "member_count": …}, …]
        """
        communities = self._store.get_all_communities()
        if not communities:
            return []

        # Collect local asset IDs for relevance filtering
        local_entity_ids: set = set()
        if lat is not None and lon is not None:
            local_assets = self._store.get_assets_near(lat, lon, radius_deg)
            local_entity_ids = {a.id for a in local_assets}

        results = []
        for comm in communities:
            member_ids = set(comm.member_ids or [])
            overlap = len(member_ids & local_entity_ids)
            results.append({
                "community_id":   comm.id,
                "label":          comm.label,
                "member_summary": comm.member_summary,
                "summary":        comm.summary,
                "member_count":   len(member_ids),
                "local_overlap":  overlap,
            })

        # Sort: local overlap first, then by member count
        if local_entity_ids:
            results.sort(key=lambda x: (-x["local_overlap"], -x["member_count"]))
        else:
            results.sort(key=lambda x: -x["member_count"])

        return results[:max_communities]

    # ------------------------------------------------------------------ #
    # Historical context (injected into report LLM prompt)                #
    # ------------------------------------------------------------------ #

    def get_historical_context(
        self,
        pairings: List[PairingRecord],
        radius_deg: float = 0.05,
    ) -> str:
        """
        Build a formatted text block that summarises historical graph knowledge
        relevant to the current pairing batch.

        The returned string is designed to be prepended to the LLM user prompt
        so that the model can reference prior observation patterns.
        """
        if not pairings:
            return ""

        # Derive representative centre from pairings
        lats = [p.lat_center for p in pairings if p.lat_center]
        lons = [p.lon_center for p in pairings if p.lon_center]
        if not lats:
            return ""
        lat_c = sum(lats) / len(lats)
        lon_c = sum(lons) / len(lons)

        # --- Local search ---
        local = self.local_search(lat_c, lon_c, radius_deg)
        assets = local["assets"]

        if not assets and not self._store.get_all_communities():
            return ""

        lines = [
            "=== GRAPHRAG HISTORICAL CONTEXT ===",
            f"Area centre: ({lat_c:.3f}, {lon_c:.3f})  |  Radius: {radius_deg}°",
            "",
        ]

        # Historical asset activity
        if assets:
            lines.append("HISTORICAL ASSET ACTIVITY (graph knowledge base):")
            for a in assets[:12]:   # cap at 12 to keep prompt concise
                cls = a["object_class"]
                new = a["new_count"]
                gone = a["disappeared_count"]
                matched = a["matched_count"]
                moved = a["moved_count"]
                obs = a["observation_count"]
                fs = a["first_seen"]
                ls = a["last_seen"]
                avg_conf = (
                    a["total_confidence"] / obs
                    if obs > 0 else 0.0
                )
                lines.append(
                    f"  {cls}: {obs} obs  "
                    f"(new={new} / persisted={matched} / moved={moved} / disappeared={gone})  "
                    f"avgConf={avg_conf:.2f}  "
                    f"first={fs}  last={ls}"
                )

            # Class-level aggregation (all observed classes + total new/gone)
            class_totals: Dict[str, Dict] = {}
            for a in assets:
                cls = a["object_class"]
                if cls not in class_totals:
                    class_totals[cls] = {"new": 0, "disappeared": 0, "total": 0}
                class_totals[cls]["new"] += a["new_count"]
                class_totals[cls]["disappeared"] += a["disappeared_count"]
                class_totals[cls]["total"] += a["observation_count"]

            if class_totals:
                dominant = sorted(class_totals.items(),
                                  key=lambda x: -x[1]["total"])[:5]
                dom_str = "  ".join(
                    f"{cls}:{v['total']}" for cls, v in dominant
                )
                lines += ["", f"  Dominant asset classes: {dom_str}"]

        # Community summaries
        global_results = self.global_search(lat_c, lon_c, radius_deg)
        if global_results:
            lines += ["", "INTELLIGENCE PATTERN COMMUNITIES (co-occurrence clusters):"]
            for cr in global_results:
                overlap_note = (
                    f"  [↑ {cr['local_overlap']} local match(es)]"
                    if cr["local_overlap"] > 0 else ""
                )
                lines.append(f"  [{cr['label']}]{overlap_note}")
                if cr["member_summary"]:
                    lines.append(f"    {cr['member_summary']}")
                if cr["summary"]:
                    lines.append(f"    INTEL: {cr['summary']}")

        lines += ["", "=== END HISTORICAL CONTEXT ===", ""]
        return "\n".join(lines)
