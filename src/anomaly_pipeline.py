"""
src/anomaly_pipeline.py
-----------------------
Construction fleet GPS anomaly detection.

Reimagined from a 2018 project at Kolon Benit using today's stack:
  - H3 spatial indexing (uniform hex grid, fast lookups, scalable)
  - DuckDB analytical SQL (zero-cloud-cost local processing of GPS streams)
  - Hybrid rule + statistical scoring (operational over academic)

Why this stack and not "real-time streaming + deep learning":
  Anomaly detection in operational fleet management is overwhelmingly
  about explainability and operator trust, not accuracy at the margins.
  A site manager needs to know WHY a truck got flagged. A neural net
  saying "score 0.87" is useless. A rule saying "entered restricted
  zone at 14:23 for 18 minutes" is actionable.

  Hybrid scoring:
    1. Hard rules (geofence violations, off-hours activity)
       → always trigger, fully explainable
    2. Statistical scoring (dwell time outliers, route deviation)
       → catch the unknown unknowns
    3. Aggregate to truck-day level with reason codes

The original 2018 system used SQL + Python loops over PostGIS.
Today's version does the same logical work in DuckDB at 10-50x speed
with no infrastructure to maintain.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict
import duckdb
import h3
import pandas as pd

logger = logging.getLogger(__name__)

# H3 resolution 9 = ~0.1 km² hex (174m edge length).
# Operational sweet spot: small enough to distinguish neighbouring zones,
# large enough to absorb GPS noise.
H3_RESOLUTION = 9


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------

def load_reference(zones_path: Path) -> Dict:
    with open(zones_path) as f:
        return json.load(f)


def authorized_h3_cells(zones: Dict, radius_cells: int = 3) -> Dict[str, str]:
    """
    Build authorized H3 cells per location type.
    radius_cells = how many hex rings around each location are considered "in zone".
    """
    cell_to_zone = {}
    for loc in zones["authorized_locations"]:
        center = h3.latlng_to_cell(loc["lat"], loc["lon"], H3_RESOLUTION)
        zone_type = loc["type"]
        zone_name = loc["name"]
        # k-ring neighbours
        for cell in h3.grid_disk(center, radius_cells):
            if cell not in cell_to_zone:
                cell_to_zone[cell] = f"{zone_type}:{zone_name}"
    return cell_to_zone


def restricted_h3_cells(zones: Dict) -> Dict[str, str]:
    """
    Build restricted H3 cells with named violations.
    Tight rings (1-2) to avoid catching trucks passing through nearby authorized routes.
    Real-world systems would use actual polygon geofences here.
    """
    cells = {}
    for r in zones["restricted_zones"]:
        center = h3.latlng_to_cell(r["lat"], r["lon"], H3_RESOLUTION)
        # Tighter: 1 ring for small zones, 2 rings for larger
        n_rings = 1 if r["radius_km"] < 1.0 else 2
        for cell in h3.grid_disk(center, n_rings):
            cells[cell] = r["name"]
    return cells


# ---------------------------------------------------------------------------
# H3 enrichment
# ---------------------------------------------------------------------------

def enrich_with_h3(gps_df: pd.DataFrame, zones: Dict) -> pd.DataFrame:
    """Add H3 cell + zone classification to each GPS ping."""
    logger.info(f"Enriching {len(gps_df):,} GPS pings with H3 cells...")

    authorized = authorized_h3_cells(zones)
    restricted = restricted_h3_cells(zones)

    gps_df = gps_df.copy()
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])
    gps_df["h3_cell"] = gps_df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RESOLUTION),
        axis=1
    )
    gps_df["zone"] = gps_df["h3_cell"].map(authorized).fillna("unknown")
    gps_df["zone_type"] = gps_df["zone"].str.split(":").str[0]
    gps_df["restricted_violation"] = gps_df["h3_cell"].map(restricted).fillna("")
    gps_df["hour"] = gps_df["timestamp"].dt.hour
    gps_df["date"] = gps_df["timestamp"].dt.date

    logger.info(
        f"  Authorized cells: {gps_df['zone'].ne('unknown').sum():,} "
        f"({gps_df['zone'].ne('unknown').mean()*100:.1f}%)"
    )
    logger.info(f"  Restricted hits:  {gps_df['restricted_violation'].ne('').sum():,}")
    return gps_df


# ---------------------------------------------------------------------------
# Anomaly detection (DuckDB SQL)
# ---------------------------------------------------------------------------

ANOMALY_QUERY = """
WITH per_ping AS (
    SELECT
        truck_id,
        date,
        timestamp,
        h3_cell,
        zone,
        zone_type,
        restricted_violation,
        hour,
        speed_kmh
    FROM gps
),
truck_day_stats AS (
    SELECT
        truck_id,
        date,
        COUNT(*)                                          AS n_pings,
        -- Rule 1: restricted zone violation
        MAX(CASE WHEN restricted_violation != ''
                 THEN restricted_violation END)           AS restricted_zone_hit,
        SUM(CASE WHEN restricted_violation != ''
                 THEN 1 ELSE 0 END)                       AS restricted_ping_count,
        -- Rule 2: off-hours activity (before 7am or after 6pm)
        SUM(CASE WHEN (hour < 7 OR hour > 18)
                 THEN 1 ELSE 0 END)                       AS off_hours_pings,
        -- Rule 3: time in unknown zones (not authorized)
        SUM(CASE WHEN zone = 'unknown' AND speed_kmh < 5
                 THEN 1 ELSE 0 END)                       AS dwell_unknown_pings,
        -- Operational stats
        COUNT(DISTINCT h3_cell)                           AS unique_cells_visited,
        COUNT(DISTINCT zone)                              AS unique_zones_visited,
        AVG(speed_kmh)                                    AS avg_speed
    FROM per_ping
    GROUP BY truck_id, date
)
SELECT
    truck_id,
    date,
    n_pings,
    restricted_zone_hit,
    restricted_ping_count,
    off_hours_pings,
    dwell_unknown_pings,
    unique_cells_visited,
    unique_zones_visited,
    ROUND(avg_speed, 1) AS avg_speed
FROM truck_day_stats
ORDER BY truck_id, date
"""


@dataclass
class AnomalyDecision:
    truck_id: str
    date: str
    is_anomaly: bool
    severity: str                       # "high" | "medium" | "low" | "none"
    primary_reason: str
    rule_score: float                   # 0-1 from hard rules
    statistical_score: float            # 0-1 from outlier detection
    final_score: float
    evidence: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_truck_day(row: pd.Series, fleet_baseline: Dict) -> AnomalyDecision:
    """
    Hybrid scoring:
      Rule score: hard violations (restricted zone, off-hours)
      Stat score: fleet-relative dwell time anomaly

    Combined with weighting that favours interpretability:
      final = max(rule, 0.7 * stat)
      → hard rules dominate, statistical only adds if no rule fired
    """
    truck_id = row["truck_id"]
    date_str = str(row["date"])

    # --- Rule-based ---
    rule_score = 0.0
    reasons = []
    evidence = {}

    if row["restricted_ping_count"] >= 3:
        rule_score = max(rule_score, 1.0)
        reasons.append(f"restricted_zone:{row['restricted_zone_hit']}")
        evidence["restricted_violation"] = {
            "zone": row["restricted_zone_hit"],
            "ping_count": int(row["restricted_ping_count"]),
        }

    if row["off_hours_pings"] > 5:
        rule_score = max(rule_score, 0.85)
        reasons.append(f"off_hours_activity:{row['off_hours_pings']}_pings")
        evidence["off_hours"] = {
            "pings": int(row["off_hours_pings"]),
        }

    # --- Statistical-based ---
    # Compare dwell-in-unknown to fleet baseline
    baseline_dwell = fleet_baseline.get("dwell_unknown_p90", 5)
    if row["dwell_unknown_pings"] > baseline_dwell:
        ratio = row["dwell_unknown_pings"] / max(baseline_dwell, 1)
        stat_score = min(1.0, 0.5 + 0.25 * (ratio - 1))  # 0.5 at threshold, up to 1.0
        if stat_score > 0.55:
            reasons.append(f"excessive_dwell_unknown:{row['dwell_unknown_pings']}_pings")
            evidence["dwell_outlier"] = {
                "pings": int(row["dwell_unknown_pings"]),
                "fleet_p90": float(baseline_dwell),
            }
    else:
        stat_score = 0.0

    # --- Combine ---
    final_score = max(rule_score, 0.7 * stat_score)

    if final_score >= 0.85:
        severity = "high"
    elif final_score >= 0.55:
        severity = "medium"
    elif final_score >= 0.30:
        severity = "low"
    else:
        severity = "none"

    return AnomalyDecision(
        truck_id=truck_id,
        date=date_str,
        is_anomaly=(final_score >= 0.55),
        severity=severity,
        primary_reason=reasons[0] if reasons else "none",
        rule_score=round(rule_score, 3),
        statistical_score=round(stat_score, 3),
        final_score=round(final_score, 3),
        evidence=evidence,
    )


def compute_fleet_baseline(stats_df: pd.DataFrame) -> Dict:
    """Compute fleet-wide percentiles for relative anomaly scoring."""
    return {
        "dwell_unknown_p50": float(stats_df["dwell_unknown_pings"].quantile(0.50)),
        "dwell_unknown_p90": float(stats_df["dwell_unknown_pings"].quantile(0.90)),
        "dwell_unknown_p95": float(stats_df["dwell_unknown_pings"].quantile(0.95)),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(gps_path: Path, zones_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    zones = load_reference(zones_path)

    # Step 1: load + enrich
    gps_df = pd.read_csv(gps_path)
    gps_df = enrich_with_h3(gps_df, zones)
    gps_df.to_csv(output_dir / "gps_enriched.csv", index=False)

    # Step 2: DuckDB aggregation
    logger.info("Running DuckDB aggregation...")
    con = duckdb.connect()
    con.register("gps", gps_df)
    stats_df = con.execute(ANOMALY_QUERY).df()
    logger.info(f"  Aggregated to {len(stats_df)} truck-day rows")

    # Step 3: baseline + scoring
    baseline = compute_fleet_baseline(stats_df)
    logger.info(f"  Fleet baseline: {baseline}")

    decisions = [score_truck_day(row, baseline) for _, row in stats_df.iterrows()]
    decisions_df = pd.DataFrame([asdict(d) for d in decisions])

    # Merge stats into decisions for full audit trail
    stats_df["date"] = stats_df["date"].astype(str)
    decisions_df = decisions_df.merge(
        stats_df, on=["truck_id", "date"], how="left",
        suffixes=("", "_raw")
    )

    decisions_df.to_csv(output_dir / "anomaly_decisions.csv", index=False)
    return gps_df, decisions_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(decisions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
    """Compare predicted anomalies vs ground truth labels."""
    decisions_df = decisions_df.copy()
    # Normalize date: drop time portion if any
    decisions_df["date"] = pd.to_datetime(decisions_df["date"]).dt.date.astype(str)
    ground_truth_df = ground_truth_df.copy()
    ground_truth_df["date"] = pd.to_datetime(ground_truth_df["date"]).dt.date.astype(str)

    merged = decisions_df.merge(
        ground_truth_df, on=["truck_id", "date"], how="inner"
    )

    tp = ((merged["is_anomaly_x"]) & (merged["is_anomaly_y"])).sum()
    fp = ((merged["is_anomaly_x"]) & (~merged["is_anomaly_y"])).sum()
    fn = ((~merged["is_anomaly_x"]) & (merged["is_anomaly_y"])).sum()
    tn = ((~merged["is_anomaly_x"]) & (~merged["is_anomaly_y"])).sum()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 0.001)

    return {
        "total_truck_days": len(merged),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data_dir = Path("data")
    output_dir = Path("output")

    gps_df, decisions_df = run_pipeline(
        gps_path=data_dir / "fleet_gps.csv",
        zones_path=data_dir / "zones.json",
        output_dir=output_dir,
    )

    # Evaluate
    gt = pd.read_csv(data_dir / "ground_truth.csv")
    metrics = evaluate(decisions_df, gt)

    print(f"\n{'='*55}")
    print(f"ANOMALY DETECTION RESULTS")
    print(f"{'='*55}")
    print(f"Total truck-days:  {metrics['total_truck_days']}")
    print(f"Flagged anomalies: {decisions_df['is_anomaly'].sum()}")
    print(f"  - high:     {(decisions_df['severity'] == 'high').sum()}")
    print(f"  - medium:   {(decisions_df['severity'] == 'medium').sum()}")
    print(f"  - low:      {(decisions_df['severity'] == 'low').sum()}")
    print(f"\nPrecision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"\nConfusion:")
    print(f"  TP={metrics['true_positives']}  FP={metrics['false_positives']}")
    print(f"  FN={metrics['false_negatives']}  TN={metrics['true_negatives']}")

    print(f"\nSample high-severity detections:")
    high = decisions_df[decisions_df["severity"] == "high"].head(3)
    for _, r in high.iterrows():
        print(f"  {r['truck_id']} | {r['date']} | "
              f"score={r['final_score']:.2f} | reason: {r['primary_reason']}")
