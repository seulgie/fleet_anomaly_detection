# Fleet GPS Anomaly Detection — 2018 → 2026

**Reimagining a 2018 project with today's stack.**

In 2018, I built a GPS anomaly detection system for a construction fleet
at Kolon Benit (Seoul) — detecting unauthorized routes, off-hours operations,
and suspicious dwell patterns. The stack was Python + PostGIS + manual SQL loops.

Eight years later, I rebuilt the same system from scratch to see what would
actually change. Here's what I found.

---

## The problem (unchanged)

Construction fleets generate large volumes of GPS telemetry. Most of it is normal:
trucks travelling between sites, quarries, and disposal zones during work hours.

But a small fraction reveals operational risk:
- Trucks operating after hours (fuel theft, unauthorized side jobs)
- Routes deviating into restricted areas (parks, residential streets)
- Excessive dwell times at unknown locations (suspicious activity)

The original 2018 system worked. But it was slow, expensive to maintain,
and required PostGIS infrastructure no one on the team really understood.

---

## What changed in 2026

| Layer | 2018 | 2026 |
|------:|:------|:------|
| Spatial indexing | PostGIS polygon joins | **H3** uniform hex grid |
| Aggregation | SQL loops in Python | **DuckDB** in-memory analytical SQL |
| Infrastructure | Server with PostGIS | **Zero**: runs on a laptop |
| Scoring | Complex rules-only | **Hybrid**: hard rules + statistical baseline |
| Visualization | Static QGIS exports | **Folium** interactive maps |
| Cost to run | Cloud server + DBA time | **€0**, local Python |

The interesting part: **the architecture is simpler now, not more complex.**
The 2026 version has fewer moving parts than the 2018 version did, despite
doing more work.

---

## Results

On 14 days of synthetic fleet data (25 trucks, ~37k GPS pings):

```
Precision: 1.000
Recall:    0.821
F1:        0.901

Confusion matrix:
  TP=32  FP=0
  FN=7   TN=261
```

**Zero false positives.** Recall is 82% — we miss 7 of 39 ground-truth anomalies,
mostly subtle off-route events without sustained dwell.

The deliberate trade-off: in operational fleet management, false positives
destroy operator trust. A site manager who gets 5 false alerts stops opening
the dashboard. The system prioritizes precision; recall improvements come
from review feedback loops, not from cranking the threshold.

---

## Architecture

```
GPS telemetry (CSV/stream)
        │
        ▼
┌──────────────────────┐
│  H3 spatial indexing │  ~0.1 km² hex cells (resolution 9)
│  (anomaly_pipeline)  │  Cell lookup is O(1), no spatial joins needed
└──────────────────────┘
        │ enriched pings with zone classification
        ▼
┌──────────────────────┐
│  DuckDB aggregation  │  truck-day stats: restricted hits, off-hours,
│  (SQL in pure Python)│  dwell time, route coverage
└──────────────────────┘
        │ truck-day feature table
        ▼
┌──────────────────────┐
│  Hybrid scoring      │  Rule layer:  hard violations → score = 1.0
│  (rule + statistical)│  Stat layer:  fleet-relative dwell outlier
│                      │  Combined: max(rule, 0.7 × stat)
└──────────────────────┘
        │ decisions with severity + reason codes
        ▼
┌──────────────────────┐
│  Folium maps         │  Color-coded trajectories, layer toggles,
│                      │  hover tooltips with full reason codes
└──────────────────────┘
```

---

## Why hybrid scoring, not ML?

The original 2018 system was rule-based. I considered going ML-first in the
rebuild — Isolation Forest on trajectory embeddings, maybe an autoencoder.

Decided against it. Three reasons:

**1. Label scarcity.** Real fleet anomaly labels are rare. The few you get
are biased (only obvious ones get reported).

**2. Operator trust.** A site manager needs to know *why* a truck was flagged.
"Score 0.87" is not actionable. "Entered Bois de Boulogne for 18 minutes
at 14:23" is.

**3. Deployment friction.** A rule + statistical baseline can be deployed
on Friday afternoon. An ML model needs versioning, monitoring, retraining,
drift detection — none of which exists yet at most fleet operators.

The hybrid model means:
- Hard rules fire on known violations (restricted zones, off-hours activity)
- Statistical layer catches *unknown* patterns (dwell time > 90th percentile)
- ML upgrade is straightforward once labeled data accumulates

This is the same lesson from the 2018 project, just expressed with newer tools:
*operational reliability over algorithmic sophistication.*

---

## What's reusable

The data model and pipeline shape is domain-agnostic. The same H3 + DuckDB +
hybrid scoring pattern applies to:

- Ride-share fleet monitoring (Uber-style mobility data)
- Delivery vehicle compliance (last-mile logistics)
- Geofence-based campaign attribution (mobility ad-tech)
- Insurance telematics (driving behavior scoring)

I've worked with several of these problem classes in past roles — the substrate
is GPS telemetry, the problem is always operational anomaly detection on
messy streams.

---

## Project structure

```
fleet-anomaly-detection/
├── src/
│   ├── generate_fleet_data.py    # Synthetic GPS + ground truth
│   ├── anomaly_pipeline.py       # H3 enrichment, DuckDB, scoring
│   └── visualize.py              # Folium maps
├── data/
│   ├── fleet_gps.csv             # 36k GPS pings
│   ├── ground_truth.csv          # Labeled anomalies for evaluation
│   └── zones.json                # Site/quarry/restricted zone definitions
└── output/
    ├── anomaly_decisions.csv     # Truck-day decisions with reason codes
    ├── fleet_overview.html       # Interactive map (open in browser)
    └── anomaly_drilldown.html    # High-severity zoom
```

---

## Running it

```bash
pip install h3 duckdb folium pandas
python src/generate_fleet_data.py   # ~5 seconds
python src/anomaly_pipeline.py      # ~3 seconds
python src/visualize.py             # ~2 seconds
open output/fleet_overview.html
```

Total runtime end-to-end: **under 15 seconds on a laptop.** The 2018 PostGIS
setup took ~10 minutes for the same workload.

---

## What I'd add next

Three honest next steps if this were going to production:

1. **Real-time streaming variant** — same H3 cells, same scoring logic,
   wrapped in a Kafka consumer. Batch and streaming share ~80% of code.

2. **Operator feedback loop** — flagged anomalies need a "review" status
   (true positive / false positive / unclear). This data is what enables
   the eventual ML upgrade.

3. **Zone definition UI** — zones are JSON today. In real ops, the site
   manager needs to draw geofences on a map.

---

## Background

I built the original 2018 version at Kolon Benit while working on
GIS-driven operational analytics in Seoul. Today I focus on mobility,
geospatial, and operational data systems in Paris.

The thing I keep relearning: real-world operational systems don't fail
because models are wrong. They fail because the data is messy, the
stakeholders don't trust the output, and no one knows what to do when
something gets flagged. The architecture has to address all three.
