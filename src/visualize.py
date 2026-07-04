"""
src/visualize.py
----------------
Folium-based visualization for fleet anomaly detection.

Two interactive HTML maps:
  - fleet_overview.html: full fleet, color-coded by anomaly status
  - anomaly_drilldown.html: focus on flagged truck-days

Color encodes meaning:
  green = normal trajectory
  red   = high-severity anomaly
  orange = medium-severity (statistical outlier)
"""

import json
import logging
from pathlib import Path
import pandas as pd
import folium
from folium import plugins

logger = logging.getLogger(__name__)


def load_data(data_dir: Path, output_dir: Path):
    gps = pd.read_csv(output_dir / "gps_enriched.csv")
    gps["timestamp"] = pd.to_datetime(gps["timestamp"])
    gps["date"] = pd.to_datetime(gps["date"]).dt.date.astype(str)
    decisions = pd.read_csv(output_dir / "anomaly_decisions.csv")
    decisions["date"] = pd.to_datetime(decisions["date"]).dt.date.astype(str)
    with open(data_dir / "zones.json") as f:
        zones = json.load(f)
    return gps, decisions, zones


def add_zone_markers(m, zones):
    zone_layer = folium.FeatureGroup(name="Zones", show=True)
    icon_map = {
        "depot":    ("home", "blue"),
        "site":     ("wrench", "green"),
        "quarry":   ("industry", "darkgreen"),
        "disposal": ("trash", "gray"),
    }
    for loc in zones["authorized_locations"]:
        icon_name, color = icon_map.get(loc["type"], ("info-sign", "gray"))
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"<b>{loc['name']}</b><br>type: {loc['type']}",
            tooltip=loc["name"],
            icon=folium.Icon(icon=icon_name, prefix="fa", color=color),
        ).add_to(zone_layer)
    for r in zones["restricted_zones"]:
        folium.Circle(
            location=[r["lat"], r["lon"]],
            radius=r["radius_km"] * 1000,
            color="#d32f2f", fill=True, fillColor="#d32f2f",
            fillOpacity=0.25, weight=2,
            popup=f"<b>RESTRICTED: {r['name']}</b>",
            tooltip=f"⛔ {r['name']}",
        ).add_to(zone_layer)
    zone_layer.add_to(m)


def add_trajectories(m, gps, decisions, sample_normal=8):
    decisions_lookup = decisions.set_index(["truck_id", "date"]).to_dict("index")
    anomaly_layer = folium.FeatureGroup(name="Anomalies (flagged)", show=True)
    normal_layer = folium.FeatureGroup(name="Normal (sample)", show=False)

    all_td = gps[["truck_id", "date"]].drop_duplicates()
    flagged = decisions[decisions["is_anomaly"]][["truck_id", "date"]]
    normal_pool = all_td.merge(flagged, how="left", indicator=True)
    normal_pool = normal_pool[normal_pool["_merge"] == "left_only"]
    normal_sample = normal_pool.sample(min(sample_normal, len(normal_pool)),
                                       random_state=1)

    for _, row in normal_sample.iterrows():
        traj = gps[(gps["truck_id"] == row["truck_id"]) &
                   (gps["date"] == row["date"])].sort_values("timestamp")
        if len(traj) < 2:
            continue
        coords = traj[["latitude", "longitude"]].values.tolist()
        folium.PolyLine(
            coords, color="#2e7d32", weight=2, opacity=0.6,
            tooltip=f"{row['truck_id']} | {row['date']} | NORMAL",
        ).add_to(normal_layer)

    for _, row in flagged.iterrows():
        traj = gps[(gps["truck_id"] == row["truck_id"]) &
                   (gps["date"] == row["date"])].sort_values("timestamp")
        if len(traj) < 2:
            continue
        coords = traj[["latitude", "longitude"]].values.tolist()
        d = decisions_lookup.get((row["truck_id"], row["date"]), {})
        severity = d.get("severity", "medium")
        color = "#c62828" if severity == "high" else "#ef6c00"
        weight = 4 if severity == "high" else 3
        tooltip_html = (
            f"<b>{row['truck_id']} | {row['date']}</b><br>"
            f"severity: <b>{severity}</b><br>"
            f"score: {d.get('final_score', 0):.2f}<br>"
            f"reason: {d.get('primary_reason', 'unknown')}"
        )
        folium.PolyLine(coords, color=color, weight=weight, opacity=0.85,
                        tooltip=tooltip_html).add_to(anomaly_layer)
        if severity == "high":
            folium.CircleMarker(
                location=coords[0], radius=4, color=color, fill=True,
                fillColor=color,
                tooltip=f"START: {row['truck_id']} {row['date']}",
            ).add_to(anomaly_layer)

    anomaly_layer.add_to(m)
    normal_layer.add_to(m)


def build_fleet_map(gps, decisions, zones, output_path):
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=10,
                   tiles="CartoDB positron")
    add_zone_markers(m, zones)
    add_trajectories(m, gps, decisions, sample_normal=8)

    n_total = len(decisions)
    n_flagged = decisions["is_anomaly"].sum()
    n_high = (decisions["severity"] == "high").sum()
    title = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background: white; padding: 12px 16px; border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15); font-family: sans-serif;
                font-size: 13px; max-width: 380px;">
      <div style="font-weight: 600; margin-bottom: 4px; font-size: 14px;">
        Fleet Anomaly Detection — Greater Paris
      </div>
      <div style="color: #555; line-height: 1.5;">
        {n_total} truck-days · <b style="color:#c62828;">{n_flagged} flagged</b>
        ({n_high} high severity)<br>
        H3 spatial indexing + DuckDB + hybrid rule/statistical scoring
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title))
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    m.save(str(output_path))
    logger.info(f"Map saved: {output_path}")


def build_drilldown_map(gps, decisions, zones, output_path):
    high = decisions[decisions["severity"] == "high"]
    if len(high) == 0:
        return
    first = high.iloc[0]
    first_traj = gps[(gps["truck_id"] == first["truck_id"]) &
                     (gps["date"] == first["date"])]
    center = ([first_traj["latitude"].mean(), first_traj["longitude"].mean()]
              if len(first_traj) > 0 else [48.8566, 2.3522])
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    add_zone_markers(m, zones)
    decisions_lookup = decisions.set_index(["truck_id", "date"]).to_dict("index")
    for _, row in high.iterrows():
        traj = gps[(gps["truck_id"] == row["truck_id"]) &
                   (gps["date"] == row["date"])].sort_values("timestamp")
        if len(traj) < 2:
            continue
        coords = traj[["latitude", "longitude"]].values.tolist()
        d = decisions_lookup.get((row["truck_id"], row["date"]), {})
        folium.PolyLine(
            coords, color="#c62828", weight=4, opacity=0.9,
            tooltip=f"<b>{row['truck_id']} | {row['date']}</b><br>"
                    f"score: {d.get('final_score', 0):.2f}<br>"
                    f"reason: {d.get('primary_reason', '')}",
        ).add_to(m)
    plugins.Fullscreen().add_to(m)
    m.save(str(output_path))
    logger.info(f"Drilldown saved: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    gps, decisions, zones = load_data(Path("data"), Path("output"))
    build_fleet_map(gps, decisions, zones, Path("output/fleet_overview.html"))
    build_drilldown_map(gps, decisions, zones, Path("output/anomaly_drilldown.html"))
    print(f"\n✓ Maps generated in output/")
