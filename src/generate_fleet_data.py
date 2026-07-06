"""
src/generate_fleet_data.py
---------------------------
Generate synthetic GPS telemetry for a construction fleet operating
in the Greater Paris area. Mimics the operational reality of the
original 2018 project: trucks moving between construction sites,
quarries, and waste disposal zones — with both compliant and
non-compliant behaviour patterns embedded for evaluation.

Scenario:
    A fleet of 25 dump trucks operating 6 days/week.
    Authorized routes connect 5 construction sites, 3 quarries,
    2 waste disposal zones, and 1 depot.

    Most trucks follow expected routes during work hours.
    A minority exhibit anomalies we want to detect:
      - Off-hours operation (fuel theft signal)
      - Off-route deviations (unauthorized side jobs)
      - Excessive dwell at unknown locations
      - Geofence violations (entering restricted zones)

This data is realistic enough to demonstrate the pipeline,
synthetic enough to be reproducible.
"""

import math
import random
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Geographic setup — Greater Paris construction zones
# ---------------------------------------------------------------------------

DEPOT = {"name": "depot_central", "lat": 48.9123, "lon": 2.3567, "type": "depot"}

CONSTRUCTION_SITES = [
    {"name": "site_la_defense",     "lat": 48.8920, "lon": 2.2380, "type": "site"},
    {"name": "site_saint_denis",    "lat": 48.9362, "lon": 2.3574, "type": "site"},
    {"name": "site_bercy",          "lat": 48.8400, "lon": 2.3826, "type": "site"},
    {"name": "site_ivry",           "lat": 48.8128, "lon": 2.3878, "type": "site"},
    {"name": "site_nanterre",       "lat": 48.8924, "lon": 2.2065, "type": "site"},
]

QUARRIES = [
    {"name": "quarry_cergy",        "lat": 49.0382, "lon": 2.0780, "type": "quarry"},
    {"name": "quarry_meaux",        "lat": 48.9601, "lon": 2.8783, "type": "quarry"},
    {"name": "quarry_evry",         "lat": 48.6298, "lon": 2.4416, "type": "quarry"},
]

DISPOSAL_ZONES = [
    {"name": "disposal_villeneuve", "lat": 48.7370, "lon": 2.4309, "type": "disposal"},
    {"name": "disposal_bonneuil",   "lat": 48.7665, "lon": 2.4870, "type": "disposal"},
]

# Restricted zones — trucks should NEVER enter these
RESTRICTED_ZONES = [
    # Bois de Boulogne - west of Paris, far from construction sites in east
    {"name": "restricted_bois_boulogne", "lat": 48.8615, "lon": 2.2530,
     "radius_km": 0.8, "type": "park"},
    # Quiet residential pocket in Montreuil - off main truck routes
    {"name": "restricted_residential_montreuil", "lat": 48.8615, "lon": 2.4480,
     "radius_km": 0.5, "type": "residential"},
]

ALL_AUTHORIZED = [DEPOT] + CONSTRUCTION_SITES + QUARRIES + DISPOSAL_ZONES

# Suspicious off-route waypoints (where anomalous trucks "stop")
SUSPICIOUS_LOCATIONS = [
    {"name": "unknown_warehouse_a",  "lat": 48.7950, "lon": 2.4520},
    {"name": "unknown_lot_b",        "lat": 48.9450, "lon": 2.4100},
    {"name": "private_address_c",    "lat": 48.7720, "lon": 2.3380},
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------

def interpolate_route(start, end, n_points, noise_m=15):
    """Linearly interpolate GPS points between two locations with realistic noise."""
    points = []
    for i in range(n_points):
        t = i / max(n_points - 1, 1)
        lat = start["lat"] + (end["lat"] - start["lat"]) * t
        lon = start["lon"] + (end["lon"] - start["lon"]) * t
        # Add GPS noise (~15m typical)
        lat += random.gauss(0, noise_m / 111000)
        lon += random.gauss(0, noise_m / (111000 * math.cos(math.radians(lat))))
        points.append((lat, lon))
    return points


def generate_normal_day(truck_id, date):
    """Normal operating day: depot → site → quarry → site → disposal → depot."""
    points = []

    # Work hours: 7:00 - 17:00
    current_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=7)

    # Random route sequence
    site = random.choice(CONSTRUCTION_SITES)
    quarry = random.choice(QUARRIES)
    disposal = random.choice(DISPOSAL_ZONES)
    sequence = [DEPOT, site, quarry, site, disposal, DEPOT]

    for i in range(len(sequence) - 1):
        origin = sequence[i]
        dest = sequence[i + 1]
        dist = haversine_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"])

        # Travel time (avg 35 km/h in urban areas)
        travel_minutes = max(10, int(dist / 35 * 60))
        n_pings = max(5, travel_minutes // 2)  # ping every ~2 min

        route_points = interpolate_route(origin, dest, n_pings)

        for lat, lon in route_points:
            points.append({
                "truck_id": truck_id,
                "timestamp": current_time,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(random.uniform(25, 55), 1),
            })
            current_time += timedelta(minutes=2)

        # Dwell at destination (loading/unloading)
        dwell_minutes = random.randint(15, 45)
        for _ in range(dwell_minutes // 5):
            points.append({
                "truck_id": truck_id,
                "timestamp": current_time,
                "latitude": round(dest["lat"] + random.gauss(0, 0.0001), 6),
                "longitude": round(dest["lon"] + random.gauss(0, 0.0001), 6),
                "speed_kmh": 0.0,
            })
            current_time += timedelta(minutes=5)

    return points


def generate_anomalous_day(truck_id, date, anomaly_type):
    """Generate a day with embedded anomaly."""
    points = generate_normal_day(truck_id, date)

    if anomaly_type == "off_hours":
        # Add evening activity (20:00-22:00)
        evening = datetime.combine(date, datetime.min.time()) + timedelta(hours=20)
        suspicious = random.choice(SUSPICIOUS_LOCATIONS)
        route = interpolate_route(DEPOT, suspicious, 20)
        for lat, lon in route:
            points.append({
                "truck_id": truck_id,
                "timestamp": evening,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(random.uniform(30, 50), 1),
            })
            evening += timedelta(minutes=3)

    elif anomaly_type == "off_route":
        # Detour to suspicious location mid-day
        midday = datetime.combine(date, datetime.min.time()) + timedelta(hours=12)
        suspicious = random.choice(SUSPICIOUS_LOCATIONS)
        detour = interpolate_route(random.choice(CONSTRUCTION_SITES), suspicious, 15)
        for lat, lon in detour:
            points.append({
                "truck_id": truck_id,
                "timestamp": midday,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(random.uniform(20, 45), 1),
            })
            midday += timedelta(minutes=2)
        # Long dwell at suspicious location (~60 min)
        for _ in range(12):
            points.append({
                "truck_id": truck_id,
                "timestamp": midday,
                "latitude": round(suspicious["lat"] + random.gauss(0, 0.0002), 6),
                "longitude": round(suspicious["lon"] + random.gauss(0, 0.0002), 6),
                "speed_kmh": 0.0,
            })
            midday += timedelta(minutes=5)

    elif anomaly_type == "restricted_zone":
        # Pass through restricted zone
        afternoon = datetime.combine(date, datetime.min.time()) + timedelta(hours=14)
        restricted = random.choice(RESTRICTED_ZONES)
        route = interpolate_route(random.choice(CONSTRUCTION_SITES), restricted, 10)
        for lat, lon in route:
            points.append({
                "truck_id": truck_id,
                "timestamp": afternoon,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(random.uniform(15, 30), 1),
            })
            afternoon += timedelta(minutes=2)

    return points


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_fleet_data(n_trucks=25, n_days=14, anomaly_rate=0.12):
    """
    Generate fleet GPS data with embedded anomalies for evaluation.

    Returns:
        gps_df: GPS telemetry rows
        ground_truth_df: per-truck-per-day anomaly labels (for evaluation)
    """
    start_date = datetime(2024, 11, 4).date()  # Monday
    all_points = []
    ground_truth = []
    anomaly_types = ["off_hours", "off_route", "restricted_zone"]

    for truck_idx in range(1, n_trucks + 1):
        truck_id = f"TRK-{truck_idx:03d}"
        for day_offset in range(n_days):
            date = start_date + timedelta(days=day_offset)
            # Skip Sundays
            if date.weekday() == 6:
                continue

            if random.random() < anomaly_rate:
                anomaly_type = random.choice(anomaly_types)
                points = generate_anomalous_day(truck_id, date, anomaly_type)
                ground_truth.append({
                    "truck_id": truck_id, "date": date,
                    "is_anomaly": True, "anomaly_type": anomaly_type,
                })
            else:
                points = generate_normal_day(truck_id, date)
                ground_truth.append({
                    "truck_id": truck_id, "date": date,
                    "is_anomaly": False, "anomaly_type": None,
                })

            all_points.extend(points)

    gps_df = pd.DataFrame(all_points)
    gt_df = pd.DataFrame(ground_truth)
    return gps_df, gt_df


def save_reference_data(output_dir: Path):
    """Save geofence/zone definitions as JSON for reuse in pipeline."""
    reference = {
        "depot": DEPOT,
        "construction_sites": CONSTRUCTION_SITES,
        "quarries": QUARRIES,
        "disposal_zones": DISPOSAL_ZONES,
        "restricted_zones": RESTRICTED_ZONES,
        "authorized_locations": ALL_AUTHORIZED,
        "work_hours": {"start": 7, "end": 17},
    }
    with open(output_dir / "zones.json", "w") as f:
        json.dump(reference, f, indent=2, default=str)


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    print("Generating fleet GPS data...")
    gps_df, gt_df = generate_fleet_data(n_trucks=25, n_days=14, anomaly_rate=0.12)

    gps_df.to_csv(output_dir / "fleet_gps.csv", index=False)
    gt_df.to_csv(output_dir / "ground_truth.csv", index=False)
    save_reference_data(output_dir)

    print(f"  GPS pings:        {len(gps_df):,}")
    print(f"  Unique trucks:    {gps_df['truck_id'].nunique()}")
    print(f"  Days covered:     {gps_df['timestamp'].dt.date.nunique()}")
    print(f"  Anomalous days:   {gt_df['is_anomaly'].sum()} / {len(gt_df)} "
          f"({gt_df['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Anomaly types:    {gt_df[gt_df['is_anomaly']]['anomaly_type'].value_counts().to_dict()}")
