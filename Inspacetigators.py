import os
import re
import subprocess
import sys

# ---- Self-launcher: lets you run locally with plain "python app.py" and safely skips on cloud ----
if os.environ.get("QUAKE_APP_CHILD") != "1" and "streamlit" not in sys.argv[0].lower():
    os.environ["QUAKE_APP_CHILD"] = "1"
    cred = os.path.expanduser("~/.streamlit/credentials.toml")
    if not os.path.exists(cred):
        os.makedirs(os.path.dirname(cred), exist_ok=True)
        with open(cred, "w") as fh:
            fh.write('[general]\nemail = ""\n')
    code = (
        "import asyncio, sys\n"
        "if sys.platform == 'win32':\n"
        "    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())\n"
        "from streamlit.web import cli\n"
        f"sys.argv = ['streamlit', 'run', {__file__!r}]\n"
        "cli.main()\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code])
    except KeyboardInterrupt:
        pass
    sys.exit()
# ------------------------------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(page_title="Japan Earthquake Explorer", layout="wide")

# ---------------- Region Polygons, Colors & Zoom Bounds ----------------
REGION_POLYS = {
    "Hokkaido": [(140.0, 41.5), (140.3, 42.4), (140.4, 43.3), (141.4, 43.4), (141.7, 44.8), (141.6, 45.5), (142.9, 44.7), (144.0, 44.1), (145.3, 44.0), (145.6, 43.3), (144.4, 42.9), (143.0, 42.0), (141.4, 42.6), (140.9, 42.5), (140.8, 41.8), (140.0, 41.5)],
    "Tohoku": [(141.5, 40.5), (142.0, 39.6), (141.9, 38.9), (141.0, 38.3), (141.0, 37.0), (140.948, 36.896), (139.4, 37.1), (139.45, 37.8), (139.528, 38.422), (139.6, 38.5), (140.0, 39.2), (139.9, 39.9), (140.0, 40.5), (140.0, 40.85), (140.9, 41.2), (141.4, 41.2), (141.5, 40.5)],
    "Kanto": [(140.8, 35.7), (140.4, 35.2), (139.9, 35.0), (140.1, 35.5), (139.7, 35.6), (139.6, 35.3), (139.7, 35.15), (139.15, 35.15), (139.0, 34.82), (139.0, 35.0), (138.9, 35.5), (138.6, 36.0), (138.7, 36.5), (139.0, 36.9), (139.4, 37.1), (140.948, 36.896), (140.6, 36.2), (140.8, 35.7)],
    "Chubu": [(138.8, 35.0), (137.4, 34.65), (137.0, 34.69), (137.0, 34.7), (136.6, 35.3), (136.2, 35.9), (136.022, 35.824), (136.1, 36.0), (136.7, 36.9), (137.0, 37.4), (137.3, 37.5), (137.1, 37.0), (137.6, 36.9), (138.5, 37.4), (139.0, 37.85), (139.522, 38.416), (139.45, 37.8), (139.4, 37.1), (139.0, 36.9), (138.7, 36.5), (138.6, 36.0), (138.9, 35.5), (139.0, 35.0), (139.0, 34.82), (138.9, 34.6), (138.8, 35.0)],
    "Kansai": [(136.9, 34.3), (136.3, 33.8), (135.8, 33.45), (135.4, 33.6), (135.1, 34.3), (135.4, 34.6), (134.7, 34.65), (134.6, 34.619), (134.6, 35.623), (134.9, 35.65), (135.9, 35.55), (136.035, 35.853), (136.022, 35.824), (136.2, 35.9), (136.6, 35.3), (137.0, 34.7), (137.0, 34.69), (136.9, 34.7), (136.9, 34.3)],
    "Chugoku": [(132.4, 34.2), (131.4, 34.0), (130.9, 33.95), (131.0, 34.4), (131.8, 34.7), (132.8, 35.4), (133.8, 35.55), (134.6, 35.623), (134.6, 34.619), (133.9, 34.4), (132.4, 34.2)],
    "Shikoku": [(132.4, 33.4), (132.5, 34.0), (133.5, 34.3), (134.6, 34.2), (134.7, 33.8), (134.2, 33.3), (133.5, 33.4), (133.0, 32.8), (132.5, 32.9), (132.4, 33.4)],
    "Kyushu": [(130.9, 33.9), (131.7, 33.6), (131.9, 32.8), (131.4, 31.5), (130.7, 31.0), (130.2, 31.3), (130.2, 32.2), (130.6, 32.6), (129.9, 32.7), (129.8, 33.3), (130.3, 33.6), (130.9, 33.9)],
    "Okinawa": [(127.65, 26.1), (127.9, 26.6), (128.2, 26.9), (128.3, 26.85), (127.9, 26.2), (127.65, 26.1)],
    "Ogasawara": [(142.8, 26.0), (142.8, 28.0), (140.5, 28.0), (140.5, 26.0), (142.8, 26.0)],
}

REGION_COLORS = {
    "Hokkaido": "#38bdf8",
    "Tohoku": "#34d399",
    "Kanto": "#facc15",
    "Chubu": "#fb923c",
    "Kansai": "#c084fc",
    "Chugoku": "#f87171",
    "Shikoku": "#2dd4bf",
    "Kyushu": "#f59e0b",
    "Okinawa": "#94a3b8",
    "Ogasawara": "#ec4899",
}

REGION_BOUNDS = {
    "Hokkaido": ([139.0, 146.5], [41.0, 46.0]),
    "Tohoku": ([138.8, 143.0], [36.7, 42.0]),
    "Kanto": ([138.0, 142.0], [34.2, 37.5]),
    "Chubu": ([135.5, 140.2], [34.2, 38.8]),
    "Kansai": ([134.2, 137.5], [33.0, 36.2]),
    "Chugoku": ([130.5, 134.8], [33.5, 36.2]),
    "Shikoku": ([131.8, 135.2], [32.4, 34.6]),
    "Kyushu": ([129.0, 132.6], [30.5, 34.4]),
    "Okinawa": ([123.5, 130.5], [24.0, 28.5]),
    "Ogasawara": ([139.5, 143.5], [24.0, 28.5]),
}

OFF = "Offshore / other"
REGIONS = ["All of Japan"] + list(REGION_POLYS.keys()) + [OFF]

DEPTHS = {
    "Any depth": (0, 1000),
    "Shallow (under 30 km)": (0, 30),
    "Medium (30-70 km)": (30, 70),
    "Deep (70-300 km)": (70, 300),
    "Very deep (over 300 km)": (300, 1000),
}

PERIODS = {
    "Last 24 hours": 1,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 12 months": 365,
}

SPEEDS = {
    "Slow (1.0s / step)": 1000,
    "Normal (0.4s / step)": 400,
    "Fast (0.15s / step)": 150,
    "Turbo (0.05s / step)": 50,
}

RESEARCH_ARCHIVE = [
    ("2026-09-26 02:21:30+00:00", 3.6, 10.0, 32.800, 130.800, "10 km E of Kumamoto, Japan (EMSC 2065649)", "emsc_2065649"),
    ("2026-07-28 07:27:14+00:00", 6.8, 8.0, 32.618, 130.660, "2026 Kumamoto Region Earthquake, Japan (Hinagu Fault)", "us6000tgb9"),
    ("2021-05-01 01:27:27+00:00", 6.7, 47.0, 38.230, 141.665, "Miyagi-Oki Interplate Earthquake, Japan (Yoshida et al., 2022)", "arch_2021_05_miyagi"),
    ("2021-03-20 09:09:44+00:00", 7.0, 43.0, 38.452, 141.648, "Miyagi-Oki Interplate Earthquake, Japan (Yoshida et al., 2022)", "arch_2021_03_miyagi"),
    ("2020-05-11 23:58:00+00:00", 6.0, 42.0, 38.820, 141.850, "Miyagi-Oki Quasi-Repeating Earthquake, Japan (Yoshida et al., 2023)", "arch_2020_miyagi"),
    ("2016-04-14 15:03:46+00:00", 6.0, 10.0, 32.700, 130.780, "2016 Kumamoto Foreshock #2, Kyushu, Japan", "arch_2016_kumamoto_2"),
    ("2016-04-14 12:26:34+00:00", 6.2, 10.0, 32.782, 130.726, "2016 Kumamoto Foreshock #1, Kyushu, Japan", "arch_2016_kumamoto_1"),
    ("2015-05-12 21:12:58+00:00", 6.8, 39.0, 38.906, 142.032, "Miyagi-Oki Interplate Earthquake, Japan (Yoshida et al., 2023)", "arch_2015_miyagi"),
    ("2012-06-17 20:32:20+00:00", 6.3, 40.0, 38.920, 141.880, "Miyagi-Oki Quasi-Repeating Earthquake, Japan (Yoshida et al., 2023)", "arch_2012_miyagi"),
    ("2011-07-24 18:51:25+00:00", 6.3, 41.0, 38.890, 141.910, "Miyagi-Oki Quasi-Repeating Earthquake, Japan (Yoshida et al., 2023)", "arch_2011_07_miyagi"),
    ("2011-03-31 07:15:30+00:00", 6.1, 42.0, 38.950, 141.920, "Miyagi-Oki Post-Tohoku Earthquake, Japan (Yoshida et al., 2023)", "arch_2011_03_miyagi"),
    ("2011-03-11 05:46:24+00:00", 9.0, 29.0, 38.297, 142.373, "2011 Great Tohoku-Oki Earthquake, Japan", "arch_2011_tohoku"),
    ("2011-03-09 02:45:20+00:00", 7.3, 32.0, 38.435, 142.842, "2011 Tohoku-Oki Foreshock, Off Miyagi, Japan", "arch_2011_tohoku_fore"),
    ("2008-06-13 23:43:45+00:00", 6.9, 7.8, 39.030, 140.881, "2008 Iwate-Miyagi Nairiku Earthquake, Japan", "arch_2008_iwate"),
    ("2005-08-16 02:46:28+00:00", 7.2, 36.0, 38.276, 142.039, "2005 Miyagi-ken-Oki Earthquake, Japan (Nakata et al., 2023)", "arch_2005_miyagi"),
    ("2002-11-03 03:37:42+00:00", 6.4, 39.0, 38.890, 141.980, "2002 Miyagi-Oki Earthquake, Japan (Yoshida et al., 2023)", "arch_2002_miyagi"),
    ("1978-06-12 08:14:26+00:00", 7.4, 44.0, 38.190, 142.028, "1978 Miyagi-ken-Oki Earthquake, Japan (Nakata et al., 2023)", "arch_1978_miyagi"),
    ("1936-11-02 20:45:56+00:00", 7.4, 40.0, 38.260, 142.050, "1936 Miyagi-ken-Oki Earthquake, Japan (Nakata et al., 2023)", "arch_1936_miyagi"),
    ("1933-06-18 21:37:36+00:00", 7.1, 35.0, 38.250, 142.350, "1933 Miyagi-ken-Oki Earthquake, Japan (Nakata et al., 2023)", "arch_1933_miyagi"),
    ("1923-09-01 02:58:32+00:00", 8.0, 23.0, 35.330, 139.150, "1923 Great Kanto Earthquake, near Fujisawa/Yokohama, Japan", "arch_1923_kanto"),
    ("1897-02-20 05:50:00+00:00", 7.4, 40.0, 38.100, 141.900, "1897 Miyagi-ken-Oki Earthquake, Japan (Nakata et al., 2023)", "arch_1897_miyagi"),
]


# ---------------- Geometry & Region Helpers ----------------
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def dist_to_poly(x: float, y: float, poly: list[tuple[float, float]]) -> float:
    if point_in_poly(x, y, poly):
        return 0.0
    best = float("inf")
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy + 1e-12)))
        px, py = x1 + t * dx, y1 + t * dy
        d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if d < best:
            best = d
    return best


def region_of(lat: float, lon: float) -> str:
    if lat < 28.5 and lon > 139:
        return "Ogasawara"
    best, bd = OFF, 1.5
    for r, poly in REGION_POLYS.items():
        d = dist_to_poly(lon, lat, poly)
        if d < bd:
            best, bd = r, d
    return best


# ---------------- Live Data Fetchers + Deduplication ----------------
def fetch_usgs(start_str: str, min_mag: float) -> list[dict]:
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
        f"&starttime={start_str}&minmagnitude={min_mag}"
        "&minlatitude=24&maxlatitude=46&minlongitude=122&maxlongitude=146"
    )
    rows = []
    for f in requests.get(url, timeout=30).json().get("features", []):
        p = f["properties"]
        lon, lat, depth = f["geometry"]["coordinates"]
        if "Japan" not in (p.get("place") or ""):
            continue
        rows.append(
            dict(
                time=pd.to_datetime(p["time"], unit="ms", utc=True),
                mag=float(p["mag"]),
                depth=float(depth),
                lat=float(lat),
                lon=float(lon),
                place=p["place"],
                region=region_of(lat, lon),
                source="USGS",
                id=str(f["id"]),
            )
        )
    return rows


def fetch_jma(cutoff_dt: datetime, min_mag: float) -> list[dict]:
    url = "https://www.jma.go.jp/bosai/quake/data/list.json"
    rows = []
    for item in requests.get(url, timeout=20).json():
        mag_str = item.get("mag")
        cod = item.get("cod")
        if not mag_str or not cod:
            continue
        try:
            mag = float(mag_str)
        except ValueError:
            continue
        if mag < min_mag:
            continue
        m = re.match(r"([+-][\d.]+)([+-][\d.]+)([+-]\d+)?", cod)
        if not m:
            continue
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (24 <= lat <= 46 and 122 <= lon <= 146):
            continue
        depth = abs(int(m.group(3))) / 1000.0 if m.group(3) else 10.0
        t = pd.to_datetime(item.get("at") or item.get("rdt"), utc=True)
        if pd.isna(t) or t < cutoff_dt:
            continue
        place = item.get("en_anm") or item.get("anm") or "Japan (JMA)"
        if "Japan" not in place:
            place = f"{place}, Japan"
        rows.append(
            dict(
                time=t,
                mag=mag,
                depth=depth,
                lat=lat,
                lon=lon,
                place=place,
                region=region_of(lat, lon),
                source="JMA",
                id=f"jma_{item.get('eid', '')}",
            )
        )
    return rows


def fetch_emsc(start_str: str, min_mag: float) -> list[dict]:
    url = (
        "https://www.seismicportal.eu/fdsnws/event/1/query?format=json"
        f"&starttime={start_str}&minmag={min_mag}"
        "&minlat=24&maxlat=46&minlon=122&maxlon=146&limit=2000"
    )
    rows = []
    for f in requests.get(url, timeout=25).json().get("features", []):
        p = f["properties"]
        lon, lat, depth = f["geometry"]["coordinates"]
        region_txt = p.get("flynn_region") or ""
        if "JAPAN" not in region_txt.upper() and region_of(lat, lon) == OFF:
            continue
        rows.append(
            dict(
                time=pd.to_datetime(p["time"], utc=True),
                mag=float(p["mag"]),
                depth=abs(float(depth)),
                lat=float(lat),
                lon=float(lon),
                place=f"{region_txt.title()} (EMSC)",
                region=region_of(lat, lon),
                source="EMSC",
                id=f"emsc_{f['id']}",
            )
        )
    return rows


def deduplicate_quakes(df: pd.DataFrame, time_tol_sec: int = 90, dist_tol_deg: float = 0.45) -> pd.DataFrame:
    if df.empty:
        return df
    priority = {"USGS": 0, "JMA": 1, "EMSC": 2, "Archive": 3}
    df = df.copy()
    df["_prio"] = df["source"].map(priority).fillna(9)
    df = df.sort_values(["time", "_prio"]).reset_index(drop=True)

    keep = []
    times = df["time"].astype("int64") // 10**9
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    for i in range(len(df)):
        is_dup = False
        for j in range(len(keep) - 1, -1, -1):
            k = keep[j]
            if abs(times[i] - times[k]) > time_tol_sec:
                break
            if abs(lats[i] - lats[k]) <= dist_tol_deg and abs(lons[i] - lons[k]) <= dist_tol_deg:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    return df.iloc[keep].drop(columns=["_prio"]).reset_index(drop=True)


@st.cache_data(ttl=300)
def load(days: int, min_mag: float, include_archive: bool):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    start_str = cutoff.strftime("%Y-%m-%d")
    rows = []
    feed_status = {}

    for name, fetcher in (
        ("USGS", lambda: fetch_usgs(start_str, min_mag)),
        ("JMA", lambda: fetch_jma(cutoff, min_mag)),
        ("EMSC", lambda: fetch_emsc(start_str, min_mag)),
    ):
        try:
            fetched = fetcher()
            rows.extend(fetched)
            feed_status[name] = {"ok": True, "count": len(fetched), "error": None}
        except Exception as exc:
            feed_status[name] = {"ok": False, "count": 0, "error": str(exc)}

    if include_archive:
        arch_count = 0
        for t_str, mag, depth, lat, lon, place, eid in RESEARCH_ARCHIVE:
            if mag >= min_mag:
                arch_count += 1
                rows.append(
                    dict(
                        time=pd.to_datetime(t_str, utc=True),
                        mag=mag,
                        depth=depth,
                        lat=lat,
                        lon=lon,
                        place=place,
                        region=region_of(lat, lon),
                        source="Archive",
                        id=eid,
                    )
                )
        feed_status["Archive"] = {"ok": True, "count": arch_count, "error": None}

    cols = ["time", "mag", "depth", "lat", "lon", "place", "region", "source", "id"]
    if not rows:
        return pd.DataFrame(columns=cols), feed_status, datetime.now(timezone.utc)
    return deduplicate_quakes(pd.DataFrame(rows, columns=cols)), feed_status, datetime.now(timezone.utc)


if "zoom_rev" not in st.session_state:
    st.session_state.zoom_rev = 0
if "region_sel" not in st.session_state:
    st.session_state.region_sel = "All of Japan"

# Handle click selection on a region border trace from the previous run
map_event = st.session_state.get("japan_map", {})
pts = map_event.get("selection", {}).get("points", []) if isinstance(map_event, dict) else []
if pts:
    curve = pts[0].get("curve_number")
    region_list = list(REGION_POLYS.keys())
    if isinstance(curve, int) and 1 <= curve <= len(region_list):
        clicked_reg = region_list[curve - 1]
        st.session_state.region_sel = "All of Japan" if st.session_state.region_sel == clicked_reg else clicked_reg

# ---------------- Sidebar: Settings & Filters ----------------
st.sidebar.title("Settings")
with st.sidebar.expander("Data", expanded=True):
    period = st.selectbox("Time period", list(PERIODS), index=2)
    min_mag = st.selectbox("Minimum magnitude", [2.5, 3.5, 4.0, 4.5, 5.0, 6.0], index=2)
    include_archive = st.toggle("Include research archive quakes (1897–2026)", value=True)
    auto = st.toggle("Refresh every 5 minutes", value=True)
    if st.button("Reload data", use_container_width=True):
        st.cache_data.clear()
with st.sidebar.expander("Filters", expanded=True):
    region_sel = st.selectbox("Region (or click region on map)", REGIONS, key="region_sel")
    depth_sel = st.selectbox("Depth", list(DEPTHS))
    source_sel = st.selectbox("Source", ["All sources", "USGS", "JMA", "EMSC", "Archive"])
    search = st.text_input("Search place")
with st.sidebar.expander("Timeline Player", expanded=True):
    play_style = st.selectbox("Playback style", ["Cumulative (build up)", "Single step (event by event)"])
    speed_label = st.selectbox("Playback speed", list(SPEEDS), index=1)
    frame_ms = SPEEDS.get(speed_label, 400)
with st.sidebar.expander("Display", expanded=True):
    color_by = st.selectbox("Colour circles by", ["Depth", "Magnitude", "Age", "Source"])
    size = st.slider("Circle size", 2, 8, 4)
    labels = st.toggle("Label magnitude 6 and above", value=True)
    light = st.toggle("Light interface")
with st.sidebar.expander("Map", expanded=True):
    sea = st.slider(
        "Sea shown around Japan (degrees)",
        0.5,
        5.0,
        1.5,
        0.5,
        help="Sets the 'All of Japan' outer boundary, which also acts as the maximum zoom-out limit.",
    )
    if st.button("Reset zoom to Japan", use_container_width=True):
        st.session_state.region_sel = "All of Japan"
        st.session_state.zoom_rev += 1
        st.rerun()

if auto and st_autorefresh is not None:
    st_autorefresh(interval=300000, key="auto")

# ---------------- Dark / Light Theme Palette & CSS ----------------
C = (
    dict(
        bg="#f4f6fa", side="#ffffff", card="#ffffff", btn="#f1f5f9", hover="#e2e8f0",
        border="#cbd5e1", tx="#0f172a", muted="#475569", land="#efe9db", line="#94a3b8",
        ocean="#d8e8f6", marker_line="#1e293b", picked="#0f172a", subcard="#f8fafc",
    )
    if light
    else dict(
        bg="#070b14", side="#0d1424", card="#111a2e", btn="#0d1424", hover="#1e293b",
        border="#23304a", tx="#f1f5f9", muted="#94a3b8", land="#162033", line="#334155",
        ocean="#09111e", marker_line="#09111e", picked="#ffffff", subcard="#0c1322",
    )
)

css = f"""<style>
.stApp, [data-testid="stAppViewContainer"] {{ background: {C["bg"]}; color: {C["tx"]}; }}
[data-testid="stHeader"] {{ background: {C["bg"]} !important; border-bottom: 1px solid {C["border"]}; }}
[data-testid="stHeader"] button, [data-testid="stHeader"] svg {{ color: {C["tx"]} !important; fill: {C["tx"]} !important; }}
[data-testid="stSidebar"] {{ background: {C["side"]}; border-right: 1px solid {C["border"]}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {C["tx"]} !important; }}
.stApp [data-testid="stCaptionContainer"] p {{ color: {C["muted"]} !important; }}
[data-testid="stExpander"] {{ background: {C["card"]}; border: 1px solid {C["border"]} !important; border-radius: 10px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary span, [data-testid="stExpander"] summary svg {{
    color: {C["tx"]} !important; fill: {C["tx"]} !important;
}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input {{
    background: {C["card"]} !important; color: {C["tx"]} !important; border-color: {C["border"]} !important;
}}
div[data-baseweb="select"] svg {{ fill: {C["tx"]} !important; }}
div[data-baseweb="popover"], ul[role="listbox"] {{ background: {C["card"]} !important; border: 1px solid {C["border"]} !important; }}
li[role="option"] {{ background: {C["card"]} !important; color: {C["tx"]} !important; }}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {{ background: {C["hover"]} !important; color: {C["tx"]} !important; }}
.stButton > button, .stDownloadButton > button {{
    background: {C["card"]} !important; color: {C["tx"]} !important; border: 1px solid {C["border"]} !important; border-radius: 8px;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {C["hover"]} !important; border-color: {C["tx"]} !important; }}
[data-testid="stMetric"] {{ background: {C["card"]}; border: 1px solid {C["border"]}; border-radius: 10px; padding: 12px 16px; }}
.key-box, .detail-card {{
    background: {C["card"]}; border: 1px solid {C["border"]}; border-radius: 10px;
    padding: 14px 18px; margin: 10px 0 14px;
}}
.key-title {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: {C["muted"]}; margin-bottom: 8px; font-weight: 700; }}
.key-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-bottom: 12px; }}
.key-item {{
    background: {C["subcard"]}; border: 1px solid {C["border"]}; border-radius: 8px;
    padding: 8px 10px; display: flex; align-items: flex-start; gap: 9px;
}}
.swatch {{ display: inline-block; width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }}
.ring-swatch {{
    display: inline-block; width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0; margin-top: 3px;
    border: 2.5px solid {C["picked"]}; background: transparent;
}}
.rswatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }}
.key-item-title {{ font-weight: 700; font-size: 0.86rem; line-height: 1.25; }}
.key-item-desc {{ font-size: 0.76rem; color: {C["muted"]}; line-height: 1.3; margin-top: 2px; }}
.key-row {{ display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 8px; }}
.feed-pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.74rem; border: 1px solid {C["border"]}; margin-right: 6px; }}
</style>"""
st.markdown(css, unsafe_allow_html=True)

# ---------------- Load & Filter Data ----------------
st.title("Japan Earthquake Explorer")
try:
    df, feed_status, synced_at = load(PERIODS[period], min_mag, include_archive)
except Exception as err:
    st.error(f"Could not load earthquake data: {err}")
    st.stop()

badges = []
for src_name, info in feed_status.items():
    dot = "🟢" if info["ok"] else "🔴"
    badges.append(f'<span class="feed-pill">{dot} {src_name} ({info["count"]})</span>')
st.markdown(
    f'{"".join(badges)} <span style="color:{C["muted"]};font-size:0.78rem;">Last sync: {synced_at:%Y-%m-%d %H:%M UTC}</span>',
    unsafe_allow_html=True,
)

d0, d1 = DEPTHS[depth_sel]
df = df[(df.depth >= d0) & (df.depth < d1)]
if region_sel != "All of Japan":
    df = df[df.region == region_sel]
if source_sel != "All sources":
    df = df[df.source == source_sel]
if search:
    df = df[df.place.str.contains(search, case=False)]
if df.empty:
    st.warning("No earthquakes match these settings. Try a longer period or lower magnitude.")
    st.stop()

df = df.sort_values("time", ascending=False).reset_index(drop=True)
df_chrono = df.sort_values("time", ascending=True).reset_index(drop=True)

# ---------------- Summary Metrics & Coverage ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Earthquakes", len(df))
c2.metric("Strongest", f"M {df.mag.max():.1f}")
c3.metric("Average depth", f"{df.depth.mean():.0f} km")
c4.metric("Magnitude 5+", int((df.mag >= 5).sum()))

days_with_data = df["time"].dt.floor("D").nunique()
st.caption(
    f"Window: {period} · Region: {region_sel} · {days_with_data} of {PERIODS[period]} UTC days contain matching events · "
    f"Latest: M{df.iloc[0]['mag']:.1f} ({df.iloc[0]['place']}) at {df.iloc[0]['time']:%Y-%m-%d %H:%M UTC}"
)

now = datetime.now(timezone.utc)
SOURCE_COLORS = {"USGS": "#38bdf8", "JMA": "#34d399", "EMSC": "#fb923c", "Archive": "#c084fc"}


def point_color(q):
    if color_by == "Source":
        return SOURCE_COLORS.get(q.source, "#94a3b8")
    if color_by == "Magnitude":
        return ("#dc2626" if q.mag >= 7 else "#f97316" if q.mag >= 6
                else "#facc15" if q.mag >= 5 else "#38bdf8")
    if color_by == "Age":
        h = (now - q.time.to_pydatetime()).total_seconds() / 3600
        return ("#f43f5e" if h < 24 else "#fbbf24" if h < 168
                else "#22d3ee" if h < 720 else "#64748b")
    return ("#ef4444" if q.depth < 30 else "#f59e0b" if q.depth < 70
            else "#22d3ee" if q.depth < 300 else "#6366f1")


def make_quake_trace(sub_df):
    colors = [point_color(q) for q in sub_df.itertuples()]
    sizes = [max(6, (q.mag - 3) * size * 2) for q in sub_df.itertuples()]
    text_labels = [f"M{q.mag:.1f}" if (labels and q.mag >= 6) else "" for q in sub_df.itertuples()]
    hover = (
        "<b>M" + sub_df["mag"].round(1).astype(str) + "</b> (" + sub_df["region"] + " · " + sub_df["source"] + ")<br>"
        + sub_df["place"] + "<br>"
        + sub_df["time"].dt.strftime("%Y-%m-%d %H:%M UTC") + "<br>"
        + "Depth " + sub_df["depth"].round(1).astype(str) + " km"
    )
    return go.Scattergeo(
        lon=sub_df["lon"],
        lat=sub_df["lat"],
        text=text_labels,
        customdata=sub_df["id"],
        textposition="middle right",
        textfont=dict(color=C["tx"], size=11),
        hovertext=hover,
        hoverinfo="text",
        mode="markers+text" if labels else "markers",
        showlegend=False,
        marker=dict(size=sizes, color=colors, opacity=0.85, line=dict(width=0.7, color=C["marker_line"])),
    )


def make_ring_trace(focus_df):
    if focus_df.empty:
        return go.Scattergeo(lon=[], lat=[], mode="markers", hoverinfo="skip", showlegend=False)
    ring_sizes = [max(16, (q.mag - 3) * size * 2 + 8) for q in focus_df.itertuples()]
    return go.Scattergeo(
        lon=focus_df["lon"],
        lat=focus_df["lat"],
        mode="markers",
        hoverinfo="skip",
        showlegend=False,
        marker=dict(size=ring_sizes, color="rgba(0,0,0,0)", line=dict(width=2.5, color=C["picked"])),
    )


# ---------------- Build Chronological Video Frames with Time of Day ----------------
n_total = len(df_chrono)
max_frames = 120
end_indices = (
    list(range(1, n_total + 1))
    if n_total <= max_frames
    else sorted(set(np.linspace(1, n_total, max_frames, dtype=int).tolist()))
)

quake_trace_idx = 1 + len(REGION_POLYS)
ring_trace_idx = quake_trace_idx + 1

frames = []
steps = []
prev_end = 0
cumulative = play_style.startswith("Cumulative")

for idx, end_i in enumerate(end_indices):
    sub = df_chrono.iloc[:end_i] if cumulative else df_chrono.iloc[prev_end:end_i]
    newest_in_step = df_chrono.iloc[prev_end:end_i]
    prev_end = end_i

    cur_row = newest_in_step.iloc[-1]
    jst_dt = cur_row["time"] + timedelta(hours=9)
    h = jst_dt.hour
    tod = "Late Night" if h < 5 else "Morning" if h < 12 else "Afternoon" if h < 17 else "Evening" if h < 21 else "Night"
    ampm = jst_dt.strftime("%I:%M %p").lstrip("0")
    step_caption = (
        f"{jst_dt:%Y-%m-%d %H:%M:%S} JST ({ampm} · {tod}) / {cur_row['time']:%H:%M} UTC  |  "
        f"M{cur_row['mag']:.1f} {cur_row['region']} ({end_i}/{n_total})"
    )

    fname = f"frame_{idx}"
    frames.append(
        go.Frame(
            name=fname,
            traces=[quake_trace_idx, ring_trace_idx],
            data=[make_quake_trace(sub), make_ring_trace(newest_in_step)],
        )
    )
    steps.append(
        dict(
            method="animate",
            label=step_caption,
            args=[[fname], dict(mode="immediate", frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0))],
        )
    )

# ---------------- Map + Region Borders + Docked Player ----------------
region_traces = []
for reg, poly in REGION_POLYS.items():
    is_sel = region_sel == reg
    alpha = 0.30 if is_sel else (0.14 if region_sel == "All of Japan" else 0.05)
    region_traces.append(
        go.Scattergeo(
            lon=[p[0] for p in poly],
            lat=[p[1] for p in poly],
            mode="lines",
            fill="toself",
            fillcolor=hex_to_rgba(REGION_COLORS[reg], alpha),
            line=dict(color=REGION_COLORS[reg], width=3.0 if is_sel else 1.8),
            name=reg,
            hoverinfo="name",
            showlegend=False,
        )
    )

fig = go.Figure(
    data=[
        go.Choropleth(
            locations=["JPN"],
            locationmode="ISO-3",
            z=[1],
            colorscale=[[0, C["land"]], [1, C["land"]]],
            showscale=False,
            hoverinfo="skip",
            marker_line_color=C["line"],
            marker_line_width=1.0,
        ),
        *region_traces,
        make_quake_trace(df_chrono),
        make_ring_trace(df_chrono.iloc[0:0]),
    ],
    frames=frames,
)

# "All of Japan" starting bounds — always used as the maximum zoom-out & pan limit
base_lon_range = [122.5 - sea, 146.0 + sea]
base_lat_range = [24.0 - sea, 46.0 + sea]

if region_sel in REGION_BOUNDS:
    lon_range, lat_range = REGION_BOUNDS[region_sel]
else:
    lon_range, lat_range = base_lon_range, base_lat_range

zoom_state = f"{region_sel}-{sea}-{st.session_state.zoom_rev}"

fig.update_geos(
    domain=dict(x=[0.0, 1.0], y=[0.22, 1.0]),
    lonaxis_range=lon_range,
    lataxis_range=lat_range,
    projection_type="mercator",
    resolution=50,
    showland=False,
    showcountries=False,
    showcoastlines=False,
    showlakes=True,
    lakecolor=C["ocean"],
    showocean=True,
    oceancolor=C["ocean"],
    bgcolor=C["ocean"],
    showframe=True,
    framecolor=C["border"],
    framewidth=1,
    uirevision=zoom_state,
)

fig.update_layout(
    height=700,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor=C["bg"],
    dragmode="zoom",
    clickmode="event+select",
    shapes=[
        dict(
            type="rect", xref="paper", yref="paper",
            x0=0.0, y0=0.0, x1=1.0, y1=0.202,
            fillcolor=C["card"], line=dict(color=C["border"], width=1), layer="below",
        )
    ],
    updatemenus=[
        dict(
            type="buttons", direction="left",
            x=0.015, y=0.140, xanchor="left", yanchor="bottom",
            bgcolor=C["btn"], bordercolor=C["border"], borderwidth=1,
            font=dict(color=C["tx"], size=12), showactive=False,
            buttons=[
                dict(label="▶ Play (Oldest → Newest)", method="animate", args=[None, dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), fromcurrent=False, mode="immediate")]),
                dict(label="▶ Resume", method="animate", args=[None, dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")]),
                dict(label="⏭ Show All", method="animate", args=[[frames[-1].name], dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")]),
            ],
        )
    ],
    sliders=[
        dict(
            active=len(steps) - 1,
            x=0.015, y=0.015, len=0.97, xanchor="left", yanchor="bottom",
            bgcolor=C["btn"], activebgcolor=C["tx"], bordercolor=C["border"], borderwidth=1,
            tickcolor=C["muted"], font=dict(color=C["muted"], size=10),
            currentvalue=dict(prefix="🕒 Time of Day: ", visible=True, xanchor="left", font=dict(color=C["tx"], size=13)),
            pad=dict(t=6, b=4),
            steps=steps,
        )
    ],
)

st.plotly_chart(
    fig,
    key="japan_map",
    on_select="rerun",
    use_container_width=True,
    config={
        "scrollZoom": True,
        "doubleClick": False,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
    },
)

# Enforce the exact "All of Japan" starting zoom level & bounds as the maximum zoom-out & pan limit
components.html(
    f"""
    <script>
    (function() {{
      const baseLon = [{base_lon_range[0]}, {base_lon_range[1]}];
      const baseLat = [{base_lat_range[0]}, {base_lat_range[1]}];
      const isBaseView = {"true" if region_sel == "All of Japan" else "false"};
      const seaKey = "{sea:.1f}";

      const mercY = (deg) => Math.log(Math.tan(Math.PI / 4 + (deg * Math.PI / 180) / 2));
      const invMercY = (y) => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * (180 / Math.PI);

      function computeAllOfJapanLimit(gd) {{
        const geoSub = gd._fullLayout.geo._subplot;
        const w = (geoSub && geoSub.width) ? geoSub.width : gd.clientWidth;
        const h = (geoSub && geoSub.height) ? geoSub.height : (gd.clientHeight * 0.78);
        const y0 = mercY(baseLat[0]);
        const y1 = mercY(baseLat[1]);
        const sx = 360 / (baseLon[1] - baseLon[0]);
        const sy = (2 * Math.PI * (h / w)) / (y1 - y0);
        return {{
          minScale: Math.min(sx, sy),
          y0: y0,
          y1: y1,
          aspect: h / w
        }};
      }}

      function attachZoomLimit() {{
        const doc = window.parent.document;
        const plots = doc.querySelectorAll('.js-plotly-plot');
        const gd = plots[plots.length - 1];
        if (!gd || !gd._fullLayout || !gd._fullLayout.geo || !window.parent.Plotly) {{
          setTimeout(attachZoomLimit, 150);
          return;
        }}

        const Plotly = window.parent.Plotly;
        const store = window.parent.__japanZoomLimits = window.parent.__japanZoomLimits || {{}};

        if (isBaseView) {{
          const computed = computeAllOfJapanLimit(gd);
          store[seaKey] = {{
            minScale: gd._fullLayout.geo.projection.scale,
            y0: computed.y0,
            y1: computed.y1,
            aspect: computed.aspect
          }};
        }} else if (!store[seaKey]) {{
          store[seaKey] = computeAllOfJapanLimit(gd);
        }}

        gd.__japanGuardCfg = {{
          baseLon,
          baseLat,
          limit: store[seaKey]
        }};

        if (gd.__japanGuardBound) return;
        gd.__japanGuardBound = true;
        let clamping = false;

        gd.on('plotly_relayout', function() {{
          if (clamping) return;
          const cfg = gd.__japanGuardCfg;
          const geo = gd._fullLayout && gd._fullLayout.geo;
          if (!cfg || !geo) return;

          const geoSub = geo._subplot;
          const w = (geoSub && geoSub.width) ? geoSub.width : gd.clientWidth;
          const h = (geoSub && geoSub.height) ? geoSub.height : (gd.clientHeight * 0.78);
          const aspect = h / w;

          const y0 = cfg.limit.y0;
          const y1 = cfg.limit.y1;
          const sx = 360 / (cfg.baseLon[1] - cfg.baseLon[0]);
          const sy = (2 * Math.PI * aspect) / (y1 - y0);
          const minScale = Math.min(sx, sy);

          const rawScale = geo.projection.scale;
          const targetScale = Math.max(minScale, rawScale);

          const visLonSpan = Math.min(cfg.baseLon[1] - cfg.baseLon[0], 360 / targetScale);
          const minCLon = cfg.baseLon[0] + visLonSpan / 2;
          const maxCLon = cfg.baseLon[1] - visLonSpan / 2;

          const visYSpan = Math.min(y1 - y0, (2 * Math.PI * aspect) / targetScale);
          const minCLat = invMercY(y0 + visYSpan / 2);
          const maxCLat = invMercY(y1 - visYSpan / 2);

          const curLon = geo.center.lon;
          const curLat = geo.center.lat;
          const clampedLon = Math.max(minCLon, Math.min(maxCLon, curLon));
          const clampedLat = Math.max(minCLat, Math.min(maxCLat, curLat));

          const scaleExceeded = rawScale < minScale - 1e-4;
          const panExceeded = Math.abs(clampedLon - curLon) > 1e-4 || Math.abs(clampedLat - curLat) > 1e-4;

          if (scaleExceeded || panExceeded) {{
            clamping = true;
            const update = {{
              'geo.center.lon': clampedLon,
              'geo.center.lat': clampedLat
            }};
            if (scaleExceeded) {{
              update['geo.projection.scale'] = minScale;
            }}
            Plotly.relayout(gd, update).then(function() {{ clamping = false; }});
          }}
        }});
      }}
      attachZoomLimit();
    }})();
    </script>
    """,
    height=0,
)

# ---------------- Comprehensive Map & Color Legend ----------------
KEY_DEFS = {
    "Depth": [
        ("#ef4444", "Red — Shallow (< 30 km)", "Crustal earthquakes near the surface; typically cause the strongest local shaking."),
        ("#f59e0b", "Amber — Medium (30–70 km)", "Plate-interface and upper-mantle quakes common along Japan's subduction zones."),
        ("#22d3ee", "Cyan — Deep (70–300 km)", "Intermediate-depth earthquakes inside the descending Pacific or Philippine Sea slabs."),
        ("#6366f1", "Indigo — Very Deep (300+ km)", "Deep-focus slab earthquakes; felt broadly but rarely cause surface damage."),
    ],
    "Magnitude": [
        ("#dc2626", "Dark Red — M 7.0+ (Major)", "Major earthquake capable of widespread damage and tsunami generation."),
        ("#f97316", "Orange — M 6.0–6.9 (Strong)", "Strong earthquake; labeled directly on the map when labels are enabled."),
        ("#facc15", "Yellow — M 5.0–5.9 (Moderate)", "Moderate earthquake widely felt across nearby prefectures."),
        ("#38bdf8", "Sky Blue — Under M 5.0 (Light)", "Light or minor seismic activity recorded by monitoring networks."),
    ],
    "Age": [
        ("#f43f5e", "Rose Red — Last 24 hours", "Most recent seismic activity within the past day."),
        ("#fbbf24", "Gold — 1 to 7 days ago", "Earthquakes occurring earlier this week."),
        ("#22d3ee", "Cyan — 7 to 30 days ago", "Earthquakes within the past month."),
        ("#64748b", "Slate Gray — Over 30 days / Archive", "Older events in the selected window or historical research archive."),
    ],
    "Source": [
        ("#38bdf8", "Sky Blue — USGS", "United States Geological Survey global seismic catalog."),
        ("#34d399", "Emerald — JMA", "Japan Meteorological Agency domestic seismic bulletin."),
        ("#fb923c", "Orange — EMSC", "European-Mediterranean Seismological Centre feed."),
        ("#c084fc", "Purple — Research Archive", "Curated historical & peer-reviewed Japan events (1897–2026)."),
    ],
}


def render_key_cards(items):
    cards = []
    for hex_c, title, desc in items:
        cards.append(
            f'<div class="key-item">'
            f'<span class="swatch" style="background:{hex_c}"></span>'
            f'<div><div class="key-item-title">{title}</div><div class="key-item-desc">{desc}</div></div>'
            f'</div>'
        )
    return "".join(cards)


active_cards_html = render_key_cards(KEY_DEFS[color_by])
region_html = "".join(f'<span><span class="rswatch" style="background:{c}"></span>{r}</span>' for r, c in REGION_COLORS.items())

st.markdown(
    f"""<div class="key-box">
    <div class="key-title">Map Color Key — Currently Coloring Circles By: {color_by}</div>
    <div class="key-grid">{active_cards_html}</div>
    <div class="key-title">Symbols &amp; Markers Guide</div>
    <div class="key-grid" style="margin-bottom:12px;">
        <div class="key-item">
            <span class="swatch" style="background:#94a3b8;width:16px;height:16px;margin-top:1px;"></span>
            <div><div class="key-item-title">Circle Size = Magnitude</div><div class="key-item-desc">Larger circles represent higher-magnitude earthquakes; M6.0+ events display text labels.</div></div>
        </div>
        <div class="key-item">
            <span class="ring-swatch"></span>
            <div><div class="key-item-title">Outer Highlight Ring = Active Step</div><div class="key-item-desc">Appears during Timeline Player playback to spotlight the newest event(s) in that frame.</div></div>
        </div>
    </div>
    <div class="key-title">Region Border Colors (Click Any Region Border on the Map to Zoom In)</div>
    <div class="key-row" style="margin-bottom:0">{region_html}</div>
    </div>""",
    unsafe_allow_html=True,
)

with st.expander("Reference: What all 4 circle color modes mean (Depth, Magnitude, Age, Source)", expanded=False):
    for mode_name, mode_items in KEY_DEFS.items():
        st.markdown(f"**{mode_name} Color Key**")
        st.markdown(f'<div class="key-grid">{render_key_cards(mode_items)}</div>', unsafe_allow_html=True)

# ---------------- Activity Chart + Event Details Inspector ----------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Earthquake Activity (UTC)")
    freq = "h" if PERIODS[period] <= 1 else ("D" if PERIODS[period] <= 90 else "W")
    bucket_label = "Hourly" if freq == "h" else ("Daily" if freq == "D" else "Weekly")
    counts = df.set_index("time").resample(freq)["mag"].agg(count="count", max_mag="max").reset_index()
    counts["max_mag"] = counts["max_mag"].fillna(0)

    act_fig = go.Figure(
        go.Bar(
            x=counts["time"],
            y=counts["count"],
            marker_color="#38bdf8",
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>Events: %{y}<extra></extra>",
        )
    )
    act_fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor=C["card"],
        plot_bgcolor=C["card"],
        font=dict(color=C["tx"], size=11),
        title=dict(text=f"{bucket_label} event counts", font=dict(size=12, color=C["muted"])),
        xaxis=dict(gridcolor=C["border"], showgrid=False),
        yaxis=dict(gridcolor=C["border"], title="Events"),
    )
    st.plotly_chart(act_fig, use_container_width=True, config={"displayModeBar": False})

with right_col:
    st.subheader("Event Details")
    default_idx = 0
    if pts and pts[0].get("curve_number") == quake_trace_idx:
        clicked_id = pts[0].get("customdata")
        matches = df.index[df["id"] == clicked_id].tolist()
        if matches:
            default_idx = int(matches[0])

    options = [
        f"M{r.mag:.1f} · {r.place} ({r.time:%Y-%m-%d %H:%M UTC})"
        for r in df.itertuples()
    ]
    chosen_idx = st.selectbox(
        "Inspect event",
        range(len(df)),
        index=default_idx,
        format_func=lambda i: options[i],
        label_visibility="collapsed",
    )
    ev = df.iloc[chosen_idx]
    jst_ev = ev["time"] + timedelta(hours=9)
    st.markdown(
        f"""<div class="detail-card">
        <div style="font-weight:700;font-size:1.05rem;margin-bottom:8px;">M {ev['mag']:.1f} · {ev['place']}</div>
        <div><b>Time (UTC):</b> {ev['time']:%Y-%m-%d %H:%M:%S UTC}</div>
        <div><b>Time (JST):</b> {jst_ev:%Y-%m-%d %H:%M:%S JST}</div>
        <div><b>Depth:</b> {ev['depth']:.1f} km</div>
        <div><b>Region:</b> {ev['region']}</div>
        <div><b>Source:</b> {ev['source']}</div>
        <div><b>Coordinates:</b> {ev['lat']:.3f}, {ev['lon']:.3f}</div>
        <div><b>Event ID:</b> <code>{ev['id']}</code></div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------- Events Table & CSV Download ----------------
st.subheader(f"Earthquake Events ({len(df):,} matching)")
table_df = df[["time", "mag", "depth", "region", "place", "source", "lat", "lon"]].copy()
table_df["time"] = table_df["time"].dt.strftime("%Y-%m-%d %H:%M UTC")
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download shown quakes (CSV)",
    df.to_csv(index=False),
    "japan_earthquakes_shown.csv",
    "text/csv",
)