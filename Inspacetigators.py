import os, re, subprocess, sys


if os.environ.get("QUAKE_APP_CHILD") != "1":
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


from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Japan Earthquake Explorer", layout="wide")


HOKKAIDO = [(140, 41.5), (140.8, 41.8), (140.9, 42.5), (141.4, 42.6), (143, 42),
            (144.4, 42.9), (145.6, 43.3), (145.3, 44), (144, 44.1), (142.9, 44.7),
            (141.6, 45.5), (141.7, 44.8), (141.4, 43.4), (140.4, 43.3), (140.3, 42.4)]
HONSHU = [(141.4, 41.2), (141.5, 40.5), (142, 39.6), (141.9, 38.9), (141, 38.3),
          (141, 37), (140.6, 36.2), (140.8, 35.7), (140.4, 35.2), (139.9, 35),
          (140.1, 35.5), (139.7, 35.6), (139.6, 35.3), (139.7, 35.15), (139.15, 35.15),
          (138.9, 34.6), (138.8, 35), (137.4, 34.65), (136.9, 34.7), (136.9, 34.3),
          (136.3, 33.8), (135.8, 33.45), (135.4, 33.6), (135.1, 34.3), (135.4, 34.6),
          (134.7, 34.65), (133.9, 34.4), (132.4, 34.2), (131.4, 34), (130.9, 33.95),
          (131, 34.4), (131.8, 34.7), (132.8, 35.4), (133.8, 35.55), (134.9, 35.65),
          (135.9, 35.55), (136.1, 36), (136.7, 36.9), (137, 37.4), (137.3, 37.5),
          (137.1, 37), (137.6, 36.9), (138.5, 37.4), (139, 37.85), (139.6, 38.5),
          (140, 39.2), (139.9, 39.9), (140, 40.5), (140, 40.85), (140.9, 41.2)]
KYUSHU = [(130.9, 33.9), (131.7, 33.6), (131.9, 32.8), (131.4, 31.5), (130.7, 31),
          (130.2, 31.3), (130.2, 32.2), (130.6, 32.6), (129.9, 32.7), (129.8, 33.3),
          (130.3, 33.6)]
SHIKOKU = [(132.4, 33.4), (132.5, 34), (133.5, 34.3), (134.6, 34.2), (134.7, 33.8),
           (134.2, 33.3), (133.5, 33.4), (133, 32.8), (132.5, 32.9)]
OKINAWA = [(127.65, 26.1), (127.9, 26.2), (128.3, 26.85), (128.2, 26.9), (127.9, 26.6)]
CUTS = {
    "Tohoku": [(139.55, 38.6), (139.45, 37.8), (139.4, 37.1), (141.3, 36.85),
               (142.5, 36.85), (142.5, 42), (139, 42), (139, 38.6)],
    "Kanto": [(139.4, 37.1), (141.3, 36.85), (142.2, 36.85), (142.2, 33.8), (139, 33.8),
              (139, 35), (138.9, 35.5), (138.6, 36), (138.7, 36.5), (139, 36.9)],
    "Chubu": [(139.55, 38.65), (139.45, 37.8), (139.4, 37.1), (139, 36.9), (138.7, 36.5),
              (138.6, 36), (138.9, 35.5), (139, 35), (139, 34.3), (137, 34.3), (137, 34.7),
              (136.6, 35.3), (136.2, 35.9), (135.5, 35.6), (135.5, 38.7)],
    "Kansai": [(137, 33), (137, 34.7), (136.6, 35.3), (136.2, 35.9), (135.5, 35.7),
               (134.6, 35.7), (134.6, 33)],
    "Chugoku": [(134.6, 36.5), (134.6, 33), (130.5, 33), (130.5, 36.5)],
}
ISLANDS = {"Hokkaido": HOKKAIDO, "Shikoku": SHIKOKU, "Kyushu": KYUSHU, "Okinawa": OKINAWA}
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
REGIONS = ["All of Japan", "Hokkaido", "Tohoku", "Kanto", "Chubu", "Kansai",
           "Chugoku", "Shikoku", "Kyushu", "Okinawa", "Ogasawara", OFF]

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

#Data
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


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def dist_to_poly(x, y, poly):
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


def region_of(lat, lon):
    if lat < 28 and lon > 139:
        return "Ogasawara"
    candidates = dict(ISLANDS)
    for r, cut_poly in CUTS.items():
        if point_in_poly(lon, lat, cut_poly):
            candidates[r] = HONSHU
    best, bd = OFF, 1.5
    for r, poly in candidates.items():
        d = dist_to_poly(lon, lat, poly)
        if d < bd:
            best, bd = r, d
    if best == OFF:
        for r, cut_poly in CUTS.items():
            d = max(dist_to_poly(lon, lat, HONSHU), dist_to_poly(lon, lat, cut_poly))
            if d < bd:
                best, bd = r, d
    return best


def fetch_usgs(start_str, min_mag):
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


def fetch_jma(cutoff_dt, min_mag):
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


def fetch_emsc(start_str, min_mag):
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


def deduplicate_quakes(df, time_tol_sec=90, dist_tol_deg=0.45):
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
def load(days, min_mag, include_archive):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    start_str = cutoff.strftime("%Y-%m-%d")
    rows = []

    for fetcher in (
        lambda: fetch_usgs(start_str, min_mag),
        lambda: fetch_jma(cutoff, min_mag),
        lambda: fetch_emsc(start_str, min_mag),
    ):
        try:
            rows.extend(fetcher())
        except Exception:
            pass

    if include_archive:
        for t_str, mag, depth, lat, lon, place, eid in RESEARCH_ARCHIVE:
            if mag >= min_mag:
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

    cols = ["time", "mag", "depth", "lat", "lon", "place", "region", "source", "id"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return deduplicate_quakes(pd.DataFrame(rows, columns=cols))


if "zoom_rev" not in st.session_state:
    st.session_state.zoom_rev = 0

# ---------------- sidebar: settings dropdowns ----------------
st.sidebar.title("Settings")
with st.sidebar.expander("Data", expanded=True):
    period = st.selectbox("Time period", list(PERIODS), index=2)
    min_mag = st.selectbox("Minimum magnitude", [2.5, 3.5, 4.0, 4.5, 5.0, 6.0], index=2)
    include_archive = st.toggle("Include research archive quakes (1897–2026)", value=True)
    auto = st.toggle("Refresh every 5 minutes", value=True)
    if st.button("Reload data", use_container_width=True):
        st.cache_data.clear()
with st.sidebar.expander("Filters", expanded=True):
    region_sel = st.selectbox("Region", REGIONS)
    depth_sel = st.selectbox("Depth", list(DEPTHS))
    source_sel = st.selectbox("Source", ["All sources", "USGS", "JMA", "EMSC", "Archive"])
    search = st.text_input("Search place")
with st.sidebar.expander("Timeline Player", expanded=True):
    play_style = st.selectbox("Playback style", ["Cumulative (build up)", "Single step (event by event)"])
    speed_label = st.selectbox("Playback speed", list(SPEEDS), index=1)
    frame_ms = SPEEDS[speed_label]
with st.sidebar.expander("Display", expanded=True):
    color_by = st.selectbox("Colour circles by", ["Depth", "Magnitude", "Age", "Source"])
    size = st.slider("Circle size", 2, 8, 4)
    labels = st.toggle("Label magnitude 6 and above", value=True)
    light = st.toggle("Light interface")
with st.sidebar.expander("Map", expanded=True):
    sea = st.slider("Sea shown around Japan (degrees)", 0.5, 5.0, 1.5, 0.5)
    if st.button("Reset zoom to Japan", use_container_width=True):
        st.session_state.zoom_rev += 1
if auto:
    st_autorefresh(interval=300000, key="auto")

#The themes light & dark
C = (
    dict(
        bg="#f4f6fa",
        side="#ffffff",
        card="#ffffff",
        btn="#f1f5f9",
        hover="#e2e8f0",
        border="#cbd5e1",
        tx="#0f172a",
        muted="#475569",
        land="#efe9db",
        line="#64748b",
        ocean="#d8e8f6",
        marker_line="#1e293b",
        picked="#0f172a",
    )
    if light
    else dict(
        bg="#070b14",
        side="#0d1424",
        card="#111a2e",
        btn="#0d1424",
        hover="#1e293b",
        border="#23304a",
        tx="#f1f5f9",
        muted="#94a3b8",
        land="#1e293b",
        line="#475569",
        ocean="#09111e",
        marker_line="#09111e",
        picked="#ffffff",
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

[data-testid="stExpander"] {{
    background: {C["card"]};
    border: 1px solid {C["border"]} !important;
    border-radius: 10px;
}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary span, [data-testid="stExpander"] summary svg {{
    color: {C["tx"]} !important;
    fill: {C["tx"]} !important;
}}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input {{
    background: {C["card"]} !important;
    color: {C["tx"]} !important;
    border-color: {C["border"]} !important;
}}
div[data-baseweb="select"] svg {{ fill: {C["tx"]} !important; }}
div[data-baseweb="popover"], ul[role="listbox"] {{
    background: {C["card"]} !important;
    border: 1px solid {C["border"]} !important;
}}
li[role="option"] {{
    background: {C["card"]} !important;
    color: {C["tx"]} !important;
}}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
    background: {C["hover"]} !important;
    color: {C["tx"]} !important;
}}

.stButton > button, .stDownloadButton > button {{
    background: {C["card"]} !important;
    color: {C["tx"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: {C["hover"]} !important;
    border-color: {C["tx"]} !important;
}}

[data-testid="stMetric"] {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 12px 16px;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]};
    border-radius: 10px;
    overflow: hidden;
}}
</style>"""
st.markdown(css, unsafe_allow_html=True)

#Filters & data
st.title("Japan Earthquake Explorer")
try:
    df = load(PERIODS[period], min_mag, include_archive)
except Exception as err:
    st.error(f"Could not load earthquake data: {err}")
    st.stop()

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

#Table oldest quake to newest
df = df.sort_values("time", ascending=False).reset_index(drop=True)
df_chrono = df.sort_values("time", ascending=True).reset_index(drop=True)

table = df[["time", "mag", "depth", "region", "source", "place"]].head(300).copy()
table["time"] = table.time.dt.strftime("%Y-%m-%d %H:%M")
sel = st.session_state.get("tbl", {}).get("selection", {}).get("rows", [])
picked = df.iloc[sel[0]] if sel and sel[0] < len(df) else None

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
        textposition="middle right",
        textfont=dict(color=C["tx"], size=11),
        hovertext=hover,
        hoverinfo="text",
        mode="markers+text" if labels else "markers",
        showlegend=False,
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=0.7, color=C["marker_line"]),
        ),
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
        marker=dict(
            size=ring_sizes,
            color="rgba(0,0,0,0)",
            line=dict(width=2.5, color=C["picked"]),
        ),
    )


#video from oldest to newest
n_total = len(df_chrono)
max_frames = 120
if n_total <= max_frames:
    end_indices = list(range(1, n_total + 1))
else:
    end_indices = sorted(set(np.linspace(1, n_total, max_frames, dtype=int).tolist()))

frames = []
steps = []
prev_end = 0
cumulative = play_style.startswith("Cumulative")

for idx, end_i in enumerate(end_indices):
    sub = df_chrono.iloc[:end_i] if cumulative else df_chrono.iloc[prev_end:end_i]
    newest_in_step = df_chrono.iloc[prev_end:end_i]
    prev_end = end_i

    cur_row = newest_in_step.iloc[-1]
    t_label = cur_row["time"].strftime("%Y-%m-%d %H:%M UTC")
    if len(newest_in_step) == 1:
        step_caption = f"{t_label}  |  M{cur_row['mag']:.1f} {cur_row['region']} ({end_i}/{n_total})"
    else:
        step_caption = f"{t_label}  |  +{len(newest_in_step)} quakes ({end_i}/{n_total})"

    fname = f"frame_{idx}"
    frames.append(
        go.Frame(
            name=fname,
            traces=[1, 2],
            data=[make_quake_trace(sub), make_ring_trace(newest_in_step)],
        )
    )
    steps.append(
        dict(
            method="animate",
            label=step_caption,
            args=[
                [fname],
                dict(
                    mode="immediate",
                    frame=dict(duration=frame_ms, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
        )
    )

#To make map on top and player at the bottom
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
            marker_line_width=1.2,
        ),
        make_quake_trace(df_chrono),
        make_ring_trace(df_chrono[df_chrono["id"] == picked.id] if picked is not None else df_chrono.iloc[0:0]),
    ],
    frames=frames,
)

if picked is not None:
    lon_range = [picked.lon - 1.4, picked.lon + 1.4]
    lat_range = [picked.lat - 1.1, picked.lat + 1.1]
elif region_sel in REGION_BOUNDS:
    lon_range, lat_range = REGION_BOUNDS[region_sel]
else:
    lon_range = [122.5 - sea, 146.0 + sea]
    lat_range = [24.0 - sea, 46.0 + sea]

zoom_state = f"{region_sel}-{sea}-{picked.id if picked is not None else ''}-{st.session_state.zoom_rev}"

# Restrict the map domain to y=[0.22, 1.0] so the video player sits cleanly below the map box
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
    # Bordered card container directly below the map frame
    shapes=[
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.0,
            y0=0.0,
            x1=1.0,
            y1=0.202,
            fillcolor=C["card"],
            line=dict(color=C["border"], width=1),
            layer="below",
        )
    ],
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.015,
            y=0.140,
            xanchor="left",
            yanchor="bottom",
            bgcolor=C["btn"],
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(color=C["tx"], size=12),
            showactive=False,
            buttons=[
                dict(
                    label="▶ Play (Oldest → Newest)",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_ms, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=False,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="▶ Resume",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_ms, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="⏭ Show All",
                    method="animate",
                    args=[
                        [frames[-1].name],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                ),
            ],
        )
    ],
    sliders=[
        dict(
            active=len(steps) - 1,
            x=0.015,
            y=0.015,
            len=0.97,
            xanchor="left",
            yanchor="bottom",
            bgcolor=C["btn"],
            activebgcolor=C["tx"],
            bordercolor=C["border"],
            borderwidth=1,
            tickcolor=C["muted"],
            font=dict(color=C["muted"], size=10),
            currentvalue=dict(
                prefix="Timeline: ",
                visible=True,
                xanchor="left",
                font=dict(color=C["tx"], size=13),
            ),
            pad=dict(t=6, b=4),
            steps=steps,
        )
    ],
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
    },
)

#bottom info
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Earthquakes", len(df))
c2.metric("Strongest", f"M {df.mag.max():.1f}")
c3.metric("Average depth", f"{df.depth.mean():.0f} km")
c4.metric("Magnitude 5+", int((df.mag >= 5).sum()))
st.caption("Click a row to zoom the map to that earthquake, or use scroll / drag-box on the map to zoom in.")
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    key="tbl",
    on_select="rerun",
    selection_mode="single-row",
)
st.download_button(
    "Download shown quakes (CSV)",
    df.to_csv(index=False),
    "japan_earthquakes_shown.csv",
    "text/csv",
)