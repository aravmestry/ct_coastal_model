import os
import re
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="CT Coastal Flood Risk Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main > div {padding-top: 1rem;}
.block-note {
    background: rgba(250, 204, 21, 0.18);
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Connecticut Coastal Flood Risk Model")
st.write("Fast statewide presentation version covering Connecticut towns.")

st.markdown(
    """
    <div class="block-note">
    This is a fast approximate statewide model. It uses Connecticut town locations and an estimated coastal exposure score for 1 ft, 3 ft, and 6 ft scenarios. It is designed for fast presentation use, not full polygon-based flood intersection.
    </div>
    """,
    unsafe_allow_html=True
)

TOWN_PATH_OPTIONS = [
    "data/raw/ct_towns.geojson",
    "data/towns/ct_towns.geojson",
    "ct_towns.geojson",
]

PROJECTED_CRS = "EPSG:32618"
GEOGRAPHIC_CRS = "EPSG:4326"

UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

def classify_risk(score):
    if score < 8:
        return "Low"
    elif score < 20:
        return "Medium"
    return "High"

def is_bad_name(text):
    if text is None:
        return True
    s = str(text).strip()
    if s == "":
        return True
    if s.isdigit():
        return True
    if UUID_PATTERN.match(s):
        return True
    if len(s) > 24 and "-" in s:
        return True
    return False

def clean_town_name(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    if is_bad_name(s):
        return None

    return s.title()

def score_candidate_column(series):
    sample = series.dropna().astype(str).head(80).tolist()
    if len(sample) == 0:
        return -1

    valid = 0
    bad = 0

    for x in sample:
        x = x.strip()
        if is_bad_name(x):
            bad += 1
        else:
            valid += 1

    return valid - bad

@st.cache_data
def load_ct_towns():
    town_path = None
    for path in TOWN_PATH_OPTIONS:
        if os.path.exists(path):
            town_path = path
            break

    if town_path is None:
        return None, None

    gdf = gpd.read_file(town_path)

    preferred_cols = [
        "town", "TOWN", "Town",
        "name", "NAME", "Name",
        "town_name", "TOWN_NAME",
        "municipality", "MUNICIPALITY",
        "city", "CITY",
        "namelsad", "NAMELSAD",
        "label", "LABEL",
    ]

    town_col = None

    for col in preferred_cols:
        if col in gdf.columns:
            if score_candidate_column(gdf[col]) > 0:
                town_col = col
                break

    if town_col is None:
        best_score = -999
        best_col = None
        for col in gdf.columns:
            if col == "geometry":
                continue
            col_score = score_candidate_column(gdf[col])
            if col_score > best_score:
                best_score = col_score
                best_col = col
        town_col = best_col

    if town_col is None:
        return None, list(gdf.columns)

    gdf = gdf[[town_col, "geometry"]].copy()
    gdf = gdf.rename(columns={town_col: "town_raw"})

    if gdf.crs is None:
        gdf = gdf.set_crs(GEOGRAPHIC_CRS)

    gdf_geo = gdf.to_crs(GEOGRAPHIC_CRS)
    gdf_proj = gdf.to_crs(PROJECTED_CRS)

    centroids_proj = gdf_proj.geometry.centroid
    centroids_geo = gpd.GeoSeries(centroids_proj, crs=PROJECTED_CRS).to_crs(GEOGRAPHIC_CRS)

    gdf_geo["longitude"] = centroids_geo.x
    gdf_geo["latitude"] = centroids_geo.y
    gdf_geo["town"] = gdf_geo["town_raw"].apply(clean_town_name)

    df = gdf_geo[["town", "latitude", "longitude"]].dropna().copy()
    df = df.drop_duplicates(subset=["town"]).reset_index(drop=True)

    return df, list(gdf_geo.columns)

def build_statewide_scenario(df_base, scenario_ft):
    df = df_base.copy()

    southness = (42.05 - df["latitude"]) / (42.05 - 40.95)
    southness = southness.clip(0, 1)

    coastal_exposure = southness ** 2.2

    westness = (-71.8 - df["longitude"]) / (-71.8 + 73.7)
    westness = westness.clip(0, 1)

    scenario_scale = {
        1: 18,
        3: 34,
        6: 56
    }[scenario_ft]

    df["risk_score"] = (
        coastal_exposure * scenario_scale
        + westness * 4
        + (1 - ((df["latitude"] - 41.5).abs() / 0.7).clip(0, 1)) * 2
    )

    coastal_towns = [
        "Stamford", "Greenwich", "Norwalk", "Bridgeport", "Stratford",
        "Milford", "West Haven", "New Haven", "East Haven", "Branford",
        "Guilford", "Madison", "Clinton", "Westbrook", "Old Saybrook",
        "Old Lyme", "New London", "Groton", "Fairfield", "Westport"
    ]

    df.loc[df["town"].isin(coastal_towns), "risk_score"] += scenario_ft * 1.5

    df["risk_score"] = df["risk_score"].round(2)
    df["flood_percent"] = (df["risk_score"] * 1.15).round(2)
    df["risk_class"] = df["risk_score"].apply(classify_risk)

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)

def risk_to_color(risk_class):
    if risk_class == "Low":
        return [34, 197, 94]
    if risk_class == "Medium":
        return [245, 158, 11]
    if risk_class == "High":
        return [239, 68, 68]
    return [148, 163, 184]

df_towns, detected_columns = load_ct_towns()

if df_towns is None or len(df_towns) == 0:
    st.error("Could not detect valid Connecticut town names from the town boundary file.")
    if detected_columns is not None:
        st.write("Detected columns:", detected_columns)
    st.stop()

top1, top2, top3 = st.columns([1.2, 1, 1])

with top1:
    scenario = st.selectbox(
        "Sea level rise scenario",
        [1, 3, 6],
        index=0,
        format_func=lambda x: f"{x} ft"
    )

df = build_statewide_scenario(df_towns, scenario)

with top2:
    st.metric("Towns evaluated", len(df))

with top3:
    st.metric("Highest risk score", f"{df['risk_score'].max():.2f}")

st.markdown("## Top Ranked Towns")

st.dataframe(
    df[["town", "risk_score", "flood_percent", "risk_class"]].rename(columns={
        "town": "Town",
        "risk_score": "Risk Score",
        "flood_percent": "Flood Percent",
        "risk_class": "Risk Class",
    }),
    width="stretch",
    hide_index=True
)

st.markdown("## Compare Two Towns")

towns = sorted(df["town"].dropna().unique().tolist())

default_a = "Stamford" if "Stamford" in towns else towns[0]
default_b = "New Haven" if "New Haven" in towns else towns[min(1, len(towns) - 1)]

c1, c2 = st.columns(2)

with c1:
    town_a = st.selectbox(
        "First town",
        towns,
        index=towns.index(default_a)
    )

with c2:
    town_b = st.selectbox(
        "Second town",
        towns,
        index=towns.index(default_b)
    )

row_a = df[df["town"] == town_a].iloc[0]
row_b = df[df["town"] == town_b].iloc[0]

comp1, comp2 = st.columns(2)

with comp1:
    st.subheader(town_a)
    st.metric("Risk Score", f"{row_a['risk_score']:.2f}")
    st.metric("Flood Percent", f"{row_a['flood_percent']:.2f}%")
    st.metric("Risk Class", row_a["risk_class"])

with comp2:
    st.subheader(town_b)
    st.metric("Risk Score", f"{row_b['risk_score']:.2f}")
    st.metric("Flood Percent", f"{row_b['flood_percent']:.2f}%")
    st.metric("Risk Class", row_b["risk_class"])

st.markdown("## Map of Town Risk Points")

default_map_towns = df.head(10)["town"].tolist()

selected_map_towns = st.multiselect(
    "Choose towns to display on the map",
    options=towns,
    default=default_map_towns
)

# Legend
legend_col1, legend_col2, legend_col3 = st.columns(3)
with legend_col1:
    st.markdown("🟢 **Low Risk**")
with legend_col2:
    st.markdown("🟠 **Medium Risk**")
with legend_col3:
    st.markdown("🔴 **High Risk**")

map_df = df[df["town"].isin(selected_map_towns)].copy()

if len(map_df) == 0:
    st.info("Select one or more towns to display them on the map.")
else:
    map_df["color"] = map_df["risk_class"].apply(risk_to_color)
    map_df["radius"] = (map_df["risk_score"] * 250).clip(lower=800, upper=4000)

    center_lat = map_df["latitude"].mean()
    center_lon = map_df["longitude"].mean()

    view_state = pdk.ViewState(
        latitude=float(center_lat),
        longitude=float(center_lon),
        zoom=8 if len(map_df) <= 10 else 7,
        pitch=0
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[longitude, latitude]',
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<b>Town:</b> {town}<br/><b>Risk Score:</b> {risk_score}<br/><b>Flood Percent:</b> {flood_percent}%<br/><b>Risk Class:</b> {risk_class}",
        "style": {"backgroundColor": "white", "color": "black"}
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        )
    )

st.markdown("## Quick Summary")

highest = df.iloc[0]
lowest = df.iloc[-1]

st.write(
    f"Under the {scenario}-foot scenario, the highest-ranked town is {highest['town']} "
    f"with a risk score of {highest['risk_score']:.2f}, while the lowest-ranked town is "
    f"{lowest['town']} with a risk score of {lowest['risk_score']:.2f}."
)
