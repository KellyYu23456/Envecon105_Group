# dashboard.py
# Streamlit Dashboard for Envecon105 – Group + Individual Results

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Envecon105 – CO₂, Energy, GDP & Temperature",
    page_icon="🌍",
    layout="wide",
)

# -------------------------------
# 0) File URLs (public GitHub)
# -------------------------------
RAW_URLS = {
    "gdp_zip": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/gdp_global.zip",
    "energy_zip": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/energy_global.zip",
    "co2_per_capita_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/co2_per_capita.xlsx",
    "disasters_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/natural_disasters_china.xlsx",
    "temp_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/temp_mean_china_cru_1901-2024.xlsx",
    "co2_wide_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/yearly_co2_emissions_1000_tonnes.xlsx",
}

# -------------------------------
# 1) Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def read_excel_from_url(url, **kwargs) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), **kwargs)

@st.cache_data(show_spinner=False)
def read_zip_worldbank_csv(url, indicator_guess=None, skiprows=3) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    members = z.namelist()
    candidates = [m for m in members if m.lower().endswith(".csv") and m.startswith("API_")]
    name = candidates[0] if candidates else members[0]
    with z.open(name) as f:
        df = pd.read_csv(f, skiprows=skiprows)

    year_cols = [c for c in df.columns if str(c).isdigit()]
    long_df = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        value_vars=year_cols, var_name="Year", value_name="Value"
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).copy()

    if indicator_guess:
        long_df = long_df[long_df["Indicator Code"].eq(indicator_guess)].copy()

    return long_df.rename(columns={"Country Name": "Country", "Indicator Name": "Indicator"})

def normalize_year_numeric_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df, None, []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.index.name and "year" in str(df.index.name).lower():
        df = df.reset_index()
    year_col = None
    for c in df.columns:
        if "year" in c.lower():
            year_col = c
            break
    for c in df.columns:
        if c != year_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if year_col and year_col in numeric_cols:
        numeric_cols.remove(year_col)
    return df, year_col, numeric_cols

# -------------------------------
# 2) Load & tidy
# -------------------------------
with st.spinner("Loading data from GitHub…"):
    # GDP (% growth) and Energy (kg oil eq per person)
    gdp_long = read_zip_worldbank_csv(RAW_URLS["gdp_zip"], indicator_guess="NY.GDP.PCAP.KD.ZG")
    gdp_long["Indicator"] = "GDP per capita (yearly growth, %)"
    energy_long = read_zip_worldbank_csv(RAW_URLS["energy_zip"], indicator_guess="EG.USE.PCAP.KG.OE")
    energy_long["Indicator"] = "Energy Use (kg oil eq. per person)"

    # CO2 per capita (wide -> long)
    co2_pc_wide = read_excel_from_url(RAW_URLS["co2_per_capita_xlsx"])
    co2_pc_wide = co2_pc_wide.rename(columns={co2_pc_wide.columns[0]: "Country"})
    co2_pc_long = co2_pc_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_pc_long["Year"] = pd.to_numeric(co2_pc_long["Year"], errors="coerce")
    co2_pc_long["Value"] = pd.to_numeric(co2_pc_long["Value"], errors="coerce")
    co2_pc_long = co2_pc_long.dropna(subset=["Year"]).copy()
    co2_pc_long["Indicator"] = "Per_Capita_Emissions"

    # CO2 totals (1,000 tonnes -> metric tons)
    co2_wide = read_excel_from_url(RAW_URLS["co2_wide_xlsx"])
    co2_wide = co2_wide.rename(columns={co2_wide.columns[0]: "Country"})
    co2_long = co2_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_long["Year"] = pd.to_numeric(co2_long["Year"], errors="coerce")
    co2_long["Value"] = pd.to_numeric(co2_long["Value"], errors="coerce") * 1000.0
    co2_long = co2_long.dropna(subset=["Year"]).copy()
    co2_long["Indicator"] = "CO2 Emissions (Metric Tons)"

        # --------- Temperature China (robust: try all sheets + several headers) ----------
    temperature_cn = pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

    # download once so we can inspect multiple sheets
    r_temp = requests.get(RAW_URLS["temp_xlsx"], timeout=60)
    r_temp.raise_for_status()
    bio = io.BytesIO(r_temp.content)

    # list sheets and try a few header offsets
    xls = pd.ExcelFile(bio)
    tried = False
    for sheet in xls.sheet_names:
        for sr in (4, 3, 5, 2, 1, 0, 6):
            tried = True
            try:
                raw = pd.read_excel(io.BytesIO(r_temp.content), sheet_name=sheet,
                                    skiprows=sr, na_values=["-99", -99])
            except Exception:
                continue

            norm, year_col, num_cols = normalize_year_numeric_table(raw)
            if year_col is None or not num_cols:
                continue

            # prefer an Annual/Mean column; otherwise average numeric columns (e.g., months)
            ann_cols = [c for c in num_cols if any(k in c.lower() for k in ["ann", "annual", "mean", "avg"])]
            tmp = norm[[year_col] + num_cols].rename(columns={year_col: "Year"}).copy()
            tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce")

            if ann_cols:
                tmp["Temperature_C"] = pd.to_numeric(tmp[ann_cols[0]], errors="coerce")
            else:
                tmp["Temperature_C"] = tmp[num_cols].mean(axis=1, skipna=True)

            tmp = tmp.dropna(subset=["Year", "Temperature_C"])
            if tmp.empty:
                continue

            out = tmp[["Year", "Temperature_C"]].copy()
            out["Country"] = "China"
            out["Indicator"] = "Temperature"
            out["Value"] = out["Temperature_C"] * 9.0 / 5.0 + 32.0  # °C → °F
            temperature_cn = out[["Country", "Year", "Indicator", "Value"]]
            break  # stop after first successful parse
        if not temperature_cn.empty:
            break

    # If still empty but we tried, leave it empty so Tab 3 shows the info banner
    # --------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

    # Disasters China
    dis_cn_raw = read_excel_from_url(RAW_URLS["disasters_xlsx"])
    dis_cn_raw.columns = [str(c).strip() for c in dis_cn_raw.columns]
    if "Year" not in dis_cn_raw.columns:
        maybe_year = [c for c in dis_cn_raw.columns if c.lower() == "year"]
        if maybe_year:
            dis_cn_raw = dis_cn_raw.rename(columns={maybe_year[0]: "Year"})
    disasters_cn = pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])
    if "Year" in dis_cn_raw.columns:
        numeric_cols = [c for c in dis_cn_raw.columns if c != "Year" and pd.api.types.is_numeric_dtype(dis_cn_raw[c])]
        if numeric_cols:
            tmp = dis_cn_raw[["Year"] + numeric_cols].copy()
            tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce")
            tmp["Value"] = tmp[numeric_cols].sum(axis=1, skipna=True)
            tmp = tmp.dropna(subset=["Year"])
            disasters_cn = tmp[["Year", "Value"]].copy()
            disasters_cn["Country"] = "China"
            disasters_cn["Indicator"] = "Disasters"
            disasters_cn = disasters_cn[["Country", "Year", "Indicator", "Value"]]

# unified long table
data_long = pd.concat(
    [
        co2_long[["Country", "Year", "Indicator", "Value"]],
        energy_long[["Country", "Year", "Indicator", "Value"]],
        gdp_long[["Country", "Year", "Indicator", "Value"]],
        co2_pc_long[["Country", "Year", "Indicator", "Value"]],
        temperature_cn,
        disasters_cn,
    ],
    ignore_index=True,
)
data_long["Region"] = np.where(data_long["Country"] == "China", "China", "Rest of the World")

# -------------------------------
# 3) Sidebar
# -------------------------------
countries = sorted(co2_long["Country"].dropna().unique().tolist())
default_country = "China" if "China" in countries else (countries[0] if countries else "China")

st.sidebar.header("Controls")
focus_country = st.sidebar.selectbox("Country focus", countries, index=countries.index(default_country))

# DRIVE DEFAULTS BY EMISSIONS (prevents picking 2024 that has no CO2)
emis_min, emis_max = int(co2_long["Year"].min()), int(co2_long["Year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=emis_min,
    max_value=emis_max,
    value=(max(1900, emis_min), emis_max),
    step=1,
)

show_smoothed = st.sidebar.checkbox("Show smoothed lines (rolling)", value=True)
window = st.sidebar.slider("Smoothing window (years)", 3, 11, 5, step=2) if show_smoothed else 1

df = data_long[(data_long["Year"] >= year_range[0]) & (data_long["Year"] <= year_range[1])].copy()

# -------------------------------
# 4) Layout (tabs)
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Global & Top Emitters", "China Deep-dive", "Per-Capita vs GDP"])

# -------------------------------------------------
# Tab 1: Overview
# -------------------------------------------------
with tab1:
    st.markdown("## Overview")

    latest_emis_year = int(co2_long["Year"].max())
    world_emis = co2_long[co2_long["Year"] == latest_emis_year]["Value"].sum()
    china_emis = co2_long[(co2_long["Country"] == "China") & (co2_long["Year"] == latest_emis_year)]["Value"].sum()
    china_share = (china_emis / world_emis * 100) if world_emis else np.nan

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Latest emissions year", latest_emis_year)
    colB.metric("Global CO₂ (Mt)", f"{world_emis/1e6:,.1f}")
    colC.metric("China CO₂ (Mt)", f"{china_emis/1e6:,.1f}")
    colD.metric("China share of global", f"{china_share:,.1f}%")

    st.markdown("### Global CO₂ Emissions over time (sum of all countries)")
    global_emissions = co2_long.groupby("Year", as_index=False)["Value"].sum()
    fig = px.line(global_emissions, x="Year", y="Value", labels={"Value": "CO₂ (metric tons)"},
                  title="World CO₂ Emissions per Year")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Tab 2: Global & Top Emitters
# -------------------------------------------------
with tab2:
    st.markdown("## Global & Top Emitters")
    st.markdown("### Country CO₂ trajectories (highlight focus country)")

    latest = co2_long[co2_long["Year"] == latest_emis_year].sort_values("Value", ascending=False)
    top_countries = latest["Country"].head(12).tolist()
    subset = co2_long[co2_long["Country"].isin(top_countries)]

    fig2 = px.line(
        subset, x="Year", y="Value", color="Country",
        line_group="Country",
        title=f"Top 12 CO₂ Emitting Countries (latest year = {latest_emis_year})",
        labels={"Value": "CO₂ (metric tons)"},
    )
    for i, d in enumerate(fig2.data):
        if d.name != focus_country:
            fig2.data[i].line.width = 1
            fig2.data[i].opacity = 0.35
        else:
            fig2.data[i].line.width = 3
            fig2.data[i].opacity = 1.0
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Heatmap: Top 10 CO₂ Emission-producing Countries (1900–latest)")
    top10 = (
        co2_long[co2_long["Year"] == latest_emis_year]
        .sort_values("Value", ascending=False)
        .head(10)["Country"].tolist()
    )
    tile_df = (
        co2_long[(co2_long["Country"].isin(top10)) & (co2_long["Year"] >= 1900)]
        .pivot_table(index="Country", columns="Year", values="Value", aggfunc="sum")
        .replace({0: np.nan})
    )
    fig3 = px.imshow(np.log(tile_df), aspect="auto", color_continuous_scale="viridis",
                     labels=dict(color="ln(CO₂)"),
                     title=f"Top 10 CO₂ Emissions – ordered by {latest_emis_year}")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# Tab 3: China Deep-dive
# -------------------------------------------------
with tab3:
    st.markdown("## China Deep-dive: Emissions, Temperature & Relationships")

    # Make sure Year is Int64 on both series (prevents float/int mismatch)
    c_em = co2_long[co2_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "Emissions"}).copy()
    c_em["Year"] = pd.to_numeric(c_em["Year"], errors="coerce").astype("Int64")
    c_temp = temperature_cn[["Year", "Value"]].rename(columns={"Value": "Temperature_F"}).copy()
    c_temp["Year"] = pd.to_numeric(c_temp["Year"], errors="coerce").astype("Int64")

    # Compute the overlap with the slider
    if not c_em.empty and not c_temp.empty:
        overlap_min = max(c_em["Year"].min(), c_temp["Year"].min(), pd.Series([year_range[0]], dtype="Int64").iloc[0])
        overlap_max = min(c_em["Year"].max(), c_temp["Year"].max(), pd.Series([year_range[1]], dtype="Int64").iloc[0])
    else:
        overlap_min, overlap_max = None, None

    if overlap_min is not None and overlap_max is not None and overlap_min <= overlap_max:
        in_overlap = (c_em["Year"] >= overlap_min) & (c_em["Year"] <= overlap_max)
        c_em_clip = c_em[in_overlap].copy()
        in_overlap_t = (c_temp["Year"] >= overlap_min) & (c_temp["Year"] <= overlap_max)
        c_temp_clip = c_temp[in_overlap_t].copy()
        merged = pd.merge(c_em_clip, c_temp_clip, on="Year", how="inner")
    else:
        merged = pd.DataFrame(columns=["Year", "Emissions", "Temperature_F"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CO₂ Emissions (China)")
        if not merged.empty:
            fig_e = px.line(merged, x="Year", y="Emissions")
            if show_smoothed and window > 1:
                ms = merged.sort_values("Year")
                ms["Smoothed"] = ms["Emissions"].rolling(window, center=True, min_periods=1).mean()
                fig_e.add_scatter(x=ms["Year"], y=ms["Smoothed"], mode="lines", name="Smoothed")
            st.plotly_chart(fig_e, use_container_width=True)
        else:
            st.info("No overlapping years between emissions and the selected range.")

    with col2:
        st.markdown("#### Temperature (°F, China)")
        if not merged.empty:
            fig_t = px.line(merged, x="Year", y="Temperature_F")
            if show_smoothed and window > 1:
                ms = merged.sort_values("Year")
                ms["Smoothed"] = ms["Temperature_F"].rolling(window, center=True, min_periods=1).mean()
                fig_t.add_scatter(x=ms["Year"], y=ms["Smoothed"], mode="lines", name="Smoothed")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No overlapping years between temperature and the selected range.")

    st.markdown("### Scaled Temperature vs CO₂ Emissions (China)")
    if len(merged) >= 3:
        scaler = StandardScaler()
        XY = scaler.fit_transform(merged[["Emissions", "Temperature_F"]])
        x, y = XY[:, 0], XY[:, 1]
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = b + m * xs
        fig_sc = px.scatter(x=x, y=y, labels={"x": "Scaled Emissions", "y": "Scaled Temperature"},
                            title=f"China CO₂ Emissions vs Temperature ({int(merged['Year'].min())}–{int(merged['Year'].max())})")
        fig_sc.add_scatter(x=xs, y=ys, mode="lines", name="Fit")
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Not enough overlapping years to plot the scaled scatter.")

    st.markdown("### CO₂ per Capita vs Total CO₂ (China)")
    c_total = co2_long[co2_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "Total"}).copy()
    c_total["Year"] = pd.to_numeric(c_total["Year"], errors="coerce").astype("Int64")
    c_pc = co2_pc_long[co2_pc_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "PerCapita"}).copy()
    c_pc["Year"] = pd.to_numeric(c_pc["Year"], errors="coerce").astype("Int64")
    # intersect with the same overlap range we computed
    if overlap_min is not None and overlap_max is not None and overlap_min <= overlap_max:
        c_total = c_total[(c_total["Year"] >= overlap_min) & (c_total["Year"] <= overlap_max)]
        c_pc = c_pc[(c_pc["Year"] >= overlap_min) & (c_pc["Year"] <= overlap_max)]
    pair = pd.merge(c_total, c_pc, on="Year", how="inner")
    if not pair.empty:
        fig_pair = px.scatter(
            pair, x="Total", y="PerCapita", color="Year",
            labels={"Total": "Total CO₂ (metric tons)", "PerCapita": "CO₂ per Capita (tonnes/person)"},
            title="China: CO₂ Total vs CO₂ per Capita",
        )
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("No overlapping years for Total vs Per-Capita within the selected range.")

# -------------------------------------------------
# Tab 4: Per-Capita vs GDP (China)
# -------------------------------------------------
with tab4:
    st.markdown("## Per-Capita Emissions vs GDP per Capita Growth (China)")
    china_gdp = gdp_long[gdp_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "GDP_Growth"}).copy()
    china_gdp["Year"] = pd.to_numeric(china_gdp["Year"], errors="coerce").astype("Int64")
    china_pc = co2_pc_long[co2_pc_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "PerCapita"}).copy()
    china_pc["Year"] = pd.to_numeric(china_pc["Year"], errors="coerce").astype("Int64")
    # intersect with emissions-driven slider range
    gg = pd.merge(china_gdp, china_pc, on="Year", how="inner")
    gg = gg[(gg["Year"] >= year_range[0]) & (gg["Year"] <= year_range[1])]
    if gg.empty:
        st.info("No overlapping data for the selected range.")
    else:
        fig4 = px.scatter(
            gg, x="GDP_Growth", y="PerCapita", color="Year",
            labels={"GDP_Growth": "GDP per Capita Growth (%)", "PerCapita": "CO₂ per Capita (tonnes/person)"},
            title=f"China: CO₂ per Capita vs GDP per Capita Growth ({int(gg['Year'].min())}–{int(gg['Year'].max())})",
        )
        x = gg["GDP_Growth"].to_numpy()
        y = gg["PerCapita"].to_numpy()
        if len(gg) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ys = b + m * xs
            fig4.add_scatter(x=xs, y=ys, mode="lines", name="Fit")
        st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.caption(
    """
**Notes**
- CO₂ totals converted from *thousand tonnes* to *metric tons*.
- Temperature for China converted °C → °F and averaged across numeric columns per year.
- GDP and Energy come from World Bank zip packages.
"""
)
