# dashboard.py
# Streamlit Dashboard for Envecon105 – Group + Individual Results
# Data sources are read directly from your public GitHub repository.

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Envecon105 – CO₂, Energy, GDP & Temperature",
                   page_icon="🌍", layout="wide")

# -------------------------------
# 0) File URLs (public GitHub)
# -------------------------------
RAW_URLS = {
    "gdp_zip": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/gdp_global.zip",
    "energy_zip": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/energy_global.zip",
    "co2_per_capita_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/co2_per_capita.xlsx",
    "disasters_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/natural_disasters_china.xlsx",
    "temp_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/temp_mean_china_cru_1901-2024.xlsx",
    "co2_wide_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/yearly_co2_emissions_1000_tonnes.xlsx"
}

# -------------------------------
# 1) Helpers: file readers
# -------------------------------
@st.cache_data(show_spinner=False)
def read_excel_from_url(url, **kwargs) -> pd.DataFrame:
    """Read an Excel file from a URL into a DataFrame."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), **kwargs)

@st.cache_data(show_spinner=False)
def read_zip_worldbank_csv(url, indicator_guess=None, skiprows=3) -> pd.DataFrame:
    """
    Read a World Bank CSV inside a .zip.
    If multiple CSVs are present, pick the one whose name starts with 'API_'.
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    members = z.namelist()

    # choose the API_*.csv file
    candidates = [m for m in members if m.lower().endswith(".csv") and m.startswith("API_")]
    name = candidates[0] if candidates else members[0]

    with z.open(name) as f:
        df = pd.read_csv(f, skiprows=skiprows)

    # World Bank format typically has these columns
    # ['Country Name','Country Code','Indicator Name','Indicator Code','1960','1961',...]
    # melt to long
    year_cols = [c for c in df.columns if c.isdigit()]
    long_df = df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
                      value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).copy()

    if indicator_guess:
        long_df = long_df[long_df["Indicator Code"].eq(indicator_guess)].copy()

    return long_df.rename(columns={"Country Name": "Country",
                                   "Indicator Name": "Indicator"})

# -------------------------------
# 2) Load & tidy all datasets
# -------------------------------
with st.spinner("Loading data from GitHub…"):
    # World Bank GDP per capita growth (%): NY.GDP.PCAP.KD.ZG
    gdp_long = read_zip_worldbank_csv(RAW_URLS["gdp_zip"], indicator_guess="NY.GDP.PCAP.KD.ZG")
    gdp_long["Indicator"] = "GDP per capita (yearly growth, %)"

    # World Bank energy use per capita (kg of oil equivalent): EG.USE.PCAP.KG.OE
    energy_long = read_zip_worldbank_csv(RAW_URLS["energy_zip"], indicator_guess="EG.USE.PCAP.KG.OE")
    energy_long["Indicator"] = "Energy Use (kg oil eq. per person)"

    # CO₂ per capita (Excel, wide)
    co2_pc_wide = read_excel_from_url(RAW_URLS["co2_per_capita_xlsx"])
    # Expect country in first column and YYYY columns afterwards
    co2_pc_wide.rename(columns={co2_pc_wide.columns[0]: "Country"}, inplace=True)
    co2_pc_long = co2_pc_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_pc_long["Year"] = pd.to_numeric(co2_pc_long["Year"], errors="coerce")
    co2_pc_long = co2_pc_long.dropna(subset=["Year"]).copy()
    co2_pc_long["Indicator"] = "Per_Capita_Emissions"

    # CO₂ totals (Excel, wide; in 1,000 tonnes)
    co2_wide = read_excel_from_url(RAW_URLS["co2_wide_xlsx"])
    co2_wide.rename(columns={co2_wide.columns[0]: "Country"}, inplace=True)
    co2_long = co2_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_long["Year"] = pd.to_numeric(co2_long["Year"], errors="coerce")
    co2_long = co2_long.dropna(subset=["Year"]).copy()
    co2_long["Indicator"] = "CO2 Emissions (Metric Tons)"
    # convert 1,000 tonnes -> metric tonnes
    co2_long["Value"] = pd.to_numeric(co2_long["Value"], errors="coerce") * 1000

    # China temperature (°C) – Excel with metadata rows
    # Your notebook used skiprows=4 and na_values=["-99"]
    temp_cn = read_excel_from_url(RAW_URLS["temp_xlsx"], skiprows=4, na_values=["-99"])
    # We expect columns like 'Year', 'China' or a single annual mean series.
    # Try to auto-detect a temperature column:
    # Keep numeric columns except Year, then sum across if needed (robust fallback)
    if "Year" not in temp_cn.columns:
        # Some files label it 'year' or similar
        possible_year = [c for c in temp_cn.columns if str(c).strip().lower() == "year"]
        if possible_year:
            temp_cn.rename(columns={possible_year[0]: "Year"}, inplace=True)

    # Reduce to Year + a single 'Temperature' column in °F for your charts
    temp_cn = temp_cn.dropna(axis=1, how="all")
    num_cols = [c for c in temp_cn.columns if c != "Year" and pd.api.types.is_numeric_dtype(temp_cn[c])]
    # --- robust normalizer for the temperature table ---
def normalize_year_and_numeric(df):
    # 1) normalize column names
    df = df.rename(columns=lambda c: str(c).strip())

    # 2) if year is in the index, bring it back as a column
    if df.index.name and "year" in str(df.index.name).lower():
        df = df.reset_index()

    # 3) find the 'year' column case/space-insensitively
    year_col = None
    for c in df.columns:
        if "year" in c.lower():
            year_col = c
            break

    # 4) coerce numerics (sometimes read as object)
    for c in df.columns:
        if c != year_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) choose a reasonable numeric temp column if you only want one
    #    (or keep them all if your chart expects multiple)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if year_col and year_col in numeric_cols:
        numeric_cols.remove(year_col)

    return df, year_col, numeric_cols

temp_cn, year_col, num_cols = normalize_year_and_numeric(temp_cn)

    # Collapse to one numeric series (mean across numeric columns)
  tmp["Temperature_C"] = tmp[num_cols].mean(axis=1, skipna=True)
    temperature_cn = tmp[["Year", "Temperature_C"]].dropna().copy()
    temperature_cn["Country"] = "China"
    temperature_cn["Indicator"] = "Temperature"
    # Convert °C -> °F to match your individual notebook visuals
    temperature_cn["Value"] = temperature_cn["Temperature_C"] * 9 / 5 + 32
    temperature_cn = temperature_cn[["Country", "Year", "Indicator", "Value"]]

    # China natural disasters (counts)
    dis_cn_raw = read_excel_from_url(RAW_URLS["disasters_xlsx"])
    # Try to find 'Year' and collapse various disaster types to total counts
    dis_cn = dis_cn_raw.copy()
    if "Year" not in dis_cn.columns:
        maybe_year = [c for c in dis_cn.columns if str(c).strip().lower() == "year"]
        if maybe_year:
            dis_cn.rename(columns={maybe_year[0]: "Year"}, inplace=True)

    num_cols = [c for c in dis_cn.columns if c != "Year" and pd.api.types.is_numeric_dtype(dis_cn[c])]
    if "Year" in dis_cn.columns and num_cols:
        dis_cn["Value"] = dis_cn[num_cols].sum(axis=1, skipna=True)
        disasters_cn = dis_cn[["Year", "Value"]].dropna().copy()
        disasters_cn["Country"] = "China"
        disasters_cn["Indicator"] = "Disasters"
        disasters_cn = disasters_cn[["Country", "Year", "Indicator", "Value"]]
    else:
        disasters_cn = pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

# Unified long table (for flexible plotting)
data_long = pd.concat([
    co2_long[["Country", "Year", "Indicator", "Value"]],
    energy_long[["Country", "Year", "Indicator", "Value"]],
    gdp_long[["Country", "Year", "Indicator", "Value"]],
    co2_pc_long[["Country", "Year", "Indicator", "Value"]],
    temperature_cn,
    disasters_cn
], ignore_index=True)

# Region flag for comparisons
data_long["Region"] = np.where(data_long["Country"] == "China", "China", "Rest of the World")

# -------------------------------
# 3) Sidebar controls
# -------------------------------
countries = sorted(co2_long["Country"].dropna().unique().tolist())
default_country = "China" if "China" in countries else countries[0]

st.sidebar.header("Controls")
focus_country = st.sidebar.selectbox("Country focus", countries, index=countries.index(default_country))
yr_min, yr_max = int(data_long["Year"].min()), int(data_long["Year"].max())
year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(max(1900, yr_min), yr_max), step=1)

show_smoothed = st.sidebar.checkbox("Show smoothed lines (rolling)", value=True)
window = st.sidebar.slider("Smoothing window (years)", 3, 11, 5, step=2) if show_smoothed else 1

# Filtered view
df = data_long[(data_long["Year"] >= year_range[0]) & (data_long["Year"] <= year_range[1])].copy()

# -------------------------------
# 4) Layout (tabs)
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Global & Top Emitters", "China Deep-dive", "Per-Capita vs GDP"
])

# -------------------------------------------------
# Tab 1: Overview – quick KPIs and global trend
# -------------------------------------------------
with tab1:
    st.markdown("## Overview")
    colA, colB, colC, colD = st.columns(4)

    latest_year = int(df["Year"].max())
    world_emis = co2_long[co2_long["Year"] == latest_year]["Value"].sum()
    china_emis = co2_long[(co2_long["Country"] == "China") & (co2_long["Year"] == latest_year)]["Value"].sum()
    china_share = (china_emis / world_emis * 100) if world_emis else np.nan

    colA.metric("Latest year", latest_year)
    colB.metric("Global CO₂ (Mt)", f"{world_emis/1e6:,.1f}")
    colC.metric("China CO₂ (Mt)", f"{china_emis/1e6:,.1f}")
    colD.metric("China share of global", f"{china_share:,.1f}%")

    st.markdown("### Global CO₂ Emissions over time (sum of all countries)")
    global_emissions = co2_long.groupby("Year", as_index=False)["Value"].sum()
    fig = px.line(global_emissions, x="Year", y="Value", labels={"Value": "CO₂ (metric tons)"},
                  title="World CO₂ Emissions per Year")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Tab 2: Global & Top Emitters (heatmap + lines)
# -------------------------------------------------
with tab2:
    st.markdown("## Global & Top Emitters")

    # 2a) Country lines with one focus highlighted
    st.markdown("### Country CO₂ trajectories (highlight focus country)")

    # Determine top N emitters in latest year to declutter
    latest = co2_long[co2_long["Year"] == latest_year].sort_values("Value", ascending=False)
    top_countries = latest["Country"].head(12).tolist()

    subset = co2_long[co2_long["Country"].isin(top_countries)]
    fig2 = px.line(subset, x="Year", y="Value", color="Country",
                   line_group="Country",
                   title=f"Top 12 CO₂ Emitting Countries (latest year = {latest_year})",
                   labels={"Value": "CO₂ (metric tons)"})
    # fade non-focus with lower alpha
    for i, d in enumerate(fig2.data):
        if d.name != focus_country:
            fig2.data[i].line.width = 1
            fig2.data[i].opacity = 0.35
        else:
            fig2.data[i].line.width = 3
            fig2.data[i].opacity = 1.0
    st.plotly_chart(fig2, use_container_width=True)

    # 2b) Heatmap for top 10 emitters (log scale feel via color)
    st.markdown("### Heatmap: Top 10 CO₂ Emission-producing Countries (1900–latest)")
    last_year_avail = int(co2_long["Year"].max())
    top10 = (co2_long[co2_long["Year"] == last_year_avail]
             .sort_values("Value", ascending=False)
             .head(10)["Country"]
             .tolist())
    tile_df = (co2_long[(co2_long["Country"].isin(top10)) & (co2_long["Year"] >= 1900)]
               .pivot_table(index="Country", columns="Year", values="Value", aggfunc="sum")
               .fillna(0.0))
    # small log transform for color contrast
    Z = np.log(tile_df.replace({0: np.nan}))
    fig3 = px.imshow(Z, aspect="auto", color_continuous_scale="viridis",
                     labels=dict(color="ln(CO₂)"),
                     title=f"Top 10 CO₂ Emissions – ordered by {last_year_avail}")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# Tab 3: China Deep-dive – emissions & temp
# -------------------------------------------------
with tab3:
    st.markdown("## China Deep-dive: Emissions, Temperature & Relationships")

    c_em = co2_long[co2_long["Country"] == "China"].copy()
    c_temp = temperature_cn.copy()

    # Align years
    merged = pd.merge(c_em[["Year", "Value"]].rename(columns={"Value": "Emissions"}),
                      c_temp[["Year", "Value"]].rename(columns={"Value": "Temperature_F"}),
                      on="Year", how="inner")
    merged = merged[(merged["Year"] >= year_range[0]) & (merged["Year"] <= year_range[1])].copy()

    # Panels: line plots with optional smoothing
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CO₂ Emissions (China)")
        ser = merged[["Year", "Emissions"]].set_index("Year").sort_index()
        if show_smoothed and window > 1:
            ser["Smoothed"] = ser["Emissions"].rolling(window, center=True, min_periods=1).mean()
            st.line_chart(ser)
        else:
            st.line_chart(ser)

    with col2:
        st.markdown("#### Temperature (°F, China)")
        ser = merged[["Year", "Temperature_F"]].set_index("Year").sort_index()
        if show_smoothed and window > 1:
            ser["Smoothed"] = ser["Temperature_F"].rolling(window, center=True, min_periods=1).mean()
            st.line_chart(ser)
        else:
            st.line_chart(ser)

    # Scatter: scaled emissions vs temperature (like your individual project)
    st.markdown("### Scaled Temperature vs CO₂ Emissions (China)")
    from sklearn.preprocessing import StandardScaler

    if len(merged) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(merged[["Emissions", "Temperature_F"]])
        x, y = scaled[:, 0], scaled[:, 1]

        # regression line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = b + m * x_line

        fig_sc = px.scatter(x=x, y=y, labels={"x": "Scaled Emissions", "y": "Scaled Temperature"},
                            title=f"China CO₂ Emissions and Temperature ({year_range[0]}–{year_range[1]})")
        fig_sc.add_traces(px.line(x=x_line, y=y_line).data)
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Not enough overlapping years to plot the scaled scatter.")

    # Per-capita vs total (your new graph)
    st.markdown("### CO₂ per Capita vs Total CO₂ (China)")
    c_total = co2_long[co2_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "Total"})
    c_pc = co2_pc_long[co2_pc_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "PerCapita"})
    pair = pd.merge(c_total, c_pc, on="Year", how="inner")
    fig_pair = px.scatter(pair, x="Total", y="PerCapita", color="Year",
                          labels={"Total": "Total CO₂ (metric tons)",
                                  "PerCapita": "CO₂ per Capita (tonnes/person)"},
                          title="China: CO₂ Total vs CO₂ per Capita")
    # 1:1 guideline (scaled to visible range)
    if not pair.empty:
        x_min, x_max = pair["Total"].min(), pair["Total"].max()
        y_min, y_max = pair["PerCapita"].min(), pair["PerCapita"].max()
        slope = (y_max - y_min) / (x_max - x_min) if x_max > x_min else 0
    st.plotly_chart(fig_pair, use_container_width=True)

# -------------------------------------------------
# Tab 4: Per-Capita vs GDP (China)
# -------------------------------------------------
with tab4:
    st.markdown("## Per-Capita Emissions vs GDP per Capita Growth (China)")

    china_gdp = gdp_long[gdp_long["Country"] == "China"][["Year", "Value"]].rename(
        columns={"Value": "GDP_Growth"})
    china_pc = co2_pc_long[co2_pc_long["Country"] == "China"][["Year", "Value"]].rename(
        columns={"Value": "PerCapita"})

    gg = pd.merge(china_gdp, china_pc, on="Year", how="inner")
    gg = gg[(gg["Year"] >= year_range[0]) & (gg["Year"] <= year_range[1])]
    if gg.empty:
        st.info("No overlapping data for the selected range.")
    else:
        fig4 = px.scatter(gg, x="GDP_Growth", y="PerCapita", color="Year",
                          labels={"GDP_Growth": "GDP per Capita Growth (%)",
                                  "PerCapita": "CO₂ per Capita (tonnes/person)"},
                          trendline="ols",
                          title=f"China: CO₂ per Capita vs GDP per Capita Growth ({year_range[0]}–{year_range[1]})")
        st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.caption("""
**Notes**
- CO₂ totals converted from *thousand tonnes* to *metric tons*.
- Temperature data for China converted °C → °F and collapsed to an annual mean.
- GDP growth and Energy use are from World Bank (zip packages).
""")

streamlit
pandas
numpy
requests
plotly
matplotlib
scikit-learn
