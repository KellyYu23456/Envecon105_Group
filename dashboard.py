# dashboard.py
# Streamlit Dashboard for Envecon105 – Group + Individual Results
# Reads all data from your public GitHub repository.

from __future__ import annotations

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px

st.set_page_config(
    page_title="Envecon105 – CO₂, Energy, GDP & Temperature",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------
# 0) File URLs (public GitHub)
# -----------------------------------
RAW_URLS = {
    "gdp_zip":           "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/gdp_global.zip",
    "energy_zip":        "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/energy_global.zip",
    "co2_per_capita_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/co2_per_capita.xlsx",
    "disasters_xlsx":    "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/natural_disasters_china.xlsx",
    "temp_xlsx":         "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/temp_mean_china_cru_1901-2024.xlsx",
    "co2_wide_xlsx":     "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/yearly_co2_emissions_1000_tonnes.xlsx",
}

# -----------------------------------
# Helpers
# -----------------------------------
def _coerce_year(series: pd.Series) -> pd.Series:
    """Coerce any 'year-like' series to integer years, dropping NaNs."""
    y = pd.to_numeric(series, errors="coerce").dropna()
    return y.astype(int)

@st.cache_data(show_spinner=False)
def read_excel_from_url(url: str, **kwargs) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), **kwargs)

@st.cache_data(show_spinner=False)
def read_zip_worldbank_csv(url: str, indicator_guess: str | None = None, skiprows: int = 3) -> pd.DataFrame:
    """
    Read a World Bank CSV inside a .zip and return a long DataFrame:
    columns: Country, Country Code, Indicator, Indicator Code, Year, Value
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    members = z.namelist()

    # Prefer API_*.csv inside the zip
    candidates = [m for m in members if m.lower().endswith(".csv") and m.startswith("API_")]
    name = candidates[0] if candidates else members[0]

    # Some WB files need skiprows=4
    try:
        with z.open(name) as f:
            df = pd.read_csv(f, skiprows=skiprows)
    except Exception:
        with z.open(name) as f:
            df = pd.read_csv(f, skiprows=4)

    # Expected WB layout: years as columns
    year_cols = [c for c in df.columns if str(c).isdigit()]
    long_df = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).copy()
    long_df["Year"] = long_df["Year"].astype(int)
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

    if indicator_guess:
        long_df = long_df[long_df["Indicator Code"].eq(indicator_guess)].copy()

    long_df = long_df.rename(columns={"Country Name": "Country", "Indicator Name": "Indicator"})
    return long_df

def normalize_year_and_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Make 'Year' column consistent and list numeric columns (excluding Year)."""
    df = df.rename(columns=lambda c: str(c).strip())
    # Bring an index 'year' back as column if present
    if df.index.name and "year" in str(df.index.name).lower():
        df = df.reset_index()

    # Try to find a year column
    year_col = None
    for c in df.columns:
        if "year" in c.lower():
            year_col = c
            break

    # Coerce numerics
    for c in df.columns:
        if c != year_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if year_col and year_col in numeric_cols:
        numeric_cols.remove(year_col)

    # Standardize to Year as int if possible
    if year_col:
        df["Year"] = _coerce_year(df[year_col])
    return df, ("Year" if year_col else None), numeric_cols

def safe_overlap(df1: pd.DataFrame, df2: pd.DataFrame, col="Year") -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Filter two frames to the intersection of years and return (df1f, df2f, sorted_years)."""
    y1 = set(df1[col].dropna().astype(int))
    y2 = set(df2[col].dropna().astype(int))
    common = sorted(y1 & y2)
    if not common:
        return df1.iloc[0:0].copy(), df2.iloc[0:0].copy(), []
    return df1[df1[col].isin(common)].copy(), df2[df2[col].isin(common)].copy(), common

# -----------------------------------
# 1) Load & tidy all datasets
# -----------------------------------
with st.spinner("Loading data from GitHub…"):
    # GDP per capita growth (%): NY.GDP.PCAP.KD.ZG
    gdp_long = read_zip_worldbank_csv(RAW_URLS["gdp_zip"], indicator_guess="NY.GDP.PCAP.KD.ZG")
    gdp_long["Indicator"] = "GDP per capita (yearly growth, %)"

    # Energy use per capita (kg oil eq): EG.USE.PCAP.KG.OE
    energy_long = read_zip_worldbank_csv(RAW_URLS["energy_zip"], indicator_guess="EG.USE.PCAP.KG.OE")
    energy_long["Indicator"] = "Energy Use (kg oil eq. per person)"

    # CO₂ per capita (Excel, wide)
    co2_pc_wide = read_excel_from_url(RAW_URLS["co2_per_capita_xlsx"])
    co2_pc_wide = co2_pc_wide.rename(columns={co2_pc_wide.columns[0]: "Country"})
    co2_pc_long = co2_pc_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_pc_long["Year"] = _coerce_year(co2_pc_long["Year"])
    co2_pc_long["Value"] = pd.to_numeric(co2_pc_long["Value"], errors="coerce")
    co2_pc_long["Indicator"] = "Per_Capita_Emissions"

    # CO₂ totals (Excel, wide; 1,000 tonnes -> metric tons)
    co2_wide = read_excel_from_url(RAW_URLS["co2_wide_xlsx"])
    co2_wide = co2_wide.rename(columns={co2_wide.columns[0]: "Country"})
    co2_long = co2_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_long["Year"] = _coerce_year(co2_long["Year"])
    co2_long["Value"] = pd.to_numeric(co2_long["Value"], errors="coerce") * 1000.0
    co2_long["Indicator"] = "CO2 Emissions (Metric Tons)"

    # China temperature (°C) – Excel with metadata rows
    temp_raw = read_excel_from_url(RAW_URLS["temp_xlsx"], skiprows=4, na_values=["-99"])
    temp_raw, temp_year_col, temp_num_cols = normalize_year_and_numeric(temp_raw)

    if temp_year_col is None or not temp_num_cols:
        temperature_cn = pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])
    else:
        # collapse multiple numeric columns to one series (mean)
        tmp = temp_raw.copy()
        tmp["Temperature_C"] = tmp[temp_num_cols].mean(axis=1, skipna=True)
        temperature_cn = tmp[["Year", "Temperature_C"]].dropna().copy()
        temperature_cn["Country"] = "China"
        temperature_cn["Indicator"] = "Temperature"
        temperature_cn["Value"] = temperature_cn["Temperature_C"] * 9 / 5 + 32
        temperature_cn = temperature_cn[["Country", "Year", "Indicator", "Value"]]

    # China natural disasters (counts)
    dis_cn_raw = read_excel_from_url(RAW_URLS["disasters_xlsx"])
    dis_cn_raw, dis_year_col, dis_num_cols = normalize_year_and_numeric(dis_cn_raw)
    if dis_year_col and dis_num_cols:
        dis_cn_raw["Value"] = dis_cn_raw[dis_num_cols].sum(axis=1, skipna=True)
        disasters_cn = dis_cn_raw[["Year", "Value"]].dropna().copy()
        disasters_cn["Country"] = "China"
        disasters_cn["Indicator"] = "Disasters"
        disasters_cn = disasters_cn[["Country", "Year", "Indicator", "Value"]]
    else:
        disasters_cn = pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

# Unified long table
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

# -----------------------------------
# 2) Sidebar controls
# -----------------------------------
countries = sorted(co2_long["Country"].dropna().unique().tolist())
default_country = "China" if "China" in countries else (countries[0] if countries else "China")

st.sidebar.header("Controls")
focus_country = st.sidebar.selectbox("Country focus", countries, index=countries.index(default_country))

# Use the *global* year limits across all series to keep the slider stable
yr_min = int(data_long["Year"].min()) if not data_long.empty else 1900
yr_max = int(data_long["Year"].max()) if not data_long.empty else 2020
year_range = st.sidebar.slider(
    "Year range",
    min_value=yr_min,
    max_value=yr_max,
    value=(max(1900, yr_min), min(yr_max, 2015)),  # center to a likely overlap
    step=1,
)

show_smoothed = st.sidebar.checkbox("Show smoothed lines (rolling)", value=True)
window = st.sidebar.slider("Smoothing window (years)", 3, 11, 5, step=2) if show_smoothed else 1

# Filtered view (for plots that aggregate across indicators we will subset again)
df_filtered = data_long[(data_long["Year"] >= year_range[0]) & (data_long["Year"] <= year_range[1])].copy()

# -----------------------------------
# 3) Layout (tabs)
# -----------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Global & Top Emitters", "China Deep-dive", "Per-Capita vs GDP"])

# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab1:
    st.markdown("## Overview")

    # KPIs should use the CO₂ table's own max year to avoid mixing with temp years.
    if not co2_long.empty:
        latest_year_co2 = int(co2_long["Year"].max())
        world_emis = co2_long[co2_long["Year"] == latest_year_co2]["Value"].sum()
        china_emis = co2_long[(co2_long["Country"] == "China") & (co2_long["Year"] == latest_year_co2)]["Value"].sum()
        china_share = (china_emis / world_emis * 100) if world_emis else np.nan
    else:
        latest_year_co2, world_emis, china_emis, china_share = yr_max, 0.0, 0.0, np.nan

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Latest year (CO₂ table)", latest_year_co2)
    colB.metric("Global CO₂ (Mt)", f"{world_emis/1e6:,.1f}")
    colC.metric("China CO₂ (Mt)", f"{china_emis/1e6:,.1f}")
    colD.metric("China share of global", "-" if np.isnan(china_share) else f"{china_share:,.1f}%")

    st.markdown("### Global CO₂ Emissions over time (sum of all countries)")
    if co2_long.empty:
        st.info("CO₂ table is empty.")
    else:
        global_emissions = co2_long.groupby("Year", as_index=False)["Value"].sum()
        fig = px.line(global_emissions, x="Year", y="Value",
                      labels={"Value": "CO₂ (metric tons)"},
                      title="World CO₂ Emissions per Year")
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab 2: Global & Top Emitters
# -------------------------------
with tab2:
    st.markdown("## Global & Top Emitters")

    if co2_long.empty:
        st.info("CO₂ table is empty; cannot draw country trajectories.")
    else:
        st.markdown("### Country CO₂ trajectories (highlight focus country)")
        latest = co2_long.loc[co2_long["Year"].idxmax()]
        latest_year_for_rank = int(latest["Year"])
        latest_rank = (co2_long[co2_long["Year"] == latest_year_for_rank]
                       .sort_values("Value", ascending=False))
        top_countries = latest_rank["Country"].head(12).tolist()

        subset = co2_long[co2_long["Country"].isin(top_countries)]
        fig2 = px.line(subset, x="Year", y="Value", color="Country",
                       line_group="Country",
                       title=f"Top 12 CO₂ Emitting Countries (latest year = {latest_year_for_rank})",
                       labels={"Value": "CO₂ (metric tons)"})
        for i, d in enumerate(fig2.data):
            if d.name != focus_country:
                fig2.data[i].line.width = 1
                fig2.data[i].opacity = 0.35
            else:
                fig2.data[i].line.width = 3
                fig2.data[i].opacity = 1.0
        st.plotly_chart(fig2, use_container_width=True)

        # Heatmap
        st.markdown("### Heatmap: Top 10 CO₂ Emission-producing Countries (since 1900)")
        last_year_avail = int(co2_long["Year"].max())
        top10 = (co2_long[co2_long["Year"] == last_year_avail]
                 .sort_values("Value", ascending=False)
                 .head(10)["Country"].tolist())
        tile_df = (co2_long[(co2_long["Country"].isin(top10)) & (co2_long["Year"] >= 1900)]
                   .pivot_table(index="Country", columns="Year", values="Value", aggfunc="sum")
                   .fillna(0.0))
        Z = np.log(tile_df.replace({0: np.nan}))
        fig3 = px.imshow(Z, aspect="auto", color_continuous_scale="viridis",
                         labels=dict(color="ln(CO₂)"),
                         title=f"Top 10 CO₂ Emissions – ordered by {last_year_avail}")
        st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# -------------------------------------------------
# Tab 3: China Deep-dive – emissions & temperature
# -------------------------------------------------
with tab3:
    st.markdown("## China Deep-dive: Emissions, Temperature & Relationships")

    # --- series
    c_em = co2_long[co2_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "Emissions"}).copy()
    c_tp = temperature_cn[["Year", "Value"]].rename(columns={"Value": "Temperature_F"}).copy()

    # --- coerce years
    for df_ in (c_em, c_tp):
        df_["Year"] = pd.to_numeric(df_["Year"], errors="coerce")
        df_.dropna(subset=["Year"], inplace=True)
        df_["Year"] = df_["Year"].astype(int)

    # --- overlap with selected range
    yr_set = set(range(year_range[0], year_range[1] + 1))
    em_years = set(c_em["Year"].tolist())
    tp_years = set(c_tp["Year"].tolist())
    common_years = sorted((em_years & tp_years) & yr_set)

    # quick telemetry so you can verify what's available
    st.caption(
        f"Data points — Emissions (CN): {len(c_em)}, "
        f"Temperature (CN): {len(c_tp)}, "
        f"Overlap in selected range: {len(common_years)} years."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### CO₂ Emissions (China)")
        if not common_years:
            st.info("No overlapping years between emissions and the selected range.")
        else:
            em_show = c_em[c_em["Year"].isin(common_years)].sort_values("Year").set_index("Year")
            ser = em_show[["Emissions"]].copy()
            if show_smoothed and window > 1:
                ser["Smoothed"] = ser["Emissions"].rolling(window, center=True, min_periods=1).mean()
            st.line_chart(ser)

    with col2:
        st.markdown("#### Temperature (°F, China)")
        if not common_years:
            st.info("No overlapping years between temperature and the selected range.")
        else:
            tp_show = c_tp[c_tp["Year"].isin(common_years)].sort_values("Year").set_index("Year")
            ser = tp_show[["Temperature_F"]].copy()
            if show_smoothed and window > 1:
                ser["Smoothed"] = ser["Temperature_F"].rolling(window, center=True, min_periods=1).mean()
            st.line_chart(ser)

    st.markdown("### Scaled Temperature vs CO₂ Emissions (China)")
    if len(common_years) < 3:
        st.info("Not enough overlapping years to plot the scaled scatter.")
    else:
        merged = (c_em[c_em["Year"].isin(common_years)]
                  .merge(c_tp[c_tp["Year"].isin(common_years)], on="Year", how="inner")
                  .sort_values("Year"))

        # numeric safety
        merged["Emissions"] = pd.to_numeric(merged["Emissions"], errors="coerce")
        merged["Temperature_F"] = pd.to_numeric(merged["Temperature_F"], errors="coerce")
        merged = merged.dropna(subset=["Emissions", "Temperature_F"])

        if len(merged) >= 3:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled = scaler.fit_transform(merged[["Emissions", "Temperature_F"]])
            x, y = scaled[:, 0], scaled[:, 1]

            # regression line (numpy)
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = b + m * x_line

            fig_sc = px.scatter(
                x=x, y=y,
                labels={"x": "Scaled Emissions", "y": "Scaled Temperature"},
                title=f"China CO₂ Emissions and Temperature ({common_years[0]}–{common_years[-1]})"
            )
            fig_sc.add_traces(px.line(x=x_line, y=y_line).data)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Not enough non-missing points to draw the scatter.")

# -------------------------------
# Tab 4: Per-Capita vs GDP Growth
# -------------------------------
with tab4:
    st.markdown("## Per-Capita Emissions vs GDP per Capita Growth (China)")

    # Prepare inputs
    china_gdp = gdp_long[gdp_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "GDP_Growth"})
    china_pc  = co2_pc_long[co2_pc_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "PerCapita"})

    # Coerce years to int
    if not china_gdp.empty:
        china_gdp["Year"] = pd.to_numeric(china_gdp["Year"], errors="coerce").dropna().astype(int)
    if not china_pc.empty:
        china_pc["Year"]  = pd.to_numeric(china_pc["Year"], errors="coerce").dropna().astype(int)

    # Enforce overlap first, then merge so we have BOTH columns
    g1, g2, common = safe_overlap(china_gdp, china_pc, "Year")

    if not common:
        st.info("No overlapping data between GDP growth and CO₂ per capita.")
    else:
        gg = pd.merge(g1, g2, on="Year", how="inner")
        # clip to selected year range
        gg = gg[(gg["Year"] >= year_range[0]) & (gg["Year"] <= year_range[1])].copy()

        # make sure plotting columns are numeric and drop rows that are NaN
        gg["GDP_Growth"] = pd.to_numeric(gg["GDP_Growth"], errors="coerce")
        gg["PerCapita"]  = pd.to_numeric(gg["PerCapita"],  errors="coerce")
        gg = gg.dropna(subset=["GDP_Growth", "PerCapita"])

        if gg.empty:
            st.info("No overlapping data for the selected year range.")
        else:
            fig4 = px.scatter(
                gg,
                x="GDP_Growth",
                y="PerCapita",
                color="Year",
                labels={
                    "GDP_Growth": "GDP per Capita Growth (%)",
                    "PerCapita": "CO₂ per Capita (tonnes/person)",
                },
                # no trendline to avoid statsmodels dependency
                title=f"China: CO₂ per Capita vs GDP per Capita Growth ({year_range[0]}–{year_range[1]})",
            )
            st.plotly_chart(fig4, use_container_width=True)
# -------------------------------
# Footer
# -------------------------------
st.caption(
    """
**Notes**
- CO₂ totals converted from *thousand tonnes* to *metric tons*.
- Temperature data for China converted °C → °F and collapsed to an annual mean.
- GDP growth and Energy use are from World Bank (ZIP packages).
- Plots automatically harmonize year types and use only the overlapping years.
"""
)
