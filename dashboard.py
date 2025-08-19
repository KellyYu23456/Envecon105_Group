# dashboard.py
# Streamlit Dashboard for Envecon105 – Group + Individual Results
# Data sources are read directly from your public GitHub repository.

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px

# --------------------------------------------------------------------------------------
# App config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Envecon105 – CO₂, Energy, GDP & Temperature",
    page_icon="🌍",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# 0) File URLs (public GitHub)
# --------------------------------------------------------------------------------------
RAW_URLS = {
    "gdp_zip": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/gdp_global.zip",
    "energy_zip": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/energy_global.zip",
    "co2_per_capita_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/co2_per_capita.xlsx",
    "disasters_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/natural_disasters_china.xlsx",
    "temp_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/temp_mean_china_cru_1901-2024.xlsx",
    "co2_wide_xlsx": "https://raw.githubusercontent.com/KellyYu23456/Envecon105_Group/main/yearly_co2_emissions_1000_tonnes.xlsx",
}

# --------------------------------------------------------------------------------------
# 1) Helpers: readers (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_excel_from_url(url, **kwargs) -> pd.DataFrame:
    """Read an Excel file from a URL into a DataFrame."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), **kwargs)

@st.cache_data(show_spinner=False)
def read_zip_worldbank_csv(url, indicator_guess=None, skiprows=3) -> pd.DataFrame:
    """
    Read a World Bank CSV inside a .zip. If multiple CSVs are present, pick the one
    whose name starts with 'API_'. Return a long table (Country, Country Code,
    Indicator, Indicator Code, Year, Value).
    """
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
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).copy()

    if indicator_guess:
        long_df = long_df[long_df["Indicator Code"].eq(indicator_guess)].copy()

    long_df = long_df.rename(columns={"Country Name": "Country", "Indicator Name": "Indicator"})
    return long_df[["Country", "Country Code", "Indicator", "Indicator Code", "Year", "Value"]]

# ---------- SMART China temperature loader (handles Ann/Monthly/odd headers) ----------
def load_china_temperature(url: str) -> pd.DataFrame:
    """
    Tries hard to get a (Year, annual temperature °F) series from the Excel file.
    - Tries several header offsets (skiprows).
    - Accepts 'Year' in any case / as index.
    - If a clear annual column exists (Ann/Annual/Mean), uses it.
    - Otherwise, if monthly columns exist (Jan..Dec), computes their mean.
    - Converts °C -> °F and returns (Country, Year, Indicator, Value).
    """
    def _try_read(skiprows):
        try:
            return read_excel_from_url(url, skiprows=skiprows, na_values=["-99", -99])
        except Exception:
            return None

    # Try a few common header offsets
    candidates = [0, 1, 2, 3, 4, 5]
    raw = None
    for sr in candidates:
        raw = _try_read(sr)
        if isinstance(raw, pd.DataFrame) and len(raw.columns) > 1 and len(raw) > 1:
            break
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

    # Normalize names
    raw.columns = [str(c).strip() for c in raw.columns]

    # Find or reconstruct Year
    year_col = None
    for c in raw.columns:
        if "year" in c.lower():
            year_col = c
            break
    if year_col is None and raw.index.name and "year" in str(raw.index.name).lower():
        raw = raw.reset_index()
        year_col = raw.columns[0]
    if year_col is None:
        # If first column looks like years, adopt it
        first = raw.columns[0]
        if pd.to_numeric(raw[first], errors="coerce").notna().mean() > 0.8:
            year_col = first
    if year_col is None:
        return pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

    df = raw.copy()
    df.rename(columns={year_col: "Year"}, inplace=True)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])

    # Annual column candidates
    ann_candidates = [c for c in df.columns
                      if c != "Year" and any(k in c.lower() for k in ["ann", "annual", "mean", "avg"])]
    ann_col = None
    if ann_candidates:
        strict = [c for c in ann_candidates if "ann" in c.lower()]
        ann_col = strict[0] if strict else ann_candidates[0]

    if ann_col is None:
        months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        def is_month_col(c):
            lc = c.lower()
            return any(m in lc for m in months)
        month_cols = [c for c in df.columns if c != "Year" and is_month_col(c)]
        month_cols = [c for c in month_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(month_cols) >= 6:
            df["Temperature_C"] = df[month_cols].mean(axis=1, skipna=True)
        else:
            num_cols = [c for c in df.columns if c != "Year" and pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df["Temperature_C"] = df[num_cols].mean(axis=1, skipna=True)
            else:
                return pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])
    else:
        df["Temperature_C"] = pd.to_numeric(df[ann_col], errors="coerce")

    out = df[["Year", "Temperature_C"]].copy()
    out["Value"] = out["Temperature_C"] * 9 / 5 + 32  # °C -> °F
    out = out.dropna(subset=["Year", "Value"])
    out["Country"] = "China"
    out["Indicator"] = "Temperature"
    return out[["Country", "Year", "Indicator", "Value"]]

def load_china_disasters(url: str) -> pd.DataFrame:
    """Collapse a China disasters table to (Country, Year, Indicator='Disasters', Value)."""
    try:
        dis = read_excel_from_url(url)
    except Exception:
        return pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

    dis.columns = [str(c).strip() for c in dis.columns]
    # match Year
    year_col = None
    for c in dis.columns:
        if "year" in c.lower():
            year_col = c
            break
    if year_col is None and dis.index.name and "year" in str(dis.index.name).lower():
        dis = dis.reset_index()
        year_col = dis.columns[0]
    if year_col is None:
        # fallback: first column looks like years?
        first = dis.columns[0]
        if pd.to_numeric(dis[first], errors="coerce").notna().mean() > 0.8:
            year_col = first
    if year_col is None:
        return pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

    dis.rename(columns={year_col: "Year"}, inplace=True)
    dis["Year"] = pd.to_numeric(dis["Year"], errors="coerce")
    dis = dis.dropna(subset=["Year"])

    num_cols = [c for c in dis.columns if c != "Year" and pd.api.types.is_numeric_dtype(dis[c])]
    if not num_cols:
        return pd.DataFrame(columns=["Country", "Year", "Indicator", "Value"])

    out = pd.DataFrame({
        "Year": dis["Year"],
        "Value": dis[num_cols].sum(axis=1, skipna=True)
    })
    out["Country"] = "China"
    out["Indicator"] = "Disasters"
    return out[["Country", "Year", "Indicator", "Value"]]

# --------------------------------------------------------------------------------------
# 2) Load & tidy all datasets
# --------------------------------------------------------------------------------------
with st.spinner("Loading data from GitHub…"):
    # GDP per capita growth (%): NY.GDP.PCAP.KD.ZG
    gdp_long = read_zip_worldbank_csv(RAW_URLS["gdp_zip"], indicator_guess="NY.GDP.PCAP.KD.ZG")
    gdp_long["Indicator"] = "GDP per capita growth (%)"

    # Energy use per capita (kg of oil equivalent): EG.USE.PCAP.KG.OE
    energy_long = read_zip_worldbank_csv(RAW_URLS["energy_zip"], indicator_guess="EG.USE.PCAP.KG.OE")
    energy_long["Indicator"] = "Energy per capita (kg oil eq.)"

    # CO₂ per capita (Excel wide -> long)
    co2_pc_wide = read_excel_from_url(RAW_URLS["co2_per_capita_xlsx"])
    co2_pc_wide.rename(columns={co2_pc_wide.columns[0]: "Country"}, inplace=True)
    co2_pc_long = co2_pc_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_pc_long["Year"] = pd.to_numeric(co2_pc_long["Year"], errors="coerce")
    co2_pc_long = co2_pc_long.dropna(subset=["Year"]).copy()
    co2_pc_long["Indicator"] = "CO₂ per capita (t/person)"

    # CO₂ totals (Excel wide -> long; file values are in 1,000 tonnes)
    co2_wide = read_excel_from_url(RAW_URLS["co2_wide_xlsx"])
    co2_wide.rename(columns={co2_wide.columns[0]: "Country"}, inplace=True)
    co2_long = co2_wide.melt(id_vars="Country", var_name="Year", value_name="Value")
    co2_long["Year"] = pd.to_numeric(co2_long["Year"], errors="coerce")
    co2_long = co2_long.dropna(subset=["Year"]).copy()
    co2_long["Value"] = pd.to_numeric(co2_long["Value"], errors="coerce") * 1000.0  # -> metric tons
    co2_long["Indicator"] = "CO₂ total (metric tons)"

    # China temperature (robust loader)
    temperature_cn = load_china_temperature(RAW_URLS["temp_xlsx"])

    # China natural disasters
    disasters_cn = load_china_disasters(RAW_URLS["disasters_xlsx"])

# Unified long table for flexible plotting
data_long = pd.concat(
    [
        co2_long[["Country", "Year", "Indicator", "Value"]],
        energy_long[["Country", "Year", "Indicator", "Value"]],
        gdp_long[["Country", "Year", "Indicator", "Value"]],
        co2_pc_long[["Country", "Year", "Indicator", "Value"]],
        temperature_cn,  # already (Country, Year, Indicator, Value)
        disasters_cn,
    ],
    ignore_index=True,
)

# Region flag
data_long["Region"] = np.where(data_long["Country"] == "China", "China", "Rest of the World")

# --------------------------------------------------------------------------------------
# 3) Sidebar controls
# --------------------------------------------------------------------------------------
countries = sorted(co2_long["Country"].dropna().unique().tolist())
default_country = "China" if "China" in countries else (countries[0] if countries else "China")

st.sidebar.header("Controls")
focus_country = st.sidebar.selectbox("Country focus", countries, index=countries.index(default_country) if default_country in countries else 0)

yr_min = int(data_long["Year"].min()) if not data_long.empty else 1900
yr_max = int(data_long["Year"].max()) if not data_long.empty else 2024
year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(max(1900, yr_min), yr_max), step=1)

show_smoothed = st.sidebar.checkbox("Show smoothed lines (rolling)", value=True)
window = st.sidebar.slider("Smoothing window (years)", 3, 11, 5, step=2) if show_smoothed else 1

# Filtered view
df = data_long[(data_long["Year"] >= year_range[0]) & (data_long["Year"] <= year_range[1])].copy()

# --------------------------------------------------------------------------------------
# 4) Layout (tabs)
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Global & Top Emitters", "China Deep-dive", "Per-Capita vs GDP"])

# -------------------------------------------------
# Tab 1: Overview
# -------------------------------------------------
with tab1:
    st.markdown("## Overview")

    colA, colB, colC, colD = st.columns(4)

    # latest year where we have ANY co2 total
    if not co2_long.empty:
        latest_year = int(co2_long["Year"].max())
        world_emis = co2_long[co2_long["Year"] == latest_year]["Value"].sum()
        china_emis = co2_long[(co2_long["Country"] == "China") & (co2_long["Year"] == latest_year)]["Value"].sum()
        china_share = (china_emis / world_emis * 100) if world_emis else np.nan

        colA.metric("Latest year", latest_year)
        colB.metric("Global CO₂ (Mt)", f"{world_emis/1e6:,.1f}")
        colC.metric("China CO₂ (Mt)", f"{china_emis/1e6:,.1f}")
        colD.metric("China share of global", f"{china_share:,.1f}%")

        st.markdown("### Global CO₂ Emissions over time (sum of all countries)")
        global_emissions = co2_long.groupby("Year", as_index=False)["Value"].sum()
        fig = px.line(global_emissions, x="Year", y="Value",
                      labels={"Value": "CO₂ (metric tons)"},
                      title="World CO₂ Emissions per Year")
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CO₂ totals found.")

# -------------------------------------------------
# Tab 2: Global & Top Emitters
# -------------------------------------------------
with tab2:
    st.markdown("## Global & Top Emitters")

    if co2_long.empty:
        st.info("No CO₂ totals available to plot.")
    else:
        # top 12 emitters in latest year
        latest = int(co2_long["Year"].max())
        latest_df = co2_long[co2_long["Year"] == latest].sort_values("Value", ascending=False)
        top_countries = latest_df["Country"].head(12).tolist()

        st.markdown("### Country CO₂ trajectories (highlight focus country)")
        subset = co2_long[co2_long["Country"].isin(top_countries)]
        fig2 = px.line(subset, x="Year", y="Value", color="Country",
                       line_group="Country",
                       title=f"Top 12 CO₂ Emitting Countries (latest year = {latest})",
                       labels={"Value": "CO₂ (metric tons)"})
        for i, d in enumerate(fig2.data):
            if d.name != focus_country:
                fig2.data[i].line.width = 1
                fig2.data[i].opacity = 0.35
            else:
                fig2.data[i].line.width = 3
                fig2.data[i].opacity = 1.0
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Heatmap: Top 10 CO₂ Emission-producing Countries (1900–latest)")
        top10 = latest_df.head(10)["Country"].tolist()
        tile_df = (co2_long[(co2_long["Country"].isin(top10)) & (co2_long["Year"] >= 1900)]
                   .pivot_table(index="Country", columns="Year", values="Value", aggfunc="sum")
                   .fillna(0.0))
        Z = np.log(tile_df.replace({0: np.nan}))
        fig3 = px.imshow(Z, aspect="auto", color_continuous_scale="viridis",
                         labels=dict(color="ln(CO₂)"),
                         title=f"Top 10 CO₂ Emissions – ordered by {latest}")
        st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# Tab 3: China Deep-dive – emissions & temperature & relationships
# -------------------------------------------------
with tab3:
    st.markdown("## China Deep-dive: Emissions, Temperature & Relationships")

    # Build overlap safely
    c_em = co2_long[co2_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value":"Emissions"})
    c_tmp = temperature_cn[["Year", "Value"]].rename(columns={"Value":"Temperature_F"}) if not temperature_cn.empty else pd.DataFrame(columns=["Year","Temperature_F"])

    overlap_years = sorted(set(c_em["Year"]).intersection(set(c_tmp["Year"])))
    # header summary
    st.caption(
        f"Data points — Emissions (CN): {len(c_em)}, "
        f"Temperature (CN): {len(c_tmp)}, "
        f"Overlap in selected range: "
        f"{len([y for y in overlap_years if year_range[0] <= y <= year_range[1]])} years."
    )

    # clamp to chosen range and overlap
    if overlap_years:
        lo_all, hi_all = min(overlap_years), max(overlap_years)
        lo = max(lo_all, year_range[0])
        hi = min(hi_all, year_range[1])
    else:
        lo = hi = None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### CO₂ Emissions (China)")
        if lo is None:
            st.info("No overlapping years between emissions and the selected range.")
        else:
            ser = c_em[(c_em["Year"] >= lo) & (c_em["Year"] <= hi)].set_index("Year").sort_index()
            if ser.empty:
                st.info("No overlapping years between emissions and the selected range.")
            else:
                if show_smoothed and window > 1:
                    ser["Smoothed"] = ser["Emissions"].rolling(window, center=True, min_periods=1).mean()
                st.line_chart(ser)

    with col2:
        st.markdown("### Temperature (°F, China)")
        if lo is None:
            st.info("No overlapping years between temperature and the selected range.")
        else:
            ser = c_tmp[(c_tmp["Year"] >= lo) & (c_tmp["Year"] <= hi)].set_index("Year").sort_index()
            if ser.empty:
                st.info("No overlapping years between temperature and the selected range.")
            else:
                if show_smoothed and window > 1:
                    ser["Smoothed"] = ser["Temperature_F"].rolling(window, center=True, min_periods=1).mean()
                st.line_chart(ser)

    # Scaled scatter (only if we have at least 3 overlapping rows)
    st.markdown("### Scaled Temperature vs CO₂ Emissions (China)")
    if lo is None:
        st.info("Not enough overlapping years to plot the scaled scatter.")
    else:
        merged = pd.merge(
            c_em[(c_em["Year"] >= lo) & (c_em["Year"] <= hi)],
            c_tmp[(c_tmp["Year"] >= lo) & (c_tmp["Year"] <= hi)],
            on="Year", how="inner"
        )
        if len(merged) < 3:
            st.info("Not enough overlapping years to plot the scaled scatter.")
        else:
            # z-score both series
            x = (merged["Emissions"] - merged["Emissions"].mean()) / merged["Emissions"].std(ddof=0)
            y = (merged["Temperature_F"] - merged["Temperature_F"].mean()) / merged["Temperature_F"].std(ddof=0)
            fig_sc = px.scatter(
                x=x, y=y, labels={"x": "Scaled Emissions", "y": "Scaled Temperature"},
                title=f"China CO₂ Emissions vs Temperature (scaled, {lo}–{hi})"
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("### Notes")
    st.markdown(
        """
        - CO₂ totals converted from *thousand tonnes* to *metric tons*.
        - Temperature data for China converted °C → °F; annual mean inferred if needed.
        - GDP growth and Energy use are from World Bank ZIP packages.
        - Plots harmonize year types and use only overlapping years where appropriate.
        """
    )

# -------------------------------------------------
# Tab 4: Per-Capita vs GDP (China)
# -------------------------------------------------
with tab4:
    st.markdown("## Per-Capita Emissions vs GDP per Capita Growth (China)")

    china_gdp = gdp_long[gdp_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "GDP_Growth"})
    china_pc  = co2_pc_long[co2_pc_long["Country"] == "China"][["Year", "Value"]].rename(columns={"Value": "PerCapita"})

    gg = pd.merge(china_gdp, china_pc, on="Year", how="inner")
    gg = gg[(gg["Year"] >= year_range[0]) & (gg["Year"] <= year_range[1])]

    if gg.empty:
        st.info("No overlapping data for the selected range.")
    else:
        # No trendline to avoid extra dependency; just show the scatter
        fig4 = px.scatter(
            gg, x="GDP_Growth", y="PerCapita", color="Year",
            labels={"GDP_Growth": "GDP per Capita Growth (%)",
                    "PerCapita": "CO₂ per Capita (t/person)"},
            title=f"China: CO₂ per Capita vs GDP per Capita Growth ({year_range[0]}–{year_range[1]})"
        )
        st.plotly_chart(fig4, use_container_width=True)

# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
st.caption(
    """
    **Data pipeline:** GitHub raw files → cached HTTP reads → robust parsing → tidy long tables → interactive Plotly charts.
    """
)
