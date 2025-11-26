import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Serverless FinOps Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Load external CSS
# -------------------------------------------------------
def load_css(css_file: str = "./styles.css"):
    path = Path(css_file)
    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{css_file}' not found. Using default Streamlit styles.")

load_css()

# -------------------------------------------------------
# Data loading & cleaning
# (handles the weird "entire row in double quotes" CSV)
# -------------------------------------------------------
@st.cache_data
def load_serverless_data(csv_path: str = "Serverless_Data.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        st.error(f"Could not find {csv_path}. Make sure it is in the same folder as app.py.")
        return pd.DataFrame()

    raw_lines = path.read_text().strip().splitlines()
    rows = [line.strip().strip('"') for line in raw_lines if line.strip()]

    header = rows[0].split(",")
    data_rows = [r.split(",") for r in rows[1:]]

    df = pd.DataFrame(data_rows, columns=header)

    numeric_cols = [
        "InvocationsPerMonth",
        "AvgDurationMs",
        "MemoryMB",
        "ColdStartRate",
        "ProvisionedConcurrency",
        "GBSeconds",
        "DataTransferGB",
        "CostUSD",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df = load_serverless_data()
if df.empty:
    st.stop()

# Helper: grouped view per function / environment
def group_functions(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return df_in.copy()
    grouped = (
        df_in.groupby(["FunctionName", "Environment"], as_index=False)
        .agg(
            {
                "CostUSD": "sum",
                "InvocationsPerMonth": "sum",
                "AvgDurationMs": "mean",
                "MemoryMB": "mean",
                "GBSeconds": "sum",
                "DataTransferGB": "sum",
                "ColdStartRate": "mean",
                "ProvisionedConcurrency": "mean",
            }
        )
    )
    return grouped

# -------------------------------------------------------
# Sidebar – Global Filters
# -------------------------------------------------------
st.sidebar.title("Filters")

env_options = sorted(df["Environment"].dropna().unique())
selected_envs = st.sidebar.multiselect(
    "Environment",
    options=env_options,
    default=env_options,
)

min_cost = float(df["CostUSD"].min())
max_cost = float(df["CostUSD"].max())
cost_range = st.sidebar.slider(
    "Monthly Cost (USD)",
    min_value=round(min_cost, 2),
    max_value=round(max_cost, 2),
    value=(round(min_cost, 2), round(max_cost, 2)),
)

min_inv = int(df["InvocationsPerMonth"].min())
max_inv = int(df["InvocationsPerMonth"].max())
inv_range = st.sidebar.slider(
    "Invocations per Month",
    min_value=min_inv,
    max_value=max_inv,
    value=(min_inv, max_inv),
    step=max(1, (max_inv - min_inv) // 20),
)

name_filter = st.sidebar.text_input(
    "Search Function Name (contains)",
    value="",
)

filtered_df = df.copy()
if selected_envs:
    filtered_df = filtered_df[filtered_df["Environment"].isin(selected_envs)]

filtered_df = filtered_df[
    (filtered_df["CostUSD"].between(cost_range[0], cost_range[1]))
    & (filtered_df["InvocationsPerMonth"].between(inv_range[0], inv_range[1]))
]

if name_filter.strip():
    filtered_df = filtered_df[
        filtered_df["FunctionName"].str.contains(name_filter.strip(), case=False, na=False)
    ]

grouped_df = group_functions(filtered_df)

# -------------------------------------------------------
# Layout – Tabs
# -------------------------------------------------------
st.title("FinOps at Scale for Serverless Applications")

tabs = st.tabs(
    [
        "Overview",
        "Exercise 1 – Top Cost Contributors",
        "Exercise 2 – Memory Right-Sizing",
        "Exercise 3 – Provisioned Concurrency",
        "Exercise 4 – Unused / Low-Value Workloads",
        "Exercise 5 – Cost Forecasting",
        "Exercise 6 – Containerization Candidates",
    ]
)

# Common Altair config (bigger fonts + colour palette)
def configure_chart(chart):
    return (
        chart
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            gridColor="#e5e7eb",
        )
        .configure_legend(
            labelFontSize=12,
            titleFontSize=14,
        )
        .configure_title(
            fontSize=16,
            color="#0f172a",
        )
        .configure_view(
            strokeOpacity=0,
            fill="#ffffff",
        )
        .configure_range(
            category=[
                "#6366F1",  # indigo
                "#EC4899",  # pink
                "#10B981",  # emerald
                "#F97316",  # orange
                "#0EA5E9",  # sky
                "#A855F7",  # purple
                "#FACC15",  # amber
            ]
        )
    )

# =======================================================
# Overview Tab
# =======================================================
with tabs[0]:
    st.subheader("Overview (Filtered)")

    total_functions = filtered_df["FunctionName"].nunique()
    total_cost = filtered_df["CostUSD"].sum()
    total_invocations = filtered_df["InvocationsPerMonth"].sum()
    total_gb_seconds = filtered_df["GBSeconds"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Functions", total_functions)
    col2.metric("Total Monthly Cost (USD)", f"${total_cost:,.2f}")
    col3.metric("Total Invocations / Month", f"{total_invocations:,}")
    col4.metric("Total GB-Seconds", f"{total_gb_seconds:,.2f}")

    st.markdown("---")

    # Cost by environment – bar chart
    st.markdown("#### Cost by Environment")
    if not filtered_df.empty:
        env_cost = (
            filtered_df.groupby("Environment", as_index=False)["CostUSD"]
            .sum()
            .sort_values("CostUSD", ascending=False)
        )

        chart_env = alt.Chart(env_cost, title="Monthly Cost by Environment").mark_bar().encode(
            x=alt.X("Environment:N", title="Environment"),
            y=alt.Y("CostUSD:Q", title="Cost (USD)"),
            color=alt.Color("Environment:N", legend=None),
            tooltip=["Environment", alt.Tooltip("CostUSD:Q", format=",.2f")],
        ).properties(
            width="container",
            height=400,
        )

        st.altair_chart(configure_chart(chart_env), use_container_width=True)
    else:
        st.info("No data matches the current filters.")

    # Top 10 functions – table + matching bar chart
    st.markdown("#### Top 10 Functions by Cost (Filtered)")
    if not filtered_df.empty:
        top10 = (
            filtered_df.sort_values("CostUSD", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        top10_chart_data = top10.copy()
        top10_chart_data["Function (Env)"] = (
            top10_chart_data["FunctionName"] + " (" + top10_chart_data["Environment"] + ")"
        )

        chart_top10 = alt.Chart(
            top10_chart_data,
            title="Top 10 Functions by Cost",
        ).mark_bar().encode(
            x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
            y=alt.Y("CostUSD:Q", title="Cost (USD)"),
            color=alt.Color("Environment:N"),
            tooltip=[
                "FunctionName",
                "Environment",
                alt.Tooltip("CostUSD:Q", format=",.2f"),
                alt.Tooltip("InvocationsPerMonth:Q", format=","),
            ],
        ).properties(
            width="container",
            height=360,
        ).interactive()

        st.altair_chart(configure_chart(chart_top10), use_container_width=True)

        st.dataframe(
            top10[
                [
                    "FunctionName",
                    "Environment",
                    "InvocationsPerMonth",
                    "AvgDurationMs",
                    "MemoryMB",
                    "CostUSD",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No data to display. Try expanding your filters.")

    # Raw data expander + mini histogram explorer
    with st.expander("Show raw filtered data"):
        st.dataframe(filtered_df, use_container_width=True)

        numeric_cols = [
            c for c in filtered_df.columns
            if pd.api.types.is_numeric_dtype(filtered_df[c])
        ]
        if numeric_cols:
            st.markdown("##### Distribution Explorer")
            col_name = st.selectbox("Pick a numeric field", numeric_cols, index=0)
            hist = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X(f"{col_name}:Q", bin=alt.Bin(maxbins=30), title=col_name),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip(f"{col_name}:Q", format=",.2f"), "count():Q"],
            ).properties(
                width="container",
                height=260,
                title=f"Distribution of {col_name}",
            )
            st.altair_chart(configure_chart(hist), use_container_width=True)

# =======================================================
# Exercise 1 – Top Cost Contributors
# =======================================================
with tabs[1]:
    st.subheader("Exercise 1 – Identify Top Cost Contributors")

    if grouped_df.empty or grouped_df["CostUSD"].sum() == 0:
        st.warning("No cost data available for the current filters. Adjust filters in the sidebar.")
    else:
        st.markdown(
            """
This analysis shows which functions contribute **most of the total monthly cost**.
We group by **FunctionName + Environment**, sort by cost, and compute the **cumulative cost share** (Pareto-style).
            """
        )

        grouped = grouped_df.sort_values("CostUSD", ascending=False).copy()
        total_cost_ex1 = grouped["CostUSD"].sum()

        grouped["CostSharePct"] = (grouped["CostUSD"] / total_cost_ex1) * 100
        grouped["CumCostSharePct"] = grouped["CostSharePct"].cumsum()

        top80 = grouped[grouped["CumCostSharePct"] <= 80].copy()
        if top80.empty:
            top80 = grouped.head(1).copy()

        num_high_cost = len(top80)
        high_cost_total = top80["CostUSD"].sum()
        high_cost_pct = (high_cost_total / total_cost_ex1) * 100

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Functions (grouped)", len(grouped))
        col_b.metric("High-Cost Functions (≈80% of spend)", num_high_cost)
        col_c.metric(
            "Cost Covered by High-Cost Group",
            f"{high_cost_pct:,.1f}%",
            delta=f"${high_cost_total:,.2f}",
        )

        st.markdown("---")

        st.markdown("#### Top Functions by Monthly Cost")
        top_n = st.slider("How many functions to show in bar chart?", 5, 40, 20)

        chart_data = grouped.head(top_n).copy()
        chart_data["Function (Env)"] = (
            chart_data["FunctionName"] + " (" + chart_data["Environment"] + ")"
        )

        bar_top_cost = alt.Chart(
            chart_data,
            title="Top Functions by Monthly Cost",
        ).mark_bar().encode(
            x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
            y=alt.Y("CostUSD:Q", title="Cost (USD)"),
            color=alt.Color("Environment:N"),
            tooltip=[
                "FunctionName",
                "Environment",
                alt.Tooltip("CostUSD:Q", format=",.2f"),
                alt.Tooltip("InvocationsPerMonth:Q", format=","),
                "AvgDurationMs",
                "MemoryMB",
            ],
        ).properties(
            width="container",
            height=420,
        ).interactive()

        st.altair_chart(configure_chart(bar_top_cost), use_container_width=True)

        st.markdown("#### Pareto View – Cumulative Cost Share")
        pareto_data = chart_data.copy()

        base = alt.Chart(pareto_data).properties(
            width="container",
            height=420,
            title="Pareto Chart of Cost Contribution",
        )

        bars = base.mark_bar(color="#6366F1").encode(
            x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
            y=alt.Y("CostUSD:Q", title="Cost (USD)"),
            tooltip=[
                "FunctionName",
                "Environment",
                alt.Tooltip("CostUSD:Q", format=",.2f"),
                alt.Tooltip("CumCostSharePct:Q", format=".1f"),
            ],
        )

        line = base.mark_line(point=True, color="#EC4899").encode(
            x=alt.X("Function (Env):N", sort="-y"),
            y=alt.Y(
                "CumCostSharePct:Q",
                axis=alt.Axis(title="Cumulative Cost Share (%)", titleColor="#EC4899"),
            ),
        )

        pareto_chart = configure_chart(bars + line)
        st.altair_chart(pareto_chart, use_container_width=True)

        st.markdown("#### Metric Explorer (Bar Chart by Function)")
        st.caption("Pick any metric to visualize for the top functions (after filters).")

        metric_options = {
            "Cost (USD)": "CostUSD",
            "Invocations per Month": "InvocationsPerMonth",
            "Average Duration (ms)": "AvgDurationMs",
            "Memory (MB)": "MemoryMB",
            "GB-Seconds": "GBSeconds",
            "Data Transfer (GB)": "DataTransferGB",
            "Cold Start Rate": "ColdStartRate",
            "Provisioned Concurrency": "ProvisionedConcurrency",
        }

        selected_metric_label = st.selectbox(
            "Metric to plot:",
            list(metric_options.keys()),
            index=0,
        )
        selected_metric = metric_options[selected_metric_label]

        metric_chart_data = grouped.head(top_n).copy()
        metric_chart_data["Function (Env)"] = (
            metric_chart_data["FunctionName"] + " (" + metric_chart_data["Environment"] + ")"
        )

        metric_bar = alt.Chart(
            metric_chart_data,
            title=f"{selected_metric_label} for Top Functions",
        ).mark_bar().encode(
            x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
            y=alt.Y(f"{selected_metric}:Q", title=selected_metric_label),
            color=alt.Color("Environment:N"),
            tooltip=[
                "FunctionName",
                "Environment",
                alt.Tooltip(f"{selected_metric}:Q", format=",.4g"),
            ],
        ).properties(
            width="container",
            height=420,
        ).interactive()

        st.altair_chart(configure_chart(metric_bar), use_container_width=True)

        st.markdown("#### Cost vs. Invocations (All Filtered Functions)")
        st.caption("Helps spot expensive functions with low invocation volume (potential optimization targets).")

        scatter_df = grouped.copy()
        scatter_df["Function (Env)"] = (
            scatter_df["FunctionName"] + " (" + scatter_df["Environment"] + ")"
        )

        scatter = alt.Chart(
            scatter_df,
            title="Cost vs Invocations",
        ).mark_circle(size=80).encode(
            x=alt.X("InvocationsPerMonth:Q", title="Invocations per Month"),
            y=alt.Y("CostUSD:Q", title="Cost (USD)"),
            color=alt.Color("Environment:N"),
            tooltip=[
                "FunctionName",
                "Environment",
                alt.Tooltip("CostUSD:Q", format=",.2f"),
                alt.Tooltip("InvocationsPerMonth:Q", format=","),
                "AvgDurationMs",
                "MemoryMB",
            ],
        ).properties(
            width="container",
            height=420,
        ).interactive()

        st.altair_chart(configure_chart(scatter), use_container_width=True)

        st.markdown("#### Functions Contributing ~80% of Total Monthly Cost")
        display_cols = [
            "FunctionName",
            "Environment",
            "InvocationsPerMonth",
            "AvgDurationMs",
            "MemoryMB",
            "GBSeconds",
            "DataTransferGB",
            "ColdStartRate",
            "ProvisionedConcurrency",
            "CostUSD",
            "CostSharePct",
            "CumCostSharePct",
        ]
        st.dataframe(
            top80[display_cols].reset_index(drop=True),
            use_container_width=True,
        )

        with st.expander("Show full grouped data for Exercise 1"):
            st.dataframe(grouped[display_cols], use_container_width=True)

# =======================================================
# Exercise 2 – Memory Right-Sizing
# =======================================================
with tabs[2]:
    st.subheader("Exercise 2 – Memory Right-Sizing")

    if grouped_df.empty:
        st.warning("No data available for the current filters.")
    else:
        st.markdown(
            """
Goal: **Find functions where duration is low but memory is high**, and estimate
the cost impact of lowering memory.
            """
        )

        low_duration_ms = st.slider(
            "Max average duration for 'low-latency' functions (ms)",
            min_value=0,
            max_value=int(grouped_df["AvgDurationMs"].max()),
            value=min(1000, int(grouped_df["AvgDurationMs"].max())),
            step=50,
        )
        high_memory_mb = st.slider(
            "Min memory to consider 'high' (MB)",
            min_value=128,
            max_value=int(grouped_df["MemoryMB"].max()),
            value=min(2048, int(grouped_df["MemoryMB"].max())),
            step=64,
        )

        target_memory_mb = st.slider(
            "Assumed new memory setting for candidates (MB)",
            min_value=128,
            max_value=int(high_memory_mb),
            value=min(1024, int(high_memory_mb)),
            step=64,
            help="We approximate cost scaling linearly with memory size.",
        )

        candidates = grouped_df[
            (grouped_df["AvgDurationMs"] <= low_duration_ms)
            & (grouped_df["MemoryMB"] >= high_memory_mb)
        ].copy()

        if candidates.empty:
            st.info("No functions meet the 'low duration, high memory' criteria with current thresholds.")
        else:
            candidates["NewMemoryMB"] = target_memory_mb
            candidates["SavingsUSD"] = candidates["CostUSD"] * (
                1 - (target_memory_mb / candidates["MemoryMB"].replace(0, np.nan))
            )
            candidates["SavingsUSD"] = candidates["SavingsUSD"].clip(lower=0).fillna(0)
            candidates["NewEstimatedCostUSD"] = candidates["CostUSD"] - candidates["SavingsUSD"]
            candidates["SavingsPct"] = (
                candidates["SavingsUSD"] / candidates["CostUSD"].replace(0, np.nan) * 100
            ).fillna(0)

            total_savings = candidates["SavingsUSD"].sum()
            avg_savings_pct = candidates["SavingsPct"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Candidate Functions", len(candidates))
            col2.metric("Estimated Monthly Savings (USD)", f"${total_savings:,.2f}")
            col3.metric("Average Savings per Function (%)", f"{avg_savings_pct:,.1f}%")

            st.markdown("---")
            st.markdown("#### Potential Savings by Function (Top 20)")
            top_savings = candidates.sort_values("SavingsUSD", ascending=False).head(20).copy()
            top_savings["Function (Env)"] = (
                top_savings["FunctionName"] + " (" + top_savings["Environment"] + ")"
            )

            chart_savings = alt.Chart(
                top_savings,
                title="Estimated Savings by Right-Sizing Memory",
            ).mark_bar().encode(
                x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
                y=alt.Y("SavingsUSD:Q", title="Estimated Savings (USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    alt.Tooltip("CostUSD:Q", title="Current Cost", format=",.2f"),
                    alt.Tooltip("NewEstimatedCostUSD:Q", title="New Est. Cost", format=",.2f"),
                    alt.Tooltip("SavingsUSD:Q", format=",.2f"),
                    alt.Tooltip("SavingsPct:Q", format=".1f"),
                    "MemoryMB",
                    "NewMemoryMB",
                    "AvgDurationMs",
                ],
            ).properties(
                width="container",
                height=420,
            ).interactive()

            st.altair_chart(configure_chart(chart_savings), use_container_width=True)

            st.markdown("#### Duration vs Memory (Highlighting Right-Sizing Candidates)")
            all_for_plot = grouped_df.copy()
            all_for_plot["Candidate"] = "No"
            all_for_plot.loc[candidates.index, "Candidate"] = "Yes"

            scatter_mem = alt.Chart(
                all_for_plot,
                title="Avg Duration vs Memory",
            ).mark_circle(size=70).encode(
                x=alt.X("AvgDurationMs:Q", title="Avg Duration (ms)"),
                y=alt.Y("MemoryMB:Q", title="Memory (MB)"),
                color=alt.Color("Candidate:N", scale=alt.Scale(range=["#CBD5F5", "#EC4899"])),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    "AvgDurationMs",
                    "MemoryMB",
                    alt.Tooltip("CostUSD:Q", format=",.2f"),
                ],
            ).properties(
                width="container",
                height=420,
            ).interactive()

            st.altair_chart(configure_chart(scatter_mem), use_container_width=True)

            st.markdown("#### Detailed Table for Memory Right-Sizing Candidates")
            show_cols = [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "AvgDurationMs",
                "MemoryMB",
                "NewMemoryMB",
                "CostUSD",
                "NewEstimatedCostUSD",
                "SavingsUSD",
                "SavingsPct",
            ]
            st.dataframe(candidates[show_cols].reset_index(drop=True), use_container_width=True)

# =======================================================
# Exercise 3 – Provisioned Concurrency Optimization
# =======================================================
with tabs[3]:
    st.subheader("Exercise 3 – Provisioned Concurrency Optimization")

    pc_df = grouped_df[grouped_df["ProvisionedConcurrency"] > 0].copy()
    if pc_df.empty:
        st.info("No functions have Provisioned Concurrency enabled under current filters.")
    else:
        st.markdown(
            """
Goal: Compare **cold start rate vs Provisioned Concurrency** and flag functions
where PC might be overused.
            """
        )

        cold_low = st.slider(
            "Cold start rate threshold for 'consider removing PC'",
            min_value=0.0,
            max_value=float(pc_df["ColdStartRate"].max()),
            value=min(0.02, float(pc_df["ColdStartRate"].max())),
            step=0.005,
        )
        cold_high = st.slider(
            "Cold start rate threshold for 'consider reducing PC'",
            min_value=cold_low,
            max_value=float(pc_df["ColdStartRate"].max()) if pc_df["ColdStartRate"].max() > cold_low else cold_low + 0.01,
            value=min(0.10, max(0.10, float(pc_df["ColdStartRate"].max()))),
            step=0.01,
        )

        pc_df["PCWeight"] = pc_df["ProvisionedConcurrency"] * pc_df["AvgDurationMs"] * pc_df["MemoryMB"]

        def classify_row(row):
            if row["ColdStartRate"] < cold_low:
                return "Consider removing PC"
            elif row["ColdStartRate"] < cold_high:
                return "Consider reducing PC"
            else:
                return "PC likely needed"

        pc_df["PCAction"] = pc_df.apply(classify_row, axis=1)

        summary = pc_df["PCAction"].value_counts().to_dict()

        col1, col2, col3 = st.columns(3)
        col1.metric("Functions with PC", len(pc_df))
        col2.metric("Consider removing PC", summary.get("Consider removing PC", 0))
        col3.metric("Consider reducing PC", summary.get("Consider reducing PC", 0))

        st.markdown("---")
        st.markdown("#### Cold Start Rate vs Monthly Cost (Size = PC Level)")

        pc_df["Function (Env)"] = pc_df["FunctionName"] + " (" + pc_df["Environment"] + ")"

        scatter_pc = alt.Chart(
            pc_df,
            title="Cold Start Rate vs Cost (with PC)",
        ).mark_circle().encode(
            x=alt.X("ColdStartRate:Q", title="Cold Start Rate"),
            y=alt.Y("CostUSD:Q", title="Cost (USD)"),
            size=alt.Size("ProvisionedConcurrency:Q", title="Provisioned Concurrency"),
            color=alt.Color("PCAction:N", title="Suggested Action"),
            tooltip=[
                "FunctionName",
                "Environment",
                "ColdStartRate",
                "ProvisionedConcurrency",
                alt.Tooltip("CostUSD:Q", format=",.2f"),
            ],
        ).properties(
            width="container",
            height=420,
        ).interactive()

        st.altair_chart(configure_chart(scatter_pc), use_container_width=True)

        # NEW: bar chart summarizing PCAction counts (graph for the table)
        st.markdown("#### Count of Functions by Suggested PC Action")

        action_counts = (
            pc_df.groupby("PCAction", as_index=False)["FunctionName"].count()
            .rename(columns={"FunctionName": "Count"})
        )

        chart_actions = alt.Chart(
            action_counts,
            title="Provisioned Concurrency Actions",
        ).mark_bar().encode(
            x=alt.X("PCAction:N", title="Suggested Action"),
            y=alt.Y("Count:Q", title="Number of Functions"),
            color=alt.Color("PCAction:N", legend=None),
            tooltip=["PCAction", "Count"],
        ).properties(
            width="container",
            height=320,
        )

        st.altair_chart(configure_chart(chart_actions), use_container_width=True)

        st.markdown("#### Detailed PC Optimization Table")
        show_cols = [
            "FunctionName",
            "Environment",
            "InvocationsPerMonth",
            "AvgDurationMs",
            "MemoryMB",
            "CostUSD",
            "ColdStartRate",
            "ProvisionedConcurrency",
            "PCAction",
        ]
        st.dataframe(pc_df[show_cols].reset_index(drop=True), use_container_width=True)

# =======================================================
# Exercise 4 – Detect Unused / Low-Value Workloads
# =======================================================
with tabs[4]:
    st.subheader("Exercise 4 – Detect Unused or Low-Value Workloads")

    if grouped_df.empty:
        st.warning("No data available for the current filters.")
    else:
        st.markdown(
            """
Goal: Find functions that have **very low invocation share** but still
incur **relatively high cost**.
            """
        )

        total_inv = grouped_df["InvocationsPerMonth"].sum()
        grouped_low = grouped_df.copy()
        grouped_low["InvocationSharePct"] = (
            grouped_low["InvocationsPerMonth"] / total_inv * 100 if total_inv > 0 else 0
        )

        inv_share_threshold = st.slider(
            "Max invocation share (%) to be considered 'low usage'",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )

        cost_threshold = st.slider(
            "Min monthly cost (USD) for a low-usage function to be concerning",
            min_value=float(grouped_low["CostUSD"].min()),
            max_value=float(grouped_low["CostUSD"].max()),
            value=min(50.0, float(grouped_low["CostUSD"].max())),
        )

        candidates = grouped_low[
            (grouped_low["InvocationSharePct"] < inv_share_threshold)
            & (grouped_low["CostUSD"] >= cost_threshold)
        ].copy()

        if candidates.empty:
            st.info("No low-usage, high-cost functions with the current thresholds.")
        else:
            total_candidate_cost = candidates["CostUSD"].sum()
            candidate_count = len(candidates)

            col1, col2 = st.columns(2)
            col1.metric("Low-Usage / High-Cost Functions", candidate_count)
            col2.metric("Total Monthly Cost for These Functions", f"${total_candidate_cost:,.2f}")

            st.markdown("---")
            st.markdown("#### Low-Usage / High-Cost Functions")

            candidates["Function (Env)"] = (
                candidates["FunctionName"] + " (" + candidates["Environment"] + ")"
            )

            chart_low = alt.Chart(
                candidates.sort_values("CostUSD", ascending=False),
                title="Low-Usage Functions with High Cost",
            ).mark_bar().encode(
                x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
                y=alt.Y("CostUSD:Q", title="Cost (USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    alt.Tooltip("CostUSD:Q", format=",.2f"),
                    alt.Tooltip("InvocationsPerMonth:Q", format=","),
                    alt.Tooltip("InvocationSharePct:Q", format=".2f"),
                ],
            ).properties(
                width="container",
                height=420,
            ).interactive()

            st.altair_chart(configure_chart(chart_low), use_container_width=True)

            st.markdown("#### Invocation Share vs Cost (All Functions)")
            scatter_low = alt.Chart(
                grouped_low,
                title="Invocation Share vs Cost",
            ).mark_circle(size=70).encode(
                x=alt.X("InvocationSharePct:Q", title="Invocation Share (%)"),
                y=alt.Y("CostUSD:Q", title="Cost (USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    alt.Tooltip("InvocationsPerMonth:Q", format=","),
                    alt.Tooltip("InvocationSharePct:Q", format=".2f"),
                    alt.Tooltip("CostUSD:Q", format=",.2f"),
                ],
            ).properties(
                width="container",
                height=420,
            ).interactive()

            st.altair_chart(configure_chart(scatter_low), use_container_width=True)

            st.markdown("#### Detailed Table – Low-Value Candidates")
            show_cols = [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "InvocationSharePct",
                "AvgDurationMs",
                "MemoryMB",
                "CostUSD",
            ]
            st.dataframe(candidates[show_cols].reset_index(drop=True), use_container_width=True)

# =======================================================
# Exercise 5 – Cost Forecasting
# =======================================================
with tabs[5]:
    st.subheader("Exercise 5 – Cost Forecasting")

    if grouped_df.empty:
        st.warning("No data available for the current filters.")
    else:
        st.markdown(
            """
We approximate cost using a simple model:  

**Cost ≈ α × (Invocations × Duration × Memory) + β × (DataTransferGB)**  

Simple but useful for **ballpark forecasts** and for spotting **outliers**.
            """
        )

        X = pd.DataFrame()
        X["WorkloadFactor"] = (
            grouped_df["InvocationsPerMonth"]
            * grouped_df["AvgDurationMs"]
            * grouped_df["MemoryMB"]
        ) / 1e9
        X["DataTransferGB"] = grouped_df["DataTransferGB"]
        y = grouped_df["CostUSD"]

        mask = np.isfinite(X["WorkloadFactor"]) & np.isfinite(X["DataTransferGB"]) & np.isfinite(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 3:
            st.warning("Not enough clean data points to build a forecasting model.")
        else:
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            y_pred = model.predict(X_clean)

            alpha = model.coef_[0]
            beta = model.coef_[1]
            r2 = model.score(X_clean, y_clean)

            col1, col2, col3 = st.columns(3)
            col1.metric("α (WorkloadFactor coefficient)", f"{alpha:,.4f}")
            col2.metric("β (DataTransferGB coefficient)", f"{beta:,.4f}")
            col3.metric("R² (goodness of fit)", f"{r2:,.3f}")

            st.markdown("---")
            st.markdown("#### Actual vs Predicted Cost")

            result_df = grouped_df.loc[mask].copy()
            result_df["PredictedCost"] = y_pred
            result_df["Residual"] = result_df["CostUSD"] - result_df["PredictedCost"]
            result_df["Function (Env)"] = (
                result_df["FunctionName"] + " (" + result_df["Environment"] + ")"
            )

            scatter_fit = alt.Chart(
                result_df,
                title="Actual vs Predicted Cost",
            ).mark_circle(size=70).encode(
                x=alt.X("PredictedCost:Q", title="Predicted Cost (USD)"),
                y=alt.Y("CostUSD:Q", title="Actual Cost (USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    alt.Tooltip("CostUSD:Q", format=",.2f"),
                    alt.Tooltip("PredictedCost:Q", format=",.2f"),
                    alt.Tooltip("Residual:Q", format=",.2f"),
                ],
            ).properties(
                width="container",
                height=420,
            ).interactive()

            st.altair_chart(configure_chart(scatter_fit), use_container_width=True)

            st.markdown("#### Functions with Highest Positive Residual (Actual > Predicted)")
            over_cost = result_df.sort_values("Residual", ascending=False).head(15)

            # NEW: residual bar chart for this table
            over_cost_chart = over_cost.copy()
            over_cost_chart["Function (Env)"] = (
                over_cost_chart["FunctionName"] + " (" + over_cost_chart["Environment"] + ")"
            )

            chart_resid = alt.Chart(
                over_cost_chart,
                title="Top Residuals (Actual Cost Above Model)",
            ).mark_bar().encode(
                x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
                y=alt.Y("Residual:Q", title="Residual (Actual - Predicted, USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    alt.Tooltip("Residual:Q", format=",.2f"),
                    alt.Tooltip("CostUSD:Q", format=",.2f"),
                    alt.Tooltip("PredictedCost:Q", format=",.2f"),
                ],
            ).properties(
                width="container",
                height=380,
            ).interactive()

            st.altair_chart(configure_chart(chart_resid), use_container_width=True)

            show_cols = [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "AvgDurationMs",
                "MemoryMB",
                "DataTransferGB",
                "CostUSD",
                "PredictedCost",
                "Residual",
            ]
            st.dataframe(over_cost[show_cols].reset_index(drop=True), use_container_width=True)

            st.markdown("---")
            st.markdown("#### What-If Cost Calculator")

            col_l, col_r = st.columns(2)
            with col_l:
                w_inv = st.number_input(
                    "Invocations per Month",
                    min_value=0,
                    value=int(grouped_df["InvocationsPerMonth"].median()),
                    step=1000,
                )
                w_dur = st.number_input(
                    "Average Duration (ms)",
                    min_value=1,
                    value=int(grouped_df["AvgDurationMs"].median()),
                    step=10,
                )
                w_mem = st.number_input(
                    "Memory (MB)",
                    min_value=128,
                    value=int(grouped_df["MemoryMB"].median()),
                    step=64,
                )
                w_xfer = st.number_input(
                    "Data Transfer (GB)",
                    min_value=0.0,
                    value=float(grouped_df["DataTransferGB"].median()),
                    step=1.0,
                )

            with col_r:
                workload_factor = (w_inv * w_dur * w_mem) / 1e9
                predicted_cost_whatif = model.predict([[workload_factor, w_xfer]])[0]
                st.metric("Forecasted Monthly Cost (USD)", f"${predicted_cost_whatif:,.2f}")
                st.caption("Based on the fitted linear model using current dataset.")

# =======================================================
# Exercise 6 – Containerization Candidates
# =======================================================
with tabs[6]:
    st.subheader("Exercise 6 – Containerization Candidates")

    if grouped_df.empty:
        st.warning("No data available for the current filters.")
    else:
        st.markdown(
            """
Goal: Spot workloads that might be better in **containers**:  

- Long-running (> 3 seconds)  
- High memory (> 2 GB)  
- Low invocation frequency  
            """
        )

        dur_threshold = st.slider(
            "Min average duration (ms) to consider 'long-running'",
            min_value=0,
            max_value=int(grouped_df["AvgDurationMs"].max()),
            value=3000,
            step=100,
        )
        mem_threshold = st.slider(
            "Min memory (MB) to consider 'high-memory'",
            min_value=128,
            max_value=int(grouped_df["MemoryMB"].max()),
            value=2048,
            step=64,
        )
        inv_threshold = st.slider(
            "Max invocations per month to consider 'low traffic'",
            min_value=0,
            max_value=int(grouped_df["InvocationsPerMonth"].max()),
            value=int(grouped_df["InvocationsPerMonth"].median()),
            step=1000,
        )

        cand = grouped_df.copy()
        cand["LongRunning"] = cand["AvgDurationMs"] >= dur_threshold
        cand["HighMemory"] = cand["MemoryMB"] >= mem_threshold
        cand["LowTraffic"] = cand["InvocationsPerMonth"] <= inv_threshold
        cand["CriteriaMet"] = cand[["LongRunning", "HighMemory", "LowTraffic"]].sum(axis=1)

        container_candidates = cand[cand["CriteriaMet"] == 3].copy()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Functions", len(cand))
        col2.metric("Meet All 3 Criteria", len(container_candidates))
        col3.metric(
            "Total Cost of Container Candidates",
            f"${container_candidates['CostUSD'].sum():,.2f}",
        )

        st.markdown("---")
        st.markdown("#### Duration vs Memory (Color = Criteria Met Count)")

        cand["CriteriaMetStr"] = cand["CriteriaMet"].astype(str) + " criteria"
        scatter_cont = alt.Chart(
            cand,
            title="Duration vs Memory – Containerization Signals",
        ).mark_circle(size=70).encode(
            x=alt.X("AvgDurationMs:Q", title="Avg Duration (ms)"),
            y=alt.Y("MemoryMB:Q", title="Memory (MB)"),
            color=alt.Color("CriteriaMetStr:N", title="Criteria Met"),
            tooltip=[
                "FunctionName",
                "Environment",
                "AvgDurationMs",
                "MemoryMB",
                "InvocationsPerMonth",
                alt.Tooltip("CostUSD:Q", format=",.2f"),
                "CriteriaMetStr",
            ],
        ).properties(
            width="container",
            height=420,
        ).interactive()

        st.altair_chart(configure_chart(scatter_cont), use_container_width=True)

        st.markdown("#### Functions Meeting All 3 Containerization Criteria")
        if container_candidates.empty:
            st.info("No functions currently meet all 3 containerization criteria.")
        else:
            container_candidates["Function (Env)"] = (
                container_candidates["FunctionName"]
                + " ("
                + container_candidates["Environment"]
                + ")"
            )

            chart_cont_cost = alt.Chart(
                container_candidates.sort_values("CostUSD", ascending=False),
                title="Containerization Candidates by Cost",
            ).mark_bar().encode(
                x=alt.X("Function (Env):N", sort="-y", title="Function (Environment)"),
                y=alt.Y("CostUSD:Q", title="Cost (USD)"),
                color=alt.Color("Environment:N"),
                tooltip=[
                    "FunctionName",
                    "Environment",
                    "AvgDurationMs",
                    "MemoryMB",
                    "InvocationsPerMonth",
                    alt.Tooltip("CostUSD:Q", format=",.2f"),
                ],
            ).properties(
                width="container",
                height=420,
            ).interactive()

            st.altair_chart(configure_chart(chart_cont_cost), use_container_width=True)

            show_cols = [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "AvgDurationMs",
                "MemoryMB",
                "CostUSD",
            ]
            st.dataframe(container_candidates[show_cols].reset_index(drop=True), use_container_width=True)
