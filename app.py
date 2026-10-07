import streamlit as st
import pandas as pd
import plotly.express as px

st.title("EC2 Instance EDA Dashboard")

df = pd.read_csv("ec2dataset.csv")

# DATASET OVERVIEW
st.header("Dataset Overview")


st.subheader("Dataset Preview")

st.dataframe(df)

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Instances", len(df))

col2.metric("Number of Columns", len(df.columns))

col3.metric("Missing Values", df.isna().sum().sum())


st.subheader("Dataset Structure")

st.write("Columns:")

st.write(df.columns.tolist())

st.subheader("Data Types")

st.write(df.dtypes)


df["Memory_GiB"] = (
    df["Instance Memory"]
    .str.extract(r"([\d.]+)")
    .astype(float)
)


df["vCPU_Count"] = (
    df["vCPUs"]
    .str.extract(r"(\d+)")
    .astype(float)
)


def clean_price(value):

    if pd.isna(value):
        return None

    value = str(value)

    if "unavailable" in value.lower():
        return None

    return float(
        value.replace("$", "")
        .replace(" hourly", "")
        .strip()
    )


price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]


for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)


df["Cost_Per_GiB"] = (
    df["On Demand_USD"] /
    df["Memory_GiB"]
)


df["Monthly_On_Demand"] = df["On Demand_USD"] * 730


df["Memory_per_vCPU"] = (
    df["Memory_GiB"] /
    df["vCPU_Count"]
)

# FILTERS
st.header("Filters")

st.sidebar.header("Filters")

max_memory = float(df["Memory_GiB"].max())

memory_filter = st.sidebar.slider(
    "Maximum Memory (GiB)",
    min_value=0.5,
    max_value=max_memory,
    value=max_memory
)


cpu_values = sorted(
    df["vCPU_Count"].dropna().unique()
)

selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values
)


network_values = sorted(
    df["Network Performance"].dropna().unique()
)

selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_values,
    default=network_values
)


storage_values = df["Instance Storage"].dropna().unique()

selected_storage = st.sidebar.multiselect(
    "Instance Storage",
    options=storage_values,
    default=storage_values
)


max_price = float(df["On Demand_USD"].max())

hourly_price_filter = st.sidebar.slider(
    "Maximum Hourly Price",
    min_value=0.0,
    max_value=max_price,
    value=max_price
)


max_monthly_cost = float(df["Monthly_On_Demand"].max())

monthly_cost_filter = st.sidebar.slider(
    "Maximum Monthly Cost",
    min_value=10.0,
    max_value=max_monthly_cost,
    value=min(500.0, max_monthly_cost)
)


filtered_df = df[
    (df["Memory_GiB"] <= memory_filter) &
    (df["vCPU_Count"].isin(selected_cpu)) &
    (df["Network Performance"].isin(selected_network)) &
    (df["Instance Storage"].isin(selected_storage)) &
    (df["On Demand_USD"] <= hourly_price_filter) &
    (df["Monthly_On_Demand"] <= monthly_cost_filter)
]


st.write(
    f"{len(filtered_df)} instances found"
)

# KEY METRICS
st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Number of Instances",
    len(filtered_df)
)

col2.metric(
    "Average Memory",
    f"{filtered_df['Memory_GiB'].mean():.2f} GiB"
)

col3.metric(
    "Average vCPUs",
    f"{filtered_df['vCPU_Count'].mean():.1f}"
)

col4.metric(
    "Average Cost",
    f"${filtered_df['On Demand_USD'].mean():.4f}"
)

# EDA
st.header("EDA")

st.subheader("Memory Distribution")

fig = px.histogram(
    filtered_df,
    x="Memory_GiB",
    nbins=30,
    title="Distribution of EC2 Memory"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("vCPU Distribution")

fig = px.histogram(
    filtered_df,
    x="vCPU_Count",
    title="Distribution of vCPUs"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Memory vs vCPU")

fig = px.scatter(
    filtered_df,
    x="vCPU_Count",
    y="Memory_GiB",
    hover_name="API Name",
    hover_data=["On Demand_USD"],
    title="EC2 Memory vs vCPU"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Memory vs Cost")

fig = px.scatter(
    filtered_df,
    x="Memory_GiB",
    y="On Demand_USD",
    hover_name="API Name",
    size="vCPU_Count",
    title="Memory vs EC2 On-Demand Cost"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Memory per vCPU")

memory_efficiency = (
    filtered_df
    .sort_values(
        "Memory_per_vCPU",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "Memory_per_vCPU"
        ]
    ]
    .head(15)
)

st.dataframe(memory_efficiency)

# EC2 COST ANALYSIS
st.header("EC2 Cost Analysis")

st.subheader("Cheapest Instances")

cheapest = (
    filtered_df
    .sort_values("On Demand_USD")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(cheapest)


st.subheader("Most Expensive Instances")

most_expensive = (
    filtered_df
    .sort_values(
        "On Demand_USD",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(most_expensive)


st.subheader("Monthly Cost")

st.dataframe(
    filtered_df[
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
)

# PRICING COMPARISON
st.header("Pricing Comparison")

pricing = filtered_df[
    [
        "Name",
        "API Name",
        "On Demand_USD",
        "Linux Reserved cost_USD",
        "Linux Spot Minimum cost_USD",
        "Windows On Demand cost_USD",
        "Windows Reserved cost_USD"
    ]
]

st.dataframe(pricing)


selected_instance = st.selectbox(
    "Select an EC2 Instance",
    filtered_df["API Name"].unique()
)


instance = filtered_df[
    filtered_df["API Name"] == selected_instance
].iloc[0]


pricing_data = pd.DataFrame({
    "Pricing Model": [
        "On Demand",
        "Linux Reserved",
        "Linux Spot"
    ],
    "Hourly Cost": [
        instance["On Demand_USD"],
        instance["Linux Reserved cost_USD"],
        instance["Linux Spot Minimum cost_USD"]
    ]
})


pricing_data = pricing_data.dropna()


fig = px.bar(
    pricing_data,
    x="Pricing Model",
    y="Hourly Cost",
    title=f"Pricing Comparison: {selected_instance}"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


pricing_data["Monthly Cost"] = (
    pricing_data["Hourly Cost"] * 730
)

st.dataframe(pricing_data)

# DATA EXPORT
st.header("Data Export")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_ec2_instances.csv",
    mime="text/csv"
)