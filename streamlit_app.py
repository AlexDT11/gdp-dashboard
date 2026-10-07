import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("EC2 Instance Cost Analysis")

# LOAD DATA
file_path = "ec2dataset.csv"
data = pd.read_csv(file_path)

st.header("Dataset Overview")

st.subheader("Dataset Information")
st.write(data.info())

st.subheader("First 5 Rows")
st.dataframe(data.head())

# COST DATA CLEANING
cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace("[$, hourly]", "", regex=True),
        errors="coerce"
    )

st.header("Cost Analysis")

st.subheader("Missing Values")
st.dataframe(data[cost_columns].isnull().sum())

cost_summary = data[cost_columns].describe()

st.subheader("Cost Summary")
st.dataframe(cost_summary)

# OVERALL COST BOXPLOT
st.subheader("EC2 Cost Comparison")

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=data[cost_columns],
    palette="Set2",
    ax=ax
)

ax.set_title(
    "Cost Comparison of Amazon EC2 Instances (Hourly)",
    fontsize=16
)
ax.set_ylabel("Cost (USD)", fontsize=12)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

st.pyplot(fig)

# OUTLIER DETECTION
st.header("Outlier Detection")

def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound)
        | (data[column] > upper_bound)
    ]


outliers_on_demand = detect_outliers("On Demand")

st.subheader("On-Demand Cost Outliers")
st.dataframe(outliers_on_demand)

# COST COMPARISON
st.subheader("Lowest On-Demand Instances")

cost_comparison = (
    data[
        [
            "Name",
            "On Demand",
            "Linux Reserved cost"
        ]
    ]
    .dropna()
    .sort_values("On Demand")
)

st.dataframe(cost_comparison.head(10))

# INSTANCE FAMILY ANALYSIS
st.header("T2 vs T3 Instance Analysis")

def filter_instance_family(family):
    return data[
        data["Name"].str.startswith(family)
    ]


t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")

t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

st.subheader("T2 Instance Cost Summary")
st.dataframe(t2_summary)

st.subheader("T3 Instance Cost Summary")
st.dataframe(t3_summary)

# T2 BOXPLOT
st.subheader("T2 Instance Cost Distribution")

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=t2_instances[cost_columns],
    palette="Blues",
    showmeans=True,
    ax=ax
)

ax.set_title(
    "Cost Distribution for T2 Instances",
    fontsize=16
)
ax.set_ylabel("Cost (USD)", fontsize=12)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

st.pyplot(fig)

# T3 BOXPLOT
st.subheader("T3 Instance Cost Distribution")

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=t3_instances[cost_columns],
    palette="Greens",
    showmeans=True,
    ax=ax
)

ax.set_title(
    "Cost Distribution for T3 Instances",
    fontsize=16
)
ax.set_ylabel("Cost (USD)", fontsize=12)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

st.pyplot(fig)

# T2/T3 COMPARISON
st.subheader("T2 and T3 Cost Comparison")

comparison = pd.concat(
    [
        t2_instances[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ],
        t3_instances[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ]
    ]
)

comparison_sorted = (
    comparison
    .dropna()
    .sort_values("On Demand")
)

st.dataframe(comparison_sorted.head(10))

# MACHINE LEARNING
st.header("On-Demand Cost Prediction")

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"].str.replace(" GiB", ""),
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"].str.extract(
        r"(\d+)",
        expand=False
    ),
    errors="coerce"
)

st.subheader("Memory and vCPU Data")
st.dataframe(
    data[
        [
            "Instance Memory",
            "vCPUs"
        ]
    ].head()
)

data_cleaned = data.dropna(
    subset=[
        "On Demand",
        "Instance Memory",
        "vCPUs"
    ]
)

st.subheader("Cleaned Dataset")
st.write(
    f"Number of usable instances: {len(data_cleaned)}"
)

X = data_cleaned[
    [
        "Instance Memory",
        "vCPUs"
    ]
]

y = data_cleaned["On Demand"]

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.write(
    f"Training samples: {len(X_train)}"
)

st.write(
    f"Testing samples: {len(X_test)}"
)

# LINEAR REGRESSION
model = LinearRegression()

model.fit(
    X_train,
    y_train
)

st.subheader("Linear Regression Model")

st.write(
    f"Intercept: {model.intercept_:.6f}"
)

st.write(
    f"Memory coefficient: {model.coef_[0]:.6f}"
)

st.write(
    f"vCPU coefficient: {model.coef_[1]:.6f}"
)

# PREDICTIONS
y_pred = model.predict(X_test)

# MODEL METRICS
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

st.subheader("Model Performance")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "MAE",
    f"{mae:.6f}"
)

metric_col2.metric(
    "MSE",
    f"{mse:.6f}"
)

metric_col3.metric(
    "RMSE",
    f"{rmse:.6f}"
)

# ACTUAL VS PREDICTED
st.subheader("Actual vs Predicted On-Demand Costs")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

ax.plot(
    [
        min(y_test),
        max(y_test)
    ],
    [
        min(y_test),
        max(y_test)
    ],
    linestyle="--"
)

ax.set_title(
    "Actual vs Predicted On-Demand Costs"
)

ax.set_xlabel(
    "Actual On-Demand Cost"
)

ax.set_ylabel(
    "Predicted On-Demand Cost"
)

plt.tight_layout()

st.pyplot(fig)

# NEW INSTANCE PREDICTION
st.subheader("Predict Cost for a New Instance")

new_instance = [[4, 2]]

predicted_cost = model.predict(
    new_instance
)

st.write(
    f"Predicted On-Demand Cost for "
    f"4 GiB, 2 vCPUs: "
    f"${predicted_cost[0]:.4f}"
)