import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ["Linear Regression", "Random Forest", "Gradient Boosting", "KNN", "XGBoost"]
rmse = [63140, 46220, 49347, 55983, 41969]
r2 = [0.58, 0.77, 0.74, 0.67, 0.82]

# Create DataFrame
df = pd.DataFrame({
    "Model": models,
    "RMSE": rmse,
    "R2": r2
})

sns.set_theme(style="whitegrid")

# --- RMSE Plot ---
plt.figure(figsize=(8,5))
ax1 = sns.barplot(x="Model", y="RMSE", data=df, palette="coolwarm", edgecolor="black")

# Annotate values on bars
for p in ax1.patches:
    ax1.annotate(format(int(p.get_height()), ','),
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom',
                 fontsize=10, fontweight="bold", color="black")

plt.title("Model Comparison - RMSE (Lower is Better)", fontsize=14, fontweight="bold")
plt.ylabel("RMSE")
plt.xlabel("Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# --- R² Plot ---
plt.figure(figsize=(8,5))
ax2 = sns.barplot(x="Model", y="R2", data=df, palette="crest", edgecolor="black")

# Annotate values on bars
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}",
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom',
                 fontsize=10, fontweight="bold", color="black")

plt.title("Model Comparison - R² Score (Higher is Better)", fontsize=14, fontweight="bold")
plt.ylabel("R² Score")
plt.xlabel("Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
