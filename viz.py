import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV into DataFrame
df = pd.read_csv("asr.csv")

# Filter Afrikaans only
df_af = df[df["lang"] == "afrikaans"]

# Static WER values (since not in CSV, hardcoded here)
wer_values = {
    "whisper_large": 0.345,
    "m4tv2": 0.211,
    "whisper_small": 0.699
}
wer_df = pd.DataFrame({
    "model": list(wer_values.keys()),
    "wer": list(wer_values.values())
})

# Plot 1: WER Comparison
plt.figure(figsize=(6, 4))
sns.barplot(data=wer_df, x="model", y="wer", palette="coolwarm")
plt.title("WER of ASR Models on Afrikaans Speech")
plt.ylabel("Word Error Rate (WER)")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("wer_comparison.png")
plt.show()

# Plot 2: Topic Coherence (NPMI and UMass)
for metric in ["npmi", "umass"]:
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df_af[df_af["eval"].isin(["lda", "bertopic"]) & df_af["metric"] == metric],
        x="model", y="value", hue="eval", palette="Set2"
    )
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Afrikaans Topic {metric.upper()} by Model and Method")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"topic_{metric}_afrikaans.png")
    plt.show()

# Plot 3: WER vs NPMI scatter
npmi_means = df_af[df_af["metric"] == "umass"].groupby("model")["value"].mean().reset_index()
npmi_wer_df = pd.merge(npmi_means, wer_df, on="model")

plt.figure(figsize=(6, 4))
sns.scatterplot(data=npmi_wer_df, x="wer", y="value", hue="model", s=100)
plt.title("WER vs Average Topic UMass (Afrikaans)")
plt.xlabel("Word Error Rate")
plt.ylabel("Avg UMass Coherence")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("wer_vs_npmi.png")
plt.show()
