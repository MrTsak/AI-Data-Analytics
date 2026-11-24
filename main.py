"""Diabetes Prediction Dashboard — desktop version (original Gatehouse Awards project).

Data loading, model training and clustering live in the shared ``analysis`` package,
which also powers the browser version in ``web/``.
"""
import gc
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analysis.clustering import cluster_patients
from analysis.data import NUMERIC, SMOKING_LABELS, encode_patient, load_dataset
from analysis.models import LR_NAME, MODEL_NAMES, RF_NAME, train_models

sns.set_style("whitegrid")

SCATTER_SAMPLE = 3000
HYPOTHESES = [
    ("1. Higher HbA1c levels correlate with higher diabetes risk",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC4933534/"),
    ("2. BMI is positively correlated with diabetes risk",
     "https://www.escardio.org/The-ESC/Press-Office/Press-releases/Body-mass-index-is-a-more-powerful-risk-factor-for-diabetes-than-genetics"),
    ("3. Age is a significant factor in diabetes risk",
     "https://www.thelancet.com/journals/lanhl/article/PIIS2666-7568(21)00177-X/fulltext"),
    ("4. Blood glucose levels show strong correlation with diabetes",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC4484145/"),
]
LABELS = {
    "age": "Age", "bmi": "BMI", "HbA1c_level": "HbA1c Level",
    "blood_glucose_level": "Blood Glucose", "hypertension": "Hypertension",
    "heart_disease": "Heart Disease", "diabetes": "Diabetes",
}


def risk_band(p: float) -> str:
    return "High" if p >= 0.7 else "Moderate" if p >= 0.4 else "Low"


class DiabetesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction Dashboard")
        self.root.geometry("1400x1000")
        self.current_figure = None
        self.canvas = None
        self.content_frame = None
        self.theme_mode = "light"
        self._after_ids = []

        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")

        self.ds = load_dataset()
        self.df = self.ds.raw
        self.results = train_models(self.ds)
        self.clusters = cluster_patients(self.df)
        self.df_vis = self.df.copy()
        self.df_vis["Cluster"] = self.clusters.labels
        self.df_vis["smoking_history"] = self.df["smoking_history"].map(SMOKING_LABELS)
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    @property
    def best(self):
        return self.results.best

    # ---------- layout ----------
    def setup_ui(self):
        self.header_frame = ctk.CTkFrame(self.root, height=80)
        self.header_frame.pack(fill="x", padx=30, pady=(30, 20))
        ctk.CTkLabel(self.header_frame, text="Diabetes Prediction Analysis",
                     font=("Arial", 20, "bold")).pack(side="left", padx=25)
        self.theme_button = ctk.CTkButton(self.header_frame, text="🌙 Dark Mode", command=self.toggle_theme,
                                          width=140, height=40, font=("Arial", 14))
        self.theme_button.pack(side="right", padx=25)
        self.tabview = ctk.CTkTabview(self.root, width=1300, height=850, command=self.on_tab_change)
        self.tabview.pack(pady=(0, 30), padx=30, fill="both", expand=True)
        names = ["Data Overview", "Correlation", "Clustering", "Model Performance",
                 "Confusion", "Live Prediction", "Conclusions"]
        self.tabs = {name: self.tabview.add(name) for name in names}
        self.safe_update_tab("Data Overview")

    def toggle_theme(self):
        self.theme_mode = "dark" if self.theme_mode == "light" else "light"
        ctk.set_appearance_mode(self.theme_mode)
        self.theme_button.configure(text="🌙 Dark Mode" if self.theme_mode == "light" else "☀️ Light Mode")
        self.safe_update_tab(self.tabview.get())

    def on_tab_change(self):
        current = self.tabview.get()
        if current in self.tabs:
            self.safe_update_tab(current)

    def safe_update_tab(self, tab_name):
        self._cancel_after()
        self._after_ids.append(self.root.after(50, lambda: self._update_tab_content(tab_name)))

    def _cancel_after(self):
        for after_id in self._after_ids:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self._after_ids = []

    def _update_tab_content(self, tab_name):
        self.cleanup_previous_tab()
        if not self.root.winfo_exists():
            return
        self.content_frame = ctk.CTkFrame(self.tabs[tab_name])
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        {
            "Data Overview": self.show_data_overview,
            "Correlation": self.show_correlation,
            "Clustering": self.show_clustering,
            "Model Performance": self.show_model_performance,
            "Confusion": self.show_confusion_matrix,
            "Live Prediction": self.show_live_prediction,
            "Conclusions": self.show_conclusions,
        }[tab_name]()

    def cleanup_previous_tab(self):
        self._cancel_after()
        if self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
        self.canvas = None
        if self.current_figure is not None:
            plt.close(self.current_figure)
        self.current_figure = None
        if self.content_frame is not None:
            try:
                self.content_frame.destroy()
            except tk.TclError:
                pass
        self.content_frame = None
        gc.collect()

    def _card(self, parent, title, size=16):
        frame = ctk.CTkFrame(parent, border_width=2, corner_radius=12)
        frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(frame, text=title, font=("Arial", size, "bold")).pack(pady=(10, 5))
        return frame

    def _draw(self, master, pad=15):
        self.canvas = FigureCanvasTkAgg(self.current_figure, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=pad, pady=pad)

    # ---------- tabs ----------
    def show_data_overview(self):
        main = ctk.CTkScrollableFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        card = self._card(main, "BASIC STATISTICS")
        ds = self.ds
        summary = (f"{ds.n_original:,} records loaded · {ds.n_duplicates:,} exact duplicates removed · "
                   f"{ds.n_other_gender} 'Other' gender rows excluded · {len(self.df):,} used · "
                   f"{self.df['diabetes'].mean() * 100:.1f}% diabetic")
        ctk.CTkLabel(card, text=summary, font=("Arial", 12)).pack(pady=(0, 10))

        cols = ["gender", "age", "hypertension", "heart_disease", "smoking_history",
                "bmi", "HbA1c_level", "blood_glucose_level"]
        rows = (len(cols) + 2) // 3
        self.current_figure = plt.Figure(figsize=(14, 5 * rows))
        axes = self.current_figure.subplots(rows, 3).flatten()
        for ax, col in zip(axes, cols):
            if col in ("gender", "smoking_history", "hypertension", "heart_disease"):
                self.df_vis[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
                ax.tick_params(axis="x", rotation=30)
            else:
                sns.histplot(data=self.df_vis, x=col, hue="diabetes", kde=False, ax=ax,
                             bins=30, element="step", stat="density", common_norm=False)
            ax.set_title(f"{LABELS.get(col, col.replace('_', ' ').title())} Distribution", fontsize=10)
            ax.set_xlabel("")
            ax.grid(True, linestyle="--", alpha=0.7)
        for ax in axes[len(cols):]:
            ax.axis("off")
        self.current_figure.tight_layout()
        self._draw(main)

    def show_correlation(self):
        main = ctk.CTkScrollableFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        card = self._card(main, "HYPOTHESES")
        for text, url in HYPOTHESES:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=text, font=("Arial", 12), anchor="w").pack(side="left")
            link = ctk.CTkLabel(row, text="[source]", font=("Arial", 12, "underline"),
                                text_color="#1E90FF", cursor="hand2")
            link.pack(side="left", padx=(5, 0))
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open_new(u))

        plot_frame = ctk.CTkFrame(main)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.current_figure = plt.Figure(figsize=(10, 8))
        ax = self.current_figure.add_subplot(111)
        cols = NUMERIC + ["diabetes"]
        corr = self.df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0,
                    fmt=".2f", linewidths=0.5, square=True, ax=ax, mask=mask,
                    annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
        names = [LABELS[c] for c in cols]
        ax.set_title("Feature Correlations with Diabetes Outcome", fontsize=14, pad=20)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(names, rotation=0, fontsize=10)
        ctk.CTkLabel(main, text="Correlation ranges from -1 (perfect negative) to +1 (perfect positive); "
                                "values near 0 indicate no linear relationship.",
                     font=("Arial", 11), wraplength=1200).pack(padx=10, pady=5)
        self.current_figure.tight_layout()
        self._draw(plot_frame)

    def show_clustering(self):
        main = ctk.CTkScrollableFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        card = self._card(main, "CLUSTER INTERPRETATION (K-means on age, BMI, HbA1c)", 14)
        for i, text in enumerate(self.clusters.descriptions):
            ctk.CTkLabel(card, text=f"• Cluster {i} ({self.clusters.sizes[i]:,} patients) — {text}",
                         font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)

        sample = self.df_vis.sample(n=min(SCATTER_SAMPLE, len(self.df_vis)), random_state=42)
        self.current_figure = plt.Figure(figsize=(14, 12))
        axes = self.current_figure.subplots(2, 2)

        ax = axes[0, 0]
        sns.scatterplot(data=sample, x="HbA1c_level", y="bmi", hue="Cluster", palette="viridis",
                        ax=ax, s=25, alpha=0.7)
        ax.set_title(f"HbA1c vs BMI by Cluster ({SCATTER_SAMPLE:,}-patient sample)", fontsize=12)
        ax.set_xlabel("HbA1c Level (%)")
        ax.set_ylabel("Body Mass Index (BMI)")

        ax = axes[0, 1]
        centers = self.clusters.centers.copy()
        centers["Cluster"] = range(len(centers))
        melted = centers.melt(id_vars="Cluster", var_name="Feature", value_name="Value")
        melted["Feature"] = melted["Feature"].map(LABELS)
        sns.barplot(data=melted, x="Cluster", y="Value", hue="Feature", ax=ax, palette="Set2")
        ax.set_title("Cluster Centres", fontsize=12)

        ax = axes[1, 0]
        rates = pd.DataFrame({"Cluster": range(3), "rate": [r * 100 for r in self.clusters.diabetes_rate]})
        bars = sns.barplot(data=rates, x="Cluster", y="rate", hue="Cluster", palette="coolwarm",
                           legend=False, ax=ax)
        for p in bars.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.5, f"{p.get_height():.1f}%",
                    ha="center", fontsize=10)
        ax.set_title("Diabetes Prevalence by Cluster", fontsize=12)
        ax.set_ylabel("Diabetes Rate (%)")

        ax = axes[1, 1]
        sns.boxplot(data=self.df_vis, x="Cluster", y="age", hue="Cluster", palette="pastel",
                    legend=False, ax=ax)
        ax.set_title("Age Distribution by Cluster", fontsize=12)
        ax.set_ylabel("Age (Years)")
        self.current_figure.tight_layout()
        self._draw(main)

    def show_model_performance(self):
        main = ctk.CTkScrollableFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        m = self.results.metrics
        summary = ctk.CTkFrame(main, border_width=2, corner_radius=12, border_color="#4e8cff")
        summary.pack(fill="x", padx=10, pady=10, ipady=10)
        ctk.CTkLabel(summary, text=f"BEST MODEL (by ROC-AUC): {self.best} (AUC {m[self.best]['auc']:.3f})",
                     font=("Arial", 18, "bold"), text_color="#4e8cff").pack(pady=(5, 5))
        ctk.CTkLabel(summary, text="Only ~9% of patients are diabetic, so accuracy alone is misleading; "
                                   "AUC, recall and F1 show how well each model finds real cases.",
                     font=("Arial", 12)).pack()

        card = self._card(main, "DETAILED MODEL COMPARISON")
        header = ctk.CTkFrame(card)
        header.pack(fill="x", padx=20, pady=5)
        for i, h in enumerate(["Metric", *MODEL_NAMES]):
            ctk.CTkLabel(header, text=h, font=("Arial", 14, "bold"), width=150 if i == 0 else 180).pack(side="left")
        for key, name in [("auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("precision", "Precision"),
                          ("recall", "Recall"), ("f1", "F1-Score")]:
            row = ctk.CTkFrame(card)
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=name, font=("Arial", 14), width=150, anchor="w").pack(side="left")
            vals = [m[n][key] for n in MODEL_NAMES]
            for v in vals:
                color = "#4e8cff" if v == max(vals) else ("gray30", "gray70")
                ctk.CTkLabel(row, text=f"{v:.3f}", font=("Arial", 14), width=180, text_color=color).pack(side="left")

        roc_frame = ctk.CTkFrame(main)
        roc_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.current_figure = plt.Figure(figsize=(10, 7))
        ax = self.current_figure.add_subplot(111)
        for name, color in zip(MODEL_NAMES, ["#4e8cff", "#2ecc71"]):
            roc = m[name]["roc"]
            ax.plot(roc["fpr"], roc["tpr"], color=color, lw=2, label=f"{name} (AUC = {m[name]['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=15)
        ax.legend(loc="lower right")
        self._draw(roc_frame)

    def show_confusion_matrix(self):
        main = ctk.CTkFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        self.current_figure = plt.Figure(figsize=(16, 7))
        for i, (name, cmap) in enumerate(zip(MODEL_NAMES, ["Blues", "Greens"])):
            ax = self.current_figure.add_subplot(1, 2, i + 1)
            mm = self.results.metrics[name]
            sns.heatmap(np.array(mm["confusion"]), annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                        annot_kws={"size": 16, "weight": "bold"}, linewidths=1, linecolor="white")
            ax.set_title(f"{name} Confusion Matrix", fontsize=14, pad=15)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_xticklabels(["No Diabetes", "Diabetes"])
            ax.set_yticklabels(["No Diabetes", "Diabetes"], rotation=0)
            ax.text(0.5, -0.25, f"Accuracy: {mm['accuracy']:.3f} | Precision: {mm['precision']:.3f}\n"
                                f"Recall: {mm['recall']:.3f} | F1: {mm['f1']:.3f}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ctk.CTkLabel(main, text="Diagonal cells are correct predictions; off-diagonal cells are misclassifications.",
                     font=("Arial", 11)).pack(padx=10, pady=5)
        self.current_figure.tight_layout()
        self._draw(main, pad=20)

    def show_live_prediction(self):
        main = ctk.CTkScrollableFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        self._card(main, "LIVE DIABETES PREDICTION")
        ctk.CTkLabel(main, text="Educational project — not a medical device or diagnosis.",
                     font=("Arial", 12, "italic")).pack()

        model_card = self._card(main, "Select Prediction Model:", 14)
        self.prediction_model_var = ctk.StringVar(value=self.best)
        for option in MODEL_NAMES:
            ctk.CTkRadioButton(model_card, text=option, variable=self.prediction_model_var,
                               value=option).pack(pady=5, padx=20, anchor="w")

        inputs = self._card(main, "Enter Patient Details:", 14)
        self.gender_var = ctk.StringVar(value="Female")
        self._labeled_row(inputs, "Gender:", lambda f: ctk.CTkSegmentedButton(
            f, values=["Female", "Male"], variable=self.gender_var))

        self.feature_widgets = {}
        ranges = {"age": (0, 80, 80, "{:.0f}"), "bmi": (10, 60, 100, "{:.1f}"),
                  "HbA1c_level": (3.5, 9.0, 55, "{:.1f}"), "blood_glucose_level": (80, 300, 110, "{:.0f}")}
        for col, (lo, hi, steps, fmt) in ranges.items():
            row = self._labeled_row(inputs, f"{LABELS[col]}:")
            value_label = ctk.CTkLabel(row, width=60)
            slider = ctk.CTkSlider(row, from_=lo, to=hi, number_of_steps=steps,
                                   command=lambda v, lbl=value_label, f=fmt: lbl.configure(text=f.format(v)))
            start = float(np.clip(self.df[col].mean(), lo, hi))
            slider.set(start)
            value_label.configure(text=fmt.format(start))
            slider.pack(side="left", expand=True, fill="x", padx=10)
            value_label.pack(side="left")
            self.feature_widgets[col] = ("slider", slider, value_label)
        for col in ("hypertension", "heart_disease"):
            var = ctk.IntVar(value=0)
            self._labeled_row(inputs, f"{LABELS[col]}:", lambda f, v=var: ctk.CTkCheckBox(
                f, text="", variable=v, onvalue=1, offvalue=0))
            self.feature_widgets[col] = ("checkbox", var)

        self.smoking_options = {label: raw for raw, label in SMOKING_LABELS.items()}
        self.smoking_var = ctk.StringVar(value=SMOKING_LABELS["No Info"])
        self._labeled_row(inputs, "Smoking History:", lambda f: ctk.CTkComboBox(
            f, values=list(self.smoking_options), variable=self.smoking_var, width=220, state="readonly"))

        ctk.CTkButton(main, text="PREDICT DIABETES RISK", command=self.make_prediction,
                      font=("Arial", 14, "bold"), height=40).pack(pady=20)
        self.result_frame = ctk.CTkFrame(main, border_width=2, corner_radius=12)
        self.result_frame.pack(fill="x", padx=10, pady=10)
        self.result_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=(20, 5))
        self.probability_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 14))
        self.probability_label.pack(pady=(5, 20))

    def _labeled_row(self, parent, text, make_widget=None):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row, text=text, width=150, anchor="w").pack(side="left")
        if make_widget:
            make_widget(row).pack(side="left", padx=10)
        return row

    def build_patient(self) -> dict:
        patient = {"gender": self.gender_var.get(),
                   "smoking_history": self.smoking_options[self.smoking_var.get()]}
        for col, info in self.feature_widgets.items():
            patient[col] = float(info[1].get())
        return patient

    def make_prediction(self):
        model = self.results.rf if self.prediction_model_var.get() == RF_NAME else self.results.lr
        probability = float(model.predict_proba(encode_patient(self.build_patient()))[0, 1])
        band = risk_band(probability)
        color = {"High": "#FF6347", "Moderate": "#FFA500", "Low": "#2E8B57"}[band]
        verdict = "Positive" if probability >= 0.5 else "Negative"
        self.result_label.configure(text=f"Model prediction: {verdict} — {band} model probability",
                                    text_color=color)
        self.probability_label.configure(text=f"Probability: {probability * 100:.1f}%", text_color=color)

    def show_conclusions(self):
        main = ctk.CTkScrollableFrame(self.content_frame)
        main.pack(fill="both", expand=True, padx=20, pady=20)
        m = self.results.metrics
        rf = self.results.rf
        top = sorted(zip(rf.feature_names_in_, rf.feature_importances_), key=lambda t: -t[1])[:2]
        top_names = " and ".join(LABELS.get(n, n) for n, _ in top)
        card = self._card(main, "KEY FINDINGS")
        findings = [
            f"1. {self.best} performed best with ROC-AUC {m[self.best]['auc']:.3f} "
            f"(recall {m[self.best]['recall'] * 100:.1f}%)",
            f"2. {top_names} are the strongest predictors (Random Forest feature importance)",
            "3. Cluster analysis separates a high-HbA1c, older group with a much higher diabetes rate",
            "4. The model could be improved with lifestyle, family-history and genetic data",
        ]
        for f in findings:
            ctk.CTkLabel(card, text=f, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=5)
        card = self._card(main, "LIMITATIONS")
        for f in [f"• {self.ds.n_duplicates:,} duplicate records were removed before training",
                  "• Results describe this dataset only and are not clinically validated",
                  "• Educational project — not a medical device or diagnosis"]:
            ctk.CTkLabel(card, text=f, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)

    def on_close(self):
        self.cleanup_previous_tab()
        self.root.quit()
        self.root.destroy()


def main():
    root = ctk.CTk()
    try:
        DiabetesPredictorApp(root)
    except (OSError, KeyError, ValueError) as e:
        root.withdraw()
        messagebox.showerror("Could not start", f"Failed to load data or train models:\n{e}")
        root.destroy()
        sys.exit(1)
    root.mainloop()


if __name__ == "__main__":
    main()
