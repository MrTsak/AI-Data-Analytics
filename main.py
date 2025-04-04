import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_curve, auc, RocCurveDisplay)
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import gc
from sklearn.preprocessing import StandardScaler

# Set the style for seaborn
sns.set_style("whitegrid")

class DiabetesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction Dashboard")
        self.root.geometry("1400x1000") 
        
        # Initialize variables
        self.current_figure = None
        self.canvas = None
        self.content_frame = None
        self.theme_mode = "light"
        self._after_ids = []  # Store after IDs for cleanup
        
        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")
        
        # Load data
        self.load_data()
        
        # Train models
        self.train_models()
        
        # Perform clustering
        self.perform_clustering()
        
        # Setup UI
        self.setup_ui()
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv('data/sample.csv', sep=';')
            self.df.rename(columns={
                'Pregnancies': 'Pregnancies',
                'Glucose': 'Glucose',
                'BloodPressure': 'BP',
                'SkinThickness': 'SkinThick',
                'Insulin': 'Insulin',
                'BMI': 'BMI',
                'DiabetesPedigreeFunction': 'DPF',
                'Age': 'Age',
                'Outcome': 'Outcome'
            }, inplace=True)
            self.df.fillna(self.df.mean(), inplace=True)
            
            # Store basic stats for display
            self.data_stats = {
                'head': self.df.head(),
                'describe': self.df.describe(),
                'info': self.df.info(),
                'null_counts': self.df.isnull().sum()
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def train_models(self):
        """Train machine learning models"""
        try:
            features = self.df.drop(columns=['Outcome'])
            target = self.df['Outcome']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features, target, test_size=0.2, random_state=42)
            
            # Random Forest
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(self.X_train, self.y_train)
            self.rf_pred = self.rf_model.predict(self.X_test)
            self.rf_probs = self.rf_model.predict_proba(self.X_test)[:, 1]
            
            # Logistic Regression
            self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
            self.lr_model.fit(self.X_train, self.y_train)
            self.lr_pred = self.lr_model.predict(self.X_test)
            self.lr_probs = self.lr_model.predict_proba(self.X_test)[:, 1]
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'Feature': features.columns,
                'Importance': self.rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
        except Exception as e:
            print(f"Error training models: {e}")
            sys.exit(1)

    def perform_clustering(self):
        """Perform clustering analysis"""
        try:
            # Select features for clustering
            cluster_features = self.df[['Glucose', 'BMI', 'Age']]
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_features)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            self.df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            # Store cluster centers
            self.cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            sys.exit(1)

    def setup_ui(self):
        """Initialize the user interface"""
        # Header frame with theme toggle
        self.header_frame = ctk.CTkFrame(self.root, height=80)
        self.header_frame.pack(fill="x", padx=30, pady=(30, 20))
        
        # Application title
        ctk.CTkLabel(self.header_frame, 
                    text="Diabetes Prediction Analysis",
                    font=("Arial", 20, "bold")).pack(side="left", padx=25)
        
        # Theme toggle button
        self.theme_button = ctk.CTkButton(
            self.header_frame,
            text="🌙 Dark Mode" if self.theme_mode == "light" else "☀️ Light Mode",
            command=self.toggle_theme,
            width=140,
            height=40,
            font=("Arial", 14)
        )
        self.theme_button.pack(side="right", padx=25)
        
        # Main tab view
        self.tabview = ctk.CTkTabview(self.root, width=1300, height=850)
        self.tabview.pack(pady=(0, 30), padx=30, fill="both", expand=True)
        
        # Add tabs
        self.tabs = {
            "Data Overview": self.tabview.add("Data Overview"),
            "Correlation": self.tabview.add("Correlation"),
            "Clustering": self.tabview.add("Clustering"),
            "Model Comparison": self.tabview.add("Model Comparison"),
            "Performance": self.tabview.add("Performance"),
            "Confusion": self.tabview.add("Confusion"),
            "Features": self.tabview.add("Features"),
            "Conclusions": self.tabview.add("Conclusions")
        }
        
        # Initialize first tab
        self.safe_update_tab("Data Overview")
        
        # Bind tab change
        self.tabview.configure(command=lambda: self.on_tab_change())

    def toggle_theme(self):
        """Toggle between light and dark themes"""
        try:
            current_tab = self.tabview.get()
            self.theme_mode = "dark" if self.theme_mode == "light" else "light"
            ctk.set_appearance_mode(self.theme_mode)
            self.theme_button.configure(
                text="🌙 Dark Mode" if self.theme_mode == "light" else "☀️ Light Mode"
            )
            self.safe_update_tab(current_tab)
        except Exception as e:
            print(f"Error in theme toggle: {e}")

    def on_tab_change(self):
        """Handle tab change events"""
        try:
            current_tab = self.tabview.get()
            if current_tab in self.tabs:
                self.safe_update_tab(current_tab)
        except Exception as e:
            print(f"Error in tab change: {e}")

    def safe_update_tab(self, tab_name):
        """Safely update tab content with error handling"""
        # Cancel any pending updates
        for after_id in self._after_ids:
            self.root.after_cancel(after_id)
        self._after_ids = []
        
        # Schedule the update
        self._after_ids.append(
            self.root.after(50, lambda: self._update_tab_content(tab_name))
        )

    def _update_tab_content(self, tab_name):
        """Actually update the tab content"""
        try:
            self.cleanup_previous_tab()
            
            if not self.root.winfo_exists():
                return
                
            self.content_frame = ctk.CTkFrame(self.tabs[tab_name])
            self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            if tab_name == "Data Overview":
                self.show_data_overview()
            elif tab_name == "Correlation":
                self.show_correlation()
            elif tab_name == "Clustering":
                self.show_clustering()
            elif tab_name == "Model Comparison":
                self.show_model_comparison()
            elif tab_name == "Performance":
                self.show_performance()
            elif tab_name == "Confusion":
                self.show_confusion_matrix()
            elif tab_name == "Features":
                self.show_feature_importance()
            elif tab_name == "Conclusions":
                self.show_conclusions()
                
        except Exception as e:
            print(f"Error updating tab {tab_name}: {e}")
            self.cleanup_previous_tab()

    def cleanup_previous_tab(self):
        """Clean up resources from previous tab"""
        # Cancel any pending events
        for after_id in self._after_ids:
            try:
                self.root.after_cancel(after_id)
            except:
                pass
        self._after_ids = []
        
        # Safe destruction order
        try:
            if self.canvas and self.canvas.get_tk_widget().winfo_exists():
                self.canvas.get_tk_widget().destroy()
        except:
            pass
        self.canvas = None
        
        try:
            if self.current_figure:
                plt.close(self.current_figure)
        except:
            pass
        self.current_figure = None
        
        try:
            if self.content_frame and self.content_frame.winfo_exists():
                self.content_frame.destroy()
        except:
            pass
        self.content_frame = None
        
        # Force garbage collection
        gc.collect()

    def show_data_overview(self):
        """Display data overview with statistics and distributions"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Basic stats
            stats_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            stats_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            
            ctk.CTkLabel(stats_frame, text="BASIC STATISTICS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            # Create distribution plots
            columns_to_plot = [col for col in self.df.columns if col not in ['Outcome', 'Cluster']]
            num_plots = len(columns_to_plot)
            
            # Calculate grid size (3 columns)
            rows = (num_plots + 2) // 3  # Round up division
            cols = 3
            
            self.current_figure = plt.Figure(figsize=(12, 4*rows))  # Adjust height based on rows
            axes = self.current_figure.subplots(rows, cols)
            self.current_figure.tight_layout(pad=4.0)
            
            # Flatten axes array if needed
            if rows > 1:
                axes = axes.flatten()
            
            for i, col in enumerate(columns_to_plot):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f"{col} Distribution", fontsize=10)
                axes[i].set_xlabel("")
            
            # Turn off any empty subplots
            for j in range(i+1, rows*cols):
                axes[j].axis('off')
            
            if self.content_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing data overview: {e}")
            self.cleanup_previous_tab()

    def show_correlation(self):
        """Display correlation heatmap with hypotheses"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            # Create main frame with scroll
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Hypotheses frame
            hypotheses_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            hypotheses_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(hypotheses_frame, text="HYPOTHESES", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            hypotheses = [
                "1. Higher glucose levels correlate with higher diabetes risk",
                "2. BMI is positively correlated with diabetes risk",
                "3. Age is a significant factor in diabetes risk",
                "4. Blood pressure shows weaker correlation than other factors"
            ]
            
            for hypo in hypotheses:
                ctk.CTkLabel(hypotheses_frame, text=hypo, 
                            font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)
            
            # Correlation plot
            plot_frame = ctk.CTkFrame(main_frame)
            plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            self.current_figure = plt.Figure(figsize=(11, 9))
            ax = self.current_figure.add_subplot(111)
            
            corr = self.df.corr()
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            
            sns.heatmap(
                corr, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt=".2f", 
                linewidths=0.5, 
                square=True,
                ax=ax, 
                mask=mask,
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8}
            )
            
            ax.set_title("Feature Correlations with Diabetes Outcome", fontsize=14, pad=20)
            ax.set_xticklabels(
                corr.columns,
                rotation=45,
                ha='right',
                fontsize=11
            )
            ax.set_yticklabels(
                corr.columns,
                rotation=0,
                fontsize=11
            )
            
            if plot_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing correlation: {e}")
            self.cleanup_previous_tab()

    def show_clustering(self):
        """Display clustering results"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Cluster info frame
            info_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            info_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(info_frame, text="CLUSTER ANALYSIS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            # Cluster scatter plot
            self.current_figure = plt.Figure(figsize=(12, 10))
            axes = self.current_figure.subplots(2, 2)
            
            # Cluster scatter plot
            ax = axes[0, 0]
            sns.scatterplot(
                data=self.df, 
                x='Glucose', 
                y='BMI', 
                hue='Cluster', 
                palette='viridis', 
                ax=ax
            )
            ax.set_title("Glucose vs BMI by Cluster")
            
            # Cluster centers
            ax = axes[0, 1]
            centers_df = pd.DataFrame(
                self.cluster_centers, 
                columns=['Glucose', 'BMI', 'Age']
            )
            centers_df['Cluster'] = range(3)
            sns.barplot(
                data=centers_df.melt(id_vars='Cluster'), 
                x='Cluster', 
                y='value', 
                hue='variable',
                ax=ax
            )
            ax.set_title("Cluster Centers (Standardized)")
            ax.legend(title='Feature')
            
            # Outcome by cluster
            ax = axes[1, 0]
            cluster_outcome = self.df.groupby('Cluster')['Outcome'].mean().reset_index()
            sns.barplot(
                data=cluster_outcome, 
                x='Cluster', 
                y='Outcome', 
                ax=ax
            )
            ax.set_title("Diabetes Prevalence by Cluster")
            ax.set_ylabel("Diabetes Rate")
            
            # Age distribution by cluster
            ax = axes[1, 1]
            sns.boxplot(
                data=self.df, 
                x='Cluster', 
                y='Age', 
                ax=ax
            )
            ax.set_title("Age Distribution by Cluster")
            
            self.current_figure.tight_layout()
            
            if main_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing clustering: {e}")
            self.cleanup_previous_tab()

    def show_model_comparison(self):
        """Compare model performance"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Model comparison frame
            compare_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            compare_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(compare_frame, text="MODEL COMPARISON", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            # Calculate metrics
            rf_accuracy = accuracy_score(self.y_test, self.rf_pred)
            lr_accuracy = accuracy_score(self.y_test, self.lr_pred)
            
            # Create comparison table
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            rf_metrics = [
                rf_accuracy,
                classification_report(self.y_test, self.rf_pred, output_dict=True)['1']['precision'],
                classification_report(self.y_test, self.rf_pred, output_dict=True)['1']['recall'],
                classification_report(self.y_test, self.rf_pred, output_dict=True)['1']['f1-score']
            ]
            
            lr_metrics = [
                lr_accuracy,
                classification_report(self.y_test, self.lr_pred, output_dict=True)['1']['precision'],
                classification_report(self.y_test, self.lr_pred, output_dict=True)['1']['recall'],
                classification_report(self.y_test, self.lr_pred, output_dict=True)['1']['f1-score']
            ]
            
            for i, metric in enumerate(metrics):
                metric_frame = ctk.CTkFrame(compare_frame)
                metric_frame.pack(fill="x", padx=20, pady=5)
                
                ctk.CTkLabel(metric_frame, text=metric, 
                            font=("Arial", 14), width=120, anchor="w").pack(side="left")
                
                ctk.CTkLabel(metric_frame, text=f"RF: {rf_metrics[i]:.3f}", 
                            font=("Arial", 14), width=120).pack(side="left", padx=20)
                
                ctk.CTkLabel(metric_frame, text=f"LR: {lr_metrics[i]:.3f}", 
                            font=("Arial", 14), width=120).pack(side="left")
            
            # ROC curves
            plot_frame = ctk.CTkFrame(main_frame)
            plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            self.current_figure = plt.Figure(figsize=(10, 6))
            ax = self.current_figure.add_subplot(111)
            
            # Calculate ROC curves
            fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.rf_probs)
            roc_auc_rf = auc(fpr_rf, tpr_rf)
            
            fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.lr_probs)
            roc_auc_lr = auc(fpr_lr, tpr_lr)
            
            # Plot ROC curves
            ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
            ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve Comparison')
            ax.legend(loc='lower right')
            
            if plot_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing model comparison: {e}")
            self.cleanup_previous_tab()

    def show_performance(self):
        """Display model performance metrics"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Accuracy display
            acc_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12,
                                    border_color="#4e8cff", fg_color=("#f0f0f0", "#2b2b2b"))
            acc_frame.pack(fill="x", padx=10, pady=10, ipady=15)
            
            accuracy = accuracy_score(self.y_test, self.rf_pred)
            ctk.CTkLabel(acc_frame, text="RANDOM FOREST PERFORMANCE", 
                        font=("Arial", 16, "bold")).pack(pady=(5,0))
            ctk.CTkLabel(acc_frame, text=f"{accuracy*100:.2f}%", 
                        font=("Arial", 32, "bold"), 
                        text_color="#4e8cff").pack(pady=(0,10))
            
            # Classification metrics
            metrics_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            metrics_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            
            ctk.CTkLabel(metrics_frame, text="CLASSIFICATION METRICS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            report = classification_report(self.y_test, self.rf_pred, output_dict=True)
            
            for label, metric in report['1'].items():
                if label in ['precision', 'recall', 'f1-score']:
                    metric_frame = ctk.CTkFrame(metrics_frame)
                    metric_frame.pack(fill="x", padx=20, pady=5)
                    
                    ctk.CTkLabel(metric_frame, text=label.title(), 
                                font=("Arial", 14), width=120, anchor="w").pack(side="left")
                    ctk.CTkLabel(metric_frame, text=f"{metric:.3f}", 
                                font=("Arial", 14, "bold")).pack(side="right")
            
        except Exception as e:
            print(f"Error showing performance: {e}")
            self.cleanup_previous_tab()

    def show_confusion_matrix(self):
        """Display confusion matrix"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            self.current_figure = plt.Figure(figsize=(8, 6))
            ax = self.current_figure.add_subplot(111)
            
            cm = confusion_matrix(self.y_test, self.rf_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=["No Diabetes", "Diabetes"], 
                       yticklabels=["No Diabetes", "Diabetes"], 
                       ax=ax, annot_kws={"size": 16})
            
            ax.set_title("Random Forest Confusion Matrix", fontsize=16, pad=20)
            ax.set_xlabel("Predicted", fontsize=14)
            ax.set_ylabel("Actual", fontsize=14)
            
            if main_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
            
        except Exception as e:
            print(f"Error showing confusion matrix: {e}")
            self.cleanup_previous_tab()

    def show_feature_importance(self):
        """Display feature importance"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Feature importance plot
            plot_frame = ctk.CTkFrame(main_frame)
            plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            self.current_figure = plt.Figure(figsize=(10, 6))
            ax = self.current_figure.add_subplot(111)
            
            sns.barplot(
                data=self.feature_importance, 
                x='Importance', 
                y='Feature', 
                color="#4e8cff", 
                ax=ax
            )
            ax.set_title("Random Forest Feature Importance", fontsize=16, pad=20)
            ax.set_xlabel("Importance Score", fontsize=14)
            ax.set_ylabel("Features", fontsize=14)
            
            if plot_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
            # Feature descriptions
            desc_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            desc_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(desc_frame, text="KEY FEATURE INSIGHTS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            insights = [
                "• Glucose is the most important predictor of diabetes",
                "• BMI and Age are significant secondary factors",
                "• Diabetes Pedigree Function captures genetic risk",
                "• Blood Pressure has relatively low importance"
            ]
            
            for insight in insights:
                ctk.CTkLabel(desc_frame, text=insight, 
                            font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)
            
        except Exception as e:
            print(f"Error showing feature importance: {e}")
            self.cleanup_previous_tab()

    def show_conclusions(self):
        """Display conclusions and recommendations"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Conclusions frame
            conclusions_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            conclusions_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(conclusions_frame, text="CONCLUSIONS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            accuracy = accuracy_score(self.y_test, self.rf_pred) * 100
            conclusions = [
                f"1. Random Forest achieved {accuracy:.2f}% accuracy",
                "2. Glucose levels are the strongest predictor of diabetes",
                "3. Clustering revealed distinct patient groups with different risk profiles",
                "4. The model could be improved with more clinical features",
                "5. Early intervention for high-Glucose patients is recommended"
            ]
            
            for conc in conclusions:
                ctk.CTkLabel(conclusions_frame, text=conc, 
                            font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=5)
            
            # Recommendations frame
            rec_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            rec_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(rec_frame, text="RECOMMENDATIONS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            recommendations = [
                "• Collect more data on patient lifestyle factors",
                "• Include HbA1c measurements for better glucose monitoring",
                "• Develop a risk score calculator based on the top features",
                "• Consider ensemble methods to further improve accuracy"
            ]
            
            for rec in recommendations:
                ctk.CTkLabel(rec_frame, text=rec, 
                            font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)
            
        except Exception as e:
            print(f"Error showing conclusions: {e}")
            self.cleanup_previous_tab()

    def on_close(self):
        """Handle application closing"""
        try:
            # Cancel all pending events
            for after_id in self._after_ids:
                try:
                    self.root.after_cancel(after_id)
                except:
                    pass
            
            # Clean up resources
            self.cleanup_previous_tab()
            
            # Destroy root window
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
                
        except Exception as e:
            print(f"Error during close: {e}")
        finally:
            sys.exit(0)

def main():
    """Main application entry point"""
    try:
        root = ctk.CTk()
        
        def on_closing():
            try:
                app.on_close()
                root.quit()
            except:
                pass
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        app = DiabetesPredictorApp(root)
        root.mainloop()
                
    except Exception as e:
        print(f"Application startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()