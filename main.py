import sys
import gc
import webbrowser
import pandas as pd
import numpy as np
import seaborn as sns
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc, f1_score, recall_score, precision_score)

sns.set_style("whitegrid")
# Set the default figure size for seaborn plots
class DiabetesPredictorApp:
    # Initialize the application
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction Dashboard")
        self.root.geometry("1400x1000") 
        self.current_figure = None
        self.canvas = None
        self.content_frame = None
        self.theme_mode = "light"
        self._after_ids = []
        # Setting the theme
        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")
        # Running the main functions
        self.load_data()     
        self.train_models()       
        self.perform_clustering()
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    # Load the data and rename the culumns
    def load_data(self):
        try:
            self.df = pd.read_csv('data/sample.csv')
            # Convert categorical variables to numerical
            self.df['gender'] = self.df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
            self.df['smoking_history'] = self.df['smoking_history'].map({'never': 0,'No Info': 1,'current': 2, 'former': 3,'ever': 4,'not current': 5
            })
            # Fill missing values with mean
            self.df.fillna(self.df.mean(), inplace=True)
            # A little bit of data printing on the console
            print(self.df.head())
            print(self.df.info())
            print(self.df.describe().round(2))
            # Error handling for missing values
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def train_models(self):
        try:
            # Splitting the data into features and target variable
            features = self.df.drop(columns=['diabetes', 'Cluster'], errors='ignore')
            # The diabetes outcome is the target variable
            target = self.df['diabetes'] 
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target)
        
            # Random Forest Classifier
            self.rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=7,
                min_samples_split=8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1)
            
            # Fit the model
            self.rf_model.fit(self.X_train, self.y_train)
            self.rf_pred = self.rf_model.predict(self.X_test)
            self.rf_probs = self.rf_model.predict_proba(self.X_test)[:, 1]

            # Logistic Regression Classifier
            self.lr_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear',
                penalty='l2',
                C=0.1,
                random_state=42)
            
            # predicting the outcome
            self.lr_model.fit(self.X_train, self.y_train)
            self.lr_pred = self.lr_model.predict(self.X_test)
            self.lr_probs = self.lr_model.predict_proba(self.X_test)[:, 1]

            # Calculations for accuracy, F1, recall and precision
            rf_accuracy = accuracy_score(self.y_test, self.rf_pred)
            lr_accuracy = accuracy_score(self.y_test, self.lr_pred)
            rf_f1 = f1_score(self.y_test, self.rf_pred)
            lr_f1 = f1_score(self.y_test, self.lr_pred)
            rf_recall = recall_score(self.y_test, self.rf_pred)
            lr_recall = recall_score(self.y_test, self.lr_pred)
            rf_precision = precision_score(self.y_test, self.rf_pred)
            lr_precision = precision_score(self.y_test, self.lr_pred)
            
            # Confusion matrices
            rf_cm = confusion_matrix(self.y_test, self.rf_pred)
            lr_cm = confusion_matrix(self.y_test, self.lr_pred)
            
            # Better model selection
            if rf_accuracy > lr_accuracy:
                self.better_model = "Random Forest"
            else:
                self.better_model = "Logistic Regression"

            # Store the best accuracy
            self.better_accuracy = max(rf_accuracy, lr_accuracy)
            
            # Store all the metrics
            self.model_metrics = {
                'Random Forest': {
                    'Accuracy': rf_accuracy,
                    'F1': rf_f1,
                    'Recall': rf_recall,
                    'Precision': rf_precision,
                    'Confusion Matrix': rf_cm},
                'Logistic Regression': {
                    'Accuracy': lr_accuracy,
                    'F1': lr_f1,
                    'Recall': lr_recall,
                    'Precision': lr_precision,
                    'Confusion Matrix': lr_cm}}

        except Exception as e:
            print(f"Error training models: {e}")
            sys.exit(1)
    # Clustering data
    def perform_clustering(self):
        try:
            # 3 factors for clustering
            cluster_features = self.df[['HbA1c_level', 'bmi', 'age']] 
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_features)  
            # KMeans usage
            kmeans = KMeans(n_clusters=3, random_state=42)
            self.df['Cluster'] = kmeans.fit_predict(scaled_features)
            self.cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

        except Exception as e:
            print(f"Error in clustering: {e}")
            sys.exit(1)
    # setting the UI
    def setup_ui(self):
        # Main frame
        self.header_frame = ctk.CTkFrame(self.root, height=80)
        self.header_frame.pack(fill="x", padx=30, pady=(30, 20))
        # Header label
        ctk.CTkLabel(self.header_frame, text="Diabetes Prediction Analysis", font=("Arial", 20, "bold")).pack(side="left", padx=25)
        # I made a theme changing feature haha
        self.theme_button = ctk.CTkButton(self.header_frame, text="🌙 Dark Mode" if self.theme_mode == "light" else "☀️ Light Mode", command=self.toggle_theme, width=140, height=40, font=("Arial", 14))
        self.theme_button.pack(side="right", padx=25)
        # tabs box
        self.tabview = ctk.CTkTabview(self.root, width=1300, height=850)
        self.tabview.pack(pady=(0, 30), padx=30, fill="both", expand=True)    
        # Adding the tabs
        self.tabs = {
            "Data Overview": self.tabview.add("Data Overview"),
            "Correlation": self.tabview.add("Correlation"),
            "Clustering": self.tabview.add("Clustering"),
            "Model Performance": self.tabview.add("Model Performance"),
            "Confusion": self.tabview.add("Confusion"),
            "Conclusions": self.tabview.add("Conclusions")}      
        # Tab update
        self.safe_update_tab("Data Overview") 
        self.tabview.configure(command=lambda: self.on_tab_change())
    # function for the theme to change
    def toggle_theme(self):
        # Dark and light theme
        try:
            current_tab = self.tabview.get()
            self.theme_mode = "dark" if self.theme_mode == "light" else "light"
            ctk.set_appearance_mode(self.theme_mode)
            self.theme_button.configure(text="🌙 Dark Mode" if self.theme_mode == "light" else "☀️ Light Mode")
            self.safe_update_tab(current_tab)
        # Again error handling
        except Exception as e:
            print(f"Error in theme toggle: {e}")
    # Check if the tab exists with error handing again (I do it everywhere)
    def on_tab_change(self):
        try:
            current_tab = self.tabview.get()
            if current_tab in self.tabs:
                self.safe_update_tab(current_tab)

        except Exception as e:
            print(f"Error in tab change: {e}")
    # A delay between changing the tabs so it doesn't bug out
    def safe_update_tab(self, tab_name):
        for after_id in self._after_ids:
            self.root.after_cancel(after_id)
        self._after_ids = []
        self._after_ids.append(self.root.after(50, lambda: self._update_tab_content(tab_name)))

    def _update_tab_content(self, tab_name):
        try:
            self.cleanup_previous_tab()
            if not self.root.winfo_exists():
                return
            # Frame for the tabs
            self.content_frame = ctk.CTkFrame(self.tabs[tab_name])
            self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)
            # Show content based on the selected tab 
            if tab_name == "Data Overview":
                self.show_data_overview()
            elif tab_name == "Correlation":
                self.show_correlation()
            elif tab_name == "Clustering":
                self.show_clustering()
            elif tab_name == "Model Performance":
                self.show_model_performance()
            elif tab_name == "Confusion":
                self.show_confusion_matrix()
            elif tab_name == "Conclusions":
                self.show_conclusions()
            
        except Exception as e:
            print(f"Error updating tab {tab_name}: {e}")
            self.cleanup_previous_tab()
    # Cleanup function when changing the tabs for memory 
    def cleanup_previous_tab(self):
        for after_id in self._after_ids:
            try:
                self.root.after_cancel(after_id)
            except:
                pass
        self._after_ids = []  
        
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
        
        gc.collect()
    # Function for the data overview tab
    def show_data_overview(self):
        # Check if the content frame exists before proceeding (I do it for every tab)
        if not self.content_frame.winfo_exists():
            return
        try:
            # Scrollale frame, I use everywhere it's needed
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            # Frame for the data overview
            stats_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            stats_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            # Header label
            ctk.CTkLabel(stats_frame, text="BASIC STATISTICS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            # Making a list of the colums despite the outcome and cluster
            columns_to_plot = [col for col in self.df.columns if col not in ['diabetes', 'Cluster']]
            # Calculate the plots, rows and columns
            num_plots = len(columns_to_plot) 
            rows = (num_plots + 2) // 3
            cols = 3
            # Making dynamic plots using matplotlib and the grids
            self.current_figure = plt.Figure(figsize=(12, 4*rows))
            axes = self.current_figure.subplots(rows, cols)
            self.current_figure.tight_layout(pad=4.0)

            if rows > 1:
                axes = axes.flatten()
            # Making the histograms with seaborn
            for i, col in enumerate(columns_to_plot):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f"{col} Distribution", fontsize=10)
                axes[i].set_xlabel("")
            # Hiding unused axes
            for j in range(i+1, rows*cols):
                axes[j].axis('off')
            # Check if the plot if the frame exists and render the plots
            if self.content_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing data overview: {e}")
            self.cleanup_previous_tab()
    # Correlation tab
    def show_correlation(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # I made a frame for "hypotheses" found in google
            hypotheses_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            hypotheses_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(hypotheses_frame, text="HYPOTHESES", font=("Arial", 16, "bold")).pack(pady=(10, 5))
            # So I put them in as well as some sources
            hypotheses = [
                ("1. Higher HbA1c levels correlate with higher diabetes risk", 
                 "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3747573/"),
                ("2. BMI is positively correlated with diabetes risk", 
                 "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8740746/"),
                ("3. Age is a significant factor in diabetes risk", 
                 "https://www.thelancet.com/journals/lanhl/article/PIIS2666-7568(21)00177-X/fulltext"),
                ("4. Blood glucose levels show strong correlation with diabetes", 
                 "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595710/")]
            # Making the hypotheses rows, hyperlinks function and the labels
            for hypo_text, url in hypotheses:
                hypo_row = ctk.CTkFrame(hypotheses_frame, fg_color="transparent")
                hypo_row.pack(fill="x", padx=20, pady=2)
                ctk.CTkLabel(hypo_row, text=hypo_text, font=("Arial", 12), anchor="w").pack(side="left")
                link_label = ctk.CTkLabel(hypo_row, text="[source]", font=("Arial", 12, "underline"), text_color="#1E90FF", cursor="hand2")
                link_label.pack(side="left", padx=(5, 0))
                link_label.bind("<Button-1>", lambda e, url=url: webbrowser.open_new(url))
            # Frame for the visualization of the heatmap
            plot_frame = ctk.CTkFrame(main_frame)
            plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.current_figure = plt.Figure(figsize=(11, 9))
            ax = self.current_figure.add_subplot(111)
            # Calculating the correlation matrix and masking the upper triangle
            corr = self.df.corr()
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            # Seaborn heatmap
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5, square=True, ax=ax, mask=mask, annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
            # Setting the title and the names of the heatmap
            ax.set_title("Feature Correlations with Diabetes Outcome", fontsize=14, pad=20)
            ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=11)
            ax.set_yticklabels(corr.columns, rotation=0, fontsize=11)
    
            if plot_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

        except Exception as e:
            print(f"Error showing correlation: {e}")
            self.cleanup_previous_tab()
    # Clustering tab        
    def show_clustering(self):
        if not self.content_frame.winfo_exists():
            return   
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            info_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            info_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(info_frame, text="CLUSTER ANALYSIS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            self.current_figure = plt.Figure(figsize=(12, 10))
            axes = self.current_figure.subplots(2, 2)
            # I made 4 differnet kind of plots for the clustering tab
            # Scatter oplot for HbA1c_level and bmi
            ax = axes[0, 0]
            sns.scatterplot(data=self.df, x='HbA1c_level', y='bmi', hue='Cluster', palette='viridis', ax=ax)
            ax.set_title("HbA1c vs BMI by Cluster")
            # Cluster centers for HbA1c_level, bmi and age
            ax = axes[0, 1]
            centers_df = pd.DataFrame(self.cluster_centers, columns=['HbA1c_level', 'bmi', 'age'])
            centers_df['Cluster'] = range(3)
            sns.barplot(data=centers_df.melt(id_vars='Cluster'), x='Cluster', y='value', hue='variable', ax=ax)
            ax.set_title("Cluster Centers (Standardized)")
            ax.legend(title='Feature')
            # Diabetes prevalence by cluster
            ax = axes[1, 0]
            cluster_outcome = self.df.groupby('Cluster')['diabetes'].mean().reset_index()
            sns.barplot(data=cluster_outcome, x='Cluster', y='diabetes', ax=ax)
            ax.set_title("Diabetes Prevalence by Cluster")
            ax.set_ylabel("Diabetes Rate")
            # Boxplot for Age distribution by cluster
            ax = axes[1, 1]
            sns.boxplot(data=self.df, x='Cluster', y='age', ax=ax)
            ax.set_title("Age Distribution by Cluster")
            
            self.current_figure.tight_layout()
            if main_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing clustering: {e}")
            self.cleanup_previous_tab()
    # Model performance tab
    def show_model_performance(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            # A "Best model" frame calculated by the results
            summary_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12, border_color="#4e8cff", fg_color=("#f0f0f0", "#2b2b2b"))
            summary_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            ctk.CTkLabel(summary_frame, text=f"BEST MODEL: {self.better_model} (Accuracy: {self.better_accuracy*100:.1f}%)", font=("Arial", 18, "bold"), text_color="#4e8cff").pack(pady=(5, 10))
            # And the frame for the detailed model comparison
            metrics_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            metrics_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            ctk.CTkLabel(metrics_frame, text="DETAILED MODEL COMPARISON", font=("Arial", 16, "bold")).pack(pady=(10,5))
            # The headers and the metrics for the comparison
            headers = ["Metric", "Random Forest", "Logistic Regression"]
            metrics = [("Accuracy", self.model_metrics['Random Forest']['Accuracy'],self.model_metrics['Logistic Regression']['Accuracy']),
                ("F1-Score", self.model_metrics['Random Forest']['F1'], self.model_metrics['Logistic Regression']['F1']),
                ("Recall", self.model_metrics['Random Forest']['Recall'],self.model_metrics['Logistic Regression']['Recall']),
                ("Precision",self.model_metrics['Random Forest']['Precision'], self.model_metrics['Logistic Regression']['Precision'])]
            header_frame = ctk.CTkFrame(metrics_frame)
            header_frame.pack(fill="x", padx=20, pady=5)
            # Formatting the headers
            for i, header in enumerate(headers):
                ctk.CTkLabel(header_frame, text=header, font=("Arial", 14, "bold"), width=120 if i>0 else 150).pack(side="left", padx=5 if i>0 else 0)
            # Creating the rows for the metrics
            for metric_name, rf_val, lr_val in metrics:
                row_frame = ctk.CTkFrame(metrics_frame)
                row_frame.pack(fill="x", padx=20, pady=2)
                # Metric name label
                ctk.CTkLabel(row_frame, text=metric_name, font=("Arial", 14), width=150, anchor="w").pack(side="left")
                # Best model check
                is_better = (rf_val > lr_val)
                # Comparing the values and setting the colors (hence the blue and gray)
                for val in [rf_val, lr_val]:
                    color = "#4e8cff" if (val == rf_val and is_better) or (val == lr_val and not is_better) else ("gray30", "gray70")
                    if isinstance(val, float):
                        display_val = f"{val:.3f}" if val >= 0.001 else str(val)
                    else:
                        display_val = str(val)
                    ctk.CTkLabel(row_frame, text=display_val, font=("Arial", 14), width=120, text_color=color).pack(side="left", padx=5)
            # Creating the ROC curve frame
            roc_frame = ctk.CTkFrame(main_frame)
            roc_frame.pack(fill="both", expand=True, padx=10, pady=10)
            # Matplotlib figure for the ROC curve
            self.current_figure = plt.Figure(figsize=(10, 6))
            ax = self.current_figure.add_subplot(111)
            # Claculating the ROC curve for both models (Random Forest and Logistic Regression)
            fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.rf_probs)
            roc_auc_rf = auc(fpr_rf, tpr_rf)
            fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.lr_probs)
            roc_auc_lr = auc(fpr_lr, tpr_lr)
            # Plotting the ROC curves and a random line 
            ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
            ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve Comparison')
            ax.legend(loc='lower right')
            # Drawing the canvas with Tkinter
            self.canvas = FigureCanvasTkAgg(self.current_figure, master=roc_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing model performance: {e}")
            self.cleanup_previous_tab()
    # Confusion matrix tab
    def show_confusion_matrix(self):
        if not self.content_frame.winfo_exists():
            return 
        try:
            main_frame = ctk.CTkFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            # Frame for the confusion matrix
            self.current_figure = plt.Figure(figsize=(16, 6))
            ax1 = self.current_figure.add_subplot(121)
            ax2 = self.current_figure.add_subplot(122)
            # Creating the confusion matrix headmap using seaborn for both models
            # Random Forest
            cm_rf = self.model_metrics['Random Forest']['Confusion Matrix']
            sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16, "weight": "bold"}, cbar=False, ax=ax1)
            ax1.set_title("Random Forest Confusion Matrix", fontsize=14, pad=15)
            ax1.set_xlabel("Predicted", fontsize=12)
            ax1.set_ylabel("Actual", fontsize=12)
            ax1.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=10)
            ax1.set_yticklabels(["No Diabetes", "Diabetes"], fontsize=10, rotation=0)
            # Logistic Regression
            cm_lr = self.model_metrics['Logistic Regression']['Confusion Matrix']
            sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens", annot_kws={"size": 16, "weight": "bold"}, cbar=False, ax=ax2)
            ax2.set_title("Logistic Regression Confusion Matrix", fontsize=14, pad=15)
            ax2.set_xlabel("Predicted", fontsize=12)
            ax2.set_ylabel("Actual", fontsize=12)
            ax2.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=10)
            ax2.set_yticklabels(["No Diabetes", "Diabetes"], fontsize=10, rotation=0)
            # Adjusting the layout of the figure
            self.current_figure.tight_layout()

            if main_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        except Exception as e:
            print(f"Error showing confusion matrix: {e}")
            self.cleanup_previous_tab()
    # Conclusions tab
    def show_conclusions(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            # Frame for the conclusions
            conclusions_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            conclusions_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(conclusions_frame, text="CONCLUSIONS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            # Just the conclusions we made with the results 
            conclusions = [
                f"1. {self.better_model} performed best with {self.better_accuracy*100:.1f}% accuracy",
                "2. HbA1c levels and blood glucose are the most critical predictors of diabetes",
                "3. The model could be improved with more lifestyle factors and genetic data" ]
            for conc in conclusions:
                ctk.CTkLabel(conclusions_frame, text=conc, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=5)
            # Frame for the recommendations
            rec_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            rec_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(rec_frame, text="RECOMMENDATIONS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            # The recommendations I found out that would help
            recommendations = [
                "• Collect more data on patients' lifestyle and genetic factors",
                "• Include continuous glucose monitoring data for better prediction",]
            for rec in recommendations:
                ctk.CTkLabel(rec_frame, text=rec, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)
            
        except Exception as e:
            print(f"Error showing conclusions: {e}")
            self.cleanup_previous_tab()
    # Closing the app
    def on_close(self):
        try:
            # It closes the app and cleans up the memory, destroying every window
            for after_id in self._after_ids:
                try:
                    self.root.after_cancel(after_id)
                except:
                    pass
            self.cleanup_previous_tab()
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
                
        except Exception as e:
            print(f"Error during close: {e}")
        finally:
            sys.exit(0)
# Main function to run the app using Tkinter
def main():
    try:
        root = ctk.CTk()
        # Properly closing the app
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