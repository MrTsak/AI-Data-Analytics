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

class DiabetesPredictorApp:
    # __init_ method to initialize the app
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction Dashboard")
        self.root.geometry("1400x1000") 
        self.current_figure = None
        self.canvas = None
        self.content_frame = None
        self.theme_mode = "light"
        self._after_ids = []
        
        # Initializing variables
        self.df = None
        self.df_vis = None 
        self.rf_model = None
        self.lr_model = None
        self.better_model = None
        self.better_accuracy = None
        self.model_metrics = None
        self.cluster_centers = None
        
        # Setting the theme
        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")
        
        # Running the main functions
        self.load_data()     
        self.train_models()       
        self.perform_clustering()
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    # load_data method to load and preprocess the data
    def load_data(self):
        try:
            self.df = pd.read_csv('data/sample.csv')
            # Create mapping for smoking history
            self.smoking_mapping = {
                'never': (0, 'Never smoked'),
                'No Info': (1, 'No info'),
                'current': (2, 'Current smoker'),
                'former': (3, 'Former smoker'),
                'ever': (4, 'Ever smoked'),
                'not current': (5, 'Not current smoker')
            }
            
            # Create numeric version for modeling
            self.df['smoking_history_num'] = self.df['smoking_history'].map(lambda x: self.smoking_mapping.get(x, (1, 'No info'))[0])
            self.df['gender_num'] = self.df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
            
            # Create a visualization DataFrame
            self.df_vis = self.df.copy()
            self.df_vis['smoking_history'] = self.df['smoking_history'].map(lambda x: self.smoking_mapping.get(x, (1, 'No info'))[1])
            self.df_vis['gender'] = self.df['gender'].map({'Female': 'Female', 'Male': 'Male', 'Other': 'Other'})
            
            # Fill missing values with mean
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            self.df_vis[numeric_cols] = self.df_vis[numeric_cols].fillna(self.df_vis[numeric_cols].mean())
            
            # Print only the first few rows and summary info
            print("Data Sample (first 5 rows):")
            print(self.df_vis.head())
            print("\nData Information:")
            print(self.df_vis.info())
            print("\nDescriptive Statistics:")
            print(self.df_vis.describe(include='all').round(2))
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def train_models(self):
        try:
            # Prepare features and target variable
            features = self.df.drop(columns=['diabetes', 'Cluster', 'gender', 'smoking_history'], errors='ignore')
            target = self.df['diabetes'] 
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
        
            # Random Forest Classifier
            self.rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=7,
                min_samples_split=8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1)
            
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
            
            self.lr_model.fit(self.X_train, self.y_train)
            self.lr_pred = self.lr_model.predict(self.X_test)
            self.lr_probs = self.lr_model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
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

            self.better_accuracy = max(rf_accuracy, lr_accuracy)
            # Store model metrics
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

    def perform_clustering(self):
        try:
            # Use numeric version for clustering
            cluster_features = self.df[['HbA1c_level', 'bmi', 'age']] 
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_features)  
            kmeans = KMeans(n_clusters=3, random_state=42)
            self.df['Cluster'] = kmeans.fit_predict(scaled_features)
            self.df_vis['Cluster'] = self.df['Cluster'] 
            self.cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

        except Exception as e:
            print(f"Error in clustering: {e}")
            sys.exit(1)

    def setup_ui(self):
        # Main frame
        self.header_frame = ctk.CTkFrame(self.root, height=80)
        self.header_frame.pack(fill="x", padx=30, pady=(30, 20))
        # Header label
        ctk.CTkLabel(self.header_frame, text="Diabetes Prediction Analysis", font=("Arial", 20, "bold")).pack(side="left", padx=25)
        # Theme changing feature
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
            "Live Prediction": self.tabview.add("Live Prediction"),
            "Conclusions": self.tabview.add("Conclusions")}      
        # Tab update
        self.safe_update_tab("Data Overview") 
        self.tabview.configure(command=lambda: self.on_tab_change())
    # Theme toggle
    def toggle_theme(self):
        try:
            current_tab = self.tabview.get()
            self.theme_mode = "dark" if self.theme_mode == "light" else "light"
            ctk.set_appearance_mode(self.theme_mode)
            self.theme_button.configure(text="🌙 Dark Mode" if self.theme_mode == "light" else "☀️ Light Mode")
            self.safe_update_tab(current_tab)
        except Exception as e:
            print(f"Error in theme toggle: {e}")
    # Update the tab when it changes
    def on_tab_change(self):
        try:
            current_tab = self.tabview.get()
            if current_tab in self.tabs:
                self.safe_update_tab(current_tab)
        except Exception as e:
            print(f"Error in tab change: {e}")
    # Safe update method to ensure the tab updates correctly
    def safe_update_tab(self, tab_name):
        for after_id in self._after_ids:
            self.root.after_cancel(after_id)
        self._after_ids = []
        self._after_ids.append(self.root.after(50, lambda: self._update_tab_content(tab_name)))
    
    # Update the content of the selected tab
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
            elif tab_name == "Live Prediction":
                self.show_live_prediction()
            elif tab_name == "Conclusions":
                self.show_conclusions()
            
        except Exception as e:
            print(f"Error updating tab {tab_name}: {e}")
            self.cleanup_previous_tab()
    # Cleanup previous tab content
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
    # Show data overview with visualizations
    def show_data_overview(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            stats_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            stats_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            ctk.CTkLabel(stats_frame, text="BASIC STATISTICS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            columns_to_plot = [col for col in self.df_vis.columns if col not in ['diabetes', 'Cluster', 'smoking_history_num', 'gender_num']]
            
            num_plots = len(columns_to_plot) 
            rows = (num_plots + 2) // 3
            cols = 3
            # Create a new figure for the plots
            self.current_figure = plt.Figure(figsize=(14, 5*rows))
            axes = self.current_figure.subplots(rows, cols)
            self.current_figure.tight_layout(pad=4.0)

            if rows > 1:
                axes = axes.flatten()
            # If there's only one row, axes is a single Axes object    
            for i, col in enumerate(columns_to_plot):
                ax = axes[i]
                
                if col == 'gender':
                    # Filter only Male and Female for gender visualization
                    gender_data = self.df_vis[self.df_vis['gender'].isin(['Male', 'Female'])]
                    value_counts = gender_data['gender'].value_counts()
                    value_counts.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title("Gender Distribution (Male/Female)", fontsize=10)
                    ax.tick_params(axis='x', rotation=0)
                    
                elif col in ['hypertension', 'heart_disease']:
                    # Convert to 0/1 and plot counts
                    value_counts = self.df_vis[col].astype(int).value_counts().sort_index()
                    value_counts.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontsize=10)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['0', '1'], rotation=0)
                    
                elif self.df_vis[col].dtype == 'object' or col == 'smoking_history':
                    # For other categorical variables
                    value_counts = self.df_vis[col].value_counts()
                    if len(value_counts) > 10:
                        value_counts = value_counts.nlargest(10)
                        value_counts.plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_title(f"Top 10 {col.replace('_', ' ').title()} Categories", fontsize=10)
                    else:
                        value_counts.plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontsize=10)
                        ax.tick_params(axis='x', rotation=45)
                else:
                    # For numerical variables
                    sns.histplot(data=self.df_vis, x=col, kde=True, ax=ax, color='skyblue')
                    ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontsize=10)
                
                ax.set_xlabel("")
                ax.grid(True, linestyle='--', alpha=0.7)
            
            for j in range(i+1, rows*cols):
                axes[j].axis('off')
                
            self.current_figure.tight_layout()
            
            if self.content_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing data overview: {e}")
            self.cleanup_previous_tab()

    def show_correlation(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Frame for hypotheses
            hypotheses_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            hypotheses_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(hypotheses_frame, text="HYPOTHESES", font=("Arial", 16, "bold")).pack(pady=(10, 5))
            
            hypotheses = [
                ("1. Higher HbA1c levels correlate with higher diabetes risk", 
                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC4933534/#:~:text=HbA1c%20not%20only%20provides%20a,subjects%20with%20or%20without%20diabetes."),
                ("2. BMI is positively correlated with diabetes risk", 
                 "https://www.escardio.org/The-ESC/Press-Office/Press-releases/Body-mass-index-is-a-more-powerful-risk-factor-for-diabetes-than-genetics#:~:text=Those%20in%20the%20highest%20BMI,groups%2C%20regardless%20of%20genetic%20risk."),
                ("3. Age is a significant factor in diabetes risk", 
                 "https://www.thelancet.com/journals/lanhl/article/PIIS2666-7568(21)00177-X/fulltext"),
                ("4. Blood glucose levels show strong correlation with diabetes", 
                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC4484145/")]
            
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
            self.current_figure = plt.Figure(figsize=(12, 10))
            ax = self.current_figure.add_subplot(111)
            
            # Use numeric version for correlation
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            corr = self.df[numeric_cols].corr()
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            
            # Improved colormap and annotations
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            
            # Seaborn heatmap with improved styling
            sns.heatmap(corr, annot=True, cmap=cmap, center=0,  fmt=".2f", linewidths=0.5, square=True, ax=ax,  mask=mask, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
            
            # Setting the title and the names of the heatmap
            ax.set_title("Feature Correlations with Diabetes Outcome", fontsize=14, pad=20)
            ax.set_xticklabels([label.replace('_', ' ').title() for label in corr.columns], rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels([label.replace('_', ' ').title() for label in corr.columns], rotation=0, fontsize=10)
            
            # Add a note about correlation interpretation
            note_frame = ctk.CTkFrame(main_frame)
            note_frame.pack(fill="x", padx=10, pady=(0, 10))
            ctk.CTkLabel(note_frame, text="Note: Correlation values range from -1 (perfect negative) to +1 (perfect positive). Values close to 0 indicate no correlation.", font=("Arial", 11), wraplength=1200).pack(padx=10, pady=5)
    
            if plot_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

        except Exception as e:
            print(f"Error showing correlation: {e}")
            self.cleanup_previous_tab()
    # Clustering function
    def show_clustering(self):
        if not self.content_frame.winfo_exists():
            return   
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            info_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            info_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(info_frame, text="CLUSTER ANALYSIS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            self.current_figure = plt.Figure(figsize=(14, 12))
            axes = self.current_figure.subplots(2, 2)
            self.current_figure.tight_layout(pad=4.0)
            
            #  Scatter plot for HbA1c_level vs bmi by cluster with improved styling
            ax = axes[0, 0]
            scatter = sns.scatterplot(data=self.df_vis, x='HbA1c_level', y='bmi', hue='Cluster', palette='viridis', ax=ax, s=60, alpha=0.7)
            ax.set_title("HbA1c vs BMI by Cluster", fontsize=12)
            ax.set_xlabel("HbA1c Level (%)", fontsize=10)
            ax.set_ylabel("Body Mass Index (BMI)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(title='Cluster', fontsize=9, title_fontsize=10)
            
            # Cluster centers for HbA1c_level, bmi and age with improved styling
            ax = axes[0, 1]
            centers_df = pd.DataFrame(self.cluster_centers, columns=['HbA1c_level', 'bmi', 'age'])
            centers_df['Cluster'] = range(3)
            melted_centers = centers_df.melt(id_vars='Cluster',  var_name='Feature', value_name='Value')
            
            # Rename features for better display
            feature_names = {'HbA1c_level': 'HbA1c Level','bmi': 'BMI','age': 'Age'
            }
            melted_centers['Feature'] = melted_centers['Feature'].map(feature_names)
            
            bar = sns.barplot(data=melted_centers, x='Cluster', y='Value', hue='Feature', ax=ax, palette='Set2')
            ax.set_title("Average Feature Values by Cluster", fontsize=12)
            ax.set_xlabel("Cluster", fontsize=10)
            ax.set_ylabel("Average Value", fontsize=10)
            ax.legend(title='Feature', fontsize=9, title_fontsize=10)
            
            # Diabetes prevalence by cluster with improved styling
            ax = axes[1, 0]
            cluster_outcome = self.df_vis.groupby('Cluster')['diabetes'].mean().reset_index()
            cluster_outcome['diabetes_percentage'] = cluster_outcome['diabetes'] * 100
            
            bar = sns.barplot(data=cluster_outcome, x='Cluster', y='diabetes_percentage', hue='Cluster', palette='coolwarm', legend=False, ax=ax)
            ax.set_title("Diabetes Prevalence by Cluster", fontsize=12)
            ax.set_xlabel("Cluster", fontsize=10)
            ax.set_ylabel("Diabetes Rate (%)", fontsize=10)
            
            # Add percentage labels on top of bars
            for p in bar.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 1,f'{height:.1f}%', ha="center", fontsize=10)
            
            # Boxplot for Age distribution by cluster with improved styling
            ax = axes[1, 1]
            box = sns.boxplot(data=self.df_vis, x='Cluster', y='age', hue='Cluster', palette='pastel', legend=False, ax=ax)
            ax.set_title("Age Distribution by Cluster", fontsize=12)
            ax.set_xlabel("Cluster", fontsize=10)
            ax.set_ylabel("Age (Years)", fontsize=10)
            
            # Add median labels
            medians = self.df_vis.groupby('Cluster')['age'].median().values
            vertical_offset = self.df_vis['age'].median() * 0.05
            
            for xtick in box.get_xticks():
                box.text(xtick, medians[xtick] + vertical_offset, f'{medians[xtick]:.1f}', horizontalalignment='center', size='small', color='black', weight='semibold')
            
            # Adjust layout
            self.current_figure.tight_layout()
            
            # Add cluster description frame
            desc_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            desc_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(desc_frame, text="CLUSTER INTERPRETATION", font=("Arial", 14, "bold")).pack(pady=(10,5))
            
            # Add cluster descriptions
            interpretations = [
                "• Cluster 0: Younger individuals with lower HbA1c and normal BMI",
                "• Cluster 1: Middle-aged individuals with moderate HbA1c and elevated BMI",
                "• Cluster 2: Older individuals with high HbA1c and high BMI"
            ]
            
            for interpretation in interpretations:
                ctk.CTkLabel(desc_frame, text=interpretation, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)
            
            if main_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing clustering: {e}")
            self.cleanup_previous_tab()

    def show_model_performance(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Best model frame
            summary_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12, border_color="#4e8cff", fg_color=("#f0f0f0", "#2b2b2b"))
            summary_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            ctk.CTkLabel(summary_frame, text=f"BEST MODEL: {self.better_model} (Accuracy: {self.better_accuracy*100:.1f}%)", font=("Arial", 18, "bold"), text_color="#4e8cff").pack(pady=(5, 10))
            
            # Model description frame
            desc_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            desc_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(desc_frame, text="MODEL DESCRIPTIONS", font=("Arial", 14, "bold")).pack(pady=(5, 10))
            
            model_descriptions = [
                ("Random Forest:", "An ensemble method that builds multiple decision trees and merges them for more accurate predictions."),
                ("Logistic Regression:", "A linear model for classification that predicts probabilities using a logistic function.")
            ]
            
            for model_name, description in model_descriptions:
                row_frame = ctk.CTkFrame(desc_frame, fg_color="transparent")
                row_frame.pack(fill="x", padx=20, pady=2)
                ctk.CTkLabel(row_frame, text=model_name, font=("Arial", 12, "bold"), width=180, anchor="w").pack(side="left")
                ctk.CTkLabel(row_frame, text=description, font=("Arial", 12), wraplength=1000, anchor="w").pack(side="left", fill="x", expand=True)
            
            # Metrics comparison frame
            metrics_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            metrics_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            ctk.CTkLabel(metrics_frame, text="DETAILED MODEL COMPARISON", font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            headers = ["Metric", "Random Forest", "Logistic Regression"]
            metrics = [
                ("Accuracy", self.model_metrics['Random Forest']['Accuracy'],
                 self.model_metrics['Logistic Regression']['Accuracy']),
                ("F1-Score", self.model_metrics['Random Forest']['F1'], 
                 self.model_metrics['Logistic Regression']['F1']),
                ("Recall", self.model_metrics['Random Forest']['Recall'],
                 self.model_metrics['Logistic Regression']['Recall']),
                ("Precision", self.model_metrics['Random Forest']['Precision'], 
                 self.model_metrics['Logistic Regression']['Precision'])
            ]
            
            # Create header row
            header_frame = ctk.CTkFrame(metrics_frame)
            header_frame.pack(fill="x", padx=20, pady=5)
            
            for i, header in enumerate(headers):
                ctk.CTkLabel(header_frame, text=header, font=("Arial", 14, "bold"), width=150 if i==0 else 180).pack(side="left", padx=5 if i>0 else 0)
            
            # Create metric rows
            for metric_name, rf_val, lr_val in metrics:
                row_frame = ctk.CTkFrame(metrics_frame)
                row_frame.pack(fill="x", padx=20, pady=2)
                
                # Metric name label
                ctk.CTkLabel(row_frame, text=metric_name, font=("Arial", 14), width=150, anchor="w").pack(side="left")
                
                # Best model check
                is_better = (rf_val > lr_val) if metric_name != "Recall" else (rf_val >= lr_val)
                
                # Comparing the values and setting the colors
                for i, val in enumerate([rf_val, lr_val]):
                    color = "#4e8cff" if (i == 0 and is_better) or (i == 1 and not is_better) else ("gray30", "gray70")
                    if isinstance(val, float):
                        display_val = f"{val:.3f}" if val >= 0.001 else str(val)
                    else:
                        display_val = str(val)
                    ctk.CTkLabel(row_frame, text=display_val, font=("Arial", 14), width=180, text_color=color).pack(side="left", padx=5)
            
            # ROC curve frame
            roc_frame = ctk.CTkFrame(main_frame)
            roc_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Matplotlib figure for ROC curve
            self.current_figure = plt.Figure(figsize=(10, 7))
            ax = self.current_figure.add_subplot(111)
            
            # Calculate ROC curves
            fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.rf_probs)
            roc_auc_rf = auc(fpr_rf, tpr_rf)
            fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.lr_probs)
            roc_auc_lr = auc(fpr_lr, tpr_lr)
            
            # Plot ROC curves
            ax.plot(fpr_rf, tpr_rf, color='#4e8cff', lw=2,  label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
            ax.plot(fpr_lr, tpr_lr, color='#2ecc71', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            
            # Customize plot
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Add AUC interpretation
            auc_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            auc_frame.pack(fill="x", padx=10, pady=(0, 10))
            ctk.CTkLabel(auc_frame, text="AUC (Area Under Curve) Interpretation: 0.9-1 = Excellent, 0.8-0.9 = Good, 0.7-0.8 = Fair, 0.6-0.7 = Poor, 0.5-0.6 = Fail", font=("Arial", 11)).pack(padx=10, pady=5)
            
            # Draw canvas
            self.canvas = FigureCanvasTkAgg(self.current_figure, master=roc_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing model performance: {e}")
            self.cleanup_previous_tab()

    def show_confusion_matrix(self):
        if not self.content_frame.winfo_exists():
            return 
        try:
            main_frame = ctk.CTkFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Frame for the confusion matrix
            self.current_figure = plt.Figure(figsize=(16, 7))
            ax1 = self.current_figure.add_subplot(121)
            ax2 = self.current_figure.add_subplot(122)
            
            # Creating the confusion matrix heatmap using seaborn for both models
            # Random Forest
            cm_rf = self.model_metrics['Random Forest']['Confusion Matrix']
            sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16, "weight": "bold"}, cbar=False, ax=ax1, linewidths=1, linecolor='white')
            ax1.set_title("Random Forest Confusion Matrix", fontsize=14, pad=15)
            ax1.set_xlabel("Predicted Label", fontsize=12)
            ax1.set_ylabel("True Label", fontsize=12)
            ax1.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=11)
            ax1.set_yticklabels(["No Diabetes", "Diabetes"], fontsize=11, rotation=0)
            
            # Add performance metrics to the plot
            rf_metrics = self.model_metrics['Random Forest']
            ax1.text(0.5, -0.25, f"Accuracy: {rf_metrics['Accuracy']:.3f} | Precision: {rf_metrics['Precision']:.3f}\nRecall: {rf_metrics['Recall']:.3f} | F1: {rf_metrics['F1']:.3f}", ha='center', va='center', transform=ax1.transAxes, fontsize=11)
            
            # Logistic Regression
            cm_lr = self.model_metrics['Logistic Regression']['Confusion Matrix']
            sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens", annot_kws={"size": 16, "weight": "bold"}, cbar=False, ax=ax2, linewidths=1, linecolor='white')
            ax2.set_title("Logistic Regression Confusion Matrix", fontsize=14, pad=15)
            ax2.set_xlabel("Predicted Label", fontsize=12)
            ax2.set_ylabel("True Label", fontsize=12)
            ax2.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=11)
            ax2.set_yticklabels(["No Diabetes", "Diabetes"], fontsize=11, rotation=0)
            
            lr_metrics = self.model_metrics['Logistic Regression']
            ax2.text(0.5, -0.25,  f"Accuracy: {lr_metrics['Accuracy']:.3f} | Precision: {lr_metrics['Precision']:.3f}\nRecall: {lr_metrics['Recall']:.3f} | F1: {lr_metrics['F1']:.3f}",ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            
            # Add note
            note_frame = ctk.CTkFrame(main_frame)
            note_frame.pack(fill="x", padx=10, pady=(0, 10))
            ctk.CTkLabel(note_frame, text="Confusion Matrix Interpretation: Diagonal elements (top-left to bottom-right) show correct predictions. Off-diagonal elements show misclassifications.",font=("Arial", 11), wraplength=1200).pack(padx=10, pady=5)
            
            # Adjusting the layout of the figure
            self.current_figure.tight_layout()

            if main_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        except Exception as e:
            print(f"Error showing confusion matrix: {e}")
            self.cleanup_previous_tab()

    # show live prediction tab
    def show_live_prediction(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)

            # Title frame
            title_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            title_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(title_frame, text="LIVE DIABETES PREDICTION", font=("Arial", 16, "bold")).pack(pady=(10,5))

            # Model selection
            model_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            model_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(model_frame, text="Select Prediction Model:", font=("Arial", 14)).pack(pady=(5,10))

            self.prediction_model_var = ctk.StringVar(value="Random Forest")
            model_options = ["Random Forest", "Logistic Regression"]

            for i, option in enumerate(model_options):
                rb = ctk.CTkRadioButton(model_frame, text=option, variable=self.prediction_model_var, value=option, font=("Arial", 12))
                rb.pack(pady=5, padx=20, anchor="w")

            # Input features frame
            input_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            input_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(input_frame, text="Enter Patient Details:", font=("Arial", 14)).pack(pady=(5,10))

            # Get feature ranges from data
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.feature_widgets = {}

            # Create input widgets for each feature
            for col in numeric_cols:
                if col in ['diabetes', 'Cluster', 'smoking_history_num', 'gender_num']:
                    continue
                    
                slider_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
                slider_frame.pack(fill="x", padx=20, pady=5)

                label = col.replace('_', ' ').title()
                ctk.CTkLabel(slider_frame, text=f"{label}:", width=150, anchor="w").pack(side="left")

                # Setting up all the sliders and checkboxes
                if col in ['hypertension', 'heart_disease']:
                    var = ctk.IntVar(value=0)
                    checkbox = ctk.CTkCheckBox(slider_frame, text="", variable=var, onvalue=1, offvalue=0)
                    checkbox.pack(side="left", padx=10)
                    self.feature_widgets[col] = ('checkbox', var)

                elif col == 'age':

                    min_val = int(self.df[col].min())
                    max_val = int(self.df[col].max())
                    mean_val = int(self.df[col].mean())

                    slider = ctk.CTkSlider(slider_frame, from_= 18,  to=max_val, number_of_steps=max_val-min_val)
                    slider.set(mean_val)
                    slider.pack(side="left", expand=True, fill="x", padx=10)

                    value_label = ctk.CTkLabel(slider_frame, text=f"{mean_val}", width=60)
                    value_label.pack(side="left")

                    slider.bind("<B1-Motion>", lambda e, lbl=value_label, s=slider: lbl.configure(text=f"{int(s.get())}"))
                    self.feature_widgets[col] = ('slider', slider, value_label)

                elif col == 'HbA1c_level':

                    min_val = float(self.df[col].min())
                    max_val = float(self.df[col].max())
                    mean_val = float(self.df[col].mean())
                    step = 0.1

                    slider = ctk.CTkSlider(slider_frame, from_=min_val, to=max_val, number_of_steps=int((max_val-min_val)/step))
                    slider.set(mean_val)
                    slider.pack(side="left", expand=True, fill="x", padx=10)

                    value_label = ctk.CTkLabel(slider_frame, text=f"{mean_val:.1f}", width=60)
                    value_label.pack(side="left")

                    slider.bind("<B1-Motion>", lambda e, lbl=value_label, s=slider: lbl.configure(text=f"{s.get():.1f}"))
                    self.feature_widgets[col] = ('slider', slider, value_label)

                else:

                    min_val = float(self.df[col].min())
                    max_val = float(self.df[col].max())
                    mean_val = float(self.df[col].mean())

                    slider = ctk.CTkSlider(slider_frame, from_=min_val, to=max_val, number_of_steps=100)
                    slider.set(mean_val)
                    slider.pack(side="left", expand=True, fill="x", padx=10)
                    
                    value_label = ctk.CTkLabel(slider_frame, text=f"{mean_val:.1f}", width=60)
                    value_label.pack(side="left")
                
                    slider.bind("<B1-Motion>", lambda e, lbl=value_label, s=slider: lbl.configure(text=f"{s.get():.1f}"))
                    self.feature_widgets[col] = ('slider', slider, value_label)
        
            # A dropdown for smoking history
            smoking_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
            smoking_frame.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(smoking_frame, text="Smoking History:", width=150, anchor="w").pack(side="left")

            smoking_options = [desc for (num, desc) in sorted(self.smoking_mapping.values(), key=lambda x: x[0])]
            self.smoking_var = ctk.StringVar(value=smoking_options[1])  # Default to "No info"

            smoking_dropdown = ctk.CTkComboBox(smoking_frame, values=smoking_options, variable=self.smoking_var, width=200)
            smoking_dropdown.pack(side="left", padx=10)

            # Prediction button
            predict_button = ctk.CTkButton(main_frame, text="PREDICT DIABETES RISK",   command=self.make_prediction,  font=("Arial", 14, "bold"),  height=40)
            predict_button.pack(pady=20)

            # Result frame
            self.result_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            self.result_frame.pack(fill="x", padx=10, pady=10)

            # Makes it invisible at first
            self.result_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 14))
            self.result_label.pack(pady=20)

            self.probability_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 14))
            self.probability_label.pack(pady=10)

            self.interpretation_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 14))

        except Exception as e:
            print(f"Error showing live prediction: {e}")
            self.cleanup_previous_tab()

    # Function to make predictions based on user input
    def make_prediction(self):
        try:
            # Get selected model
            model_name = self.prediction_model_var.get()
            model = self.rf_model if model_name == "Random Forest" else self.lr_model

            # Prepare input data
            input_data = {}
            for col, widget_info in self.feature_widgets.items():
                widget_type = widget_info[0]

                if widget_type == 'checkbox':
                    input_data[col] = widget_info[1].get()
                elif widget_type == 'slider':
                    value = widget_info[1].get()
                    if col == 'age':
                        input_data[col] = int(value)
                    else:
                        input_data[col] = float(value)

            # Add smoking history
            smoking_num = next(num for num, desc in self.smoking_mapping.values()  if desc == self.smoking_var.get())
            input_data['smoking_history_num'] = smoking_num

            # Add gender
            input_data['gender_num'] = 1

            # Create DataFrame with same column order as training data
            input_df = pd.DataFrame([input_data], columns=self.X_train.columns)
             
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            # Update result display
            result_text = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
            color = "#FF6347" if prediction == 1 else "#2E8B57"  # Red for positive, green for negative

            self.result_label.configure(text=f"Prediction: {result_text}", text_color=color,font=("Arial", 16, "bold"))

            # Color probability based on risk level
            if probability > 0.7:
                prob_color = "#FF6347"
                risk_level = "High"
            elif probability > 0.4:
                prob_color = "#FFA500"
                risk_level = "Moderate"
            else:
                prob_color = "#2E8B57"
                risk_level = "Low"

            self.probability_label.configure(text=f"Probability: {probability*100:.1f}%", text_color=prob_color, font=("Arial", 14))

            # Update interpretation
            if hasattr(self, 'interpretation_label') and self.interpretation_label.winfo_exists():
                self.interpretation_label.pack_forget()

            self.interpretation_label = ctk.CTkLabel(self.result_frame, text=f"Risk Level: {risk_level}", font=("Arial", 14), text_color=prob_color)
            self.interpretation_label.pack(pady=10)

        except Exception as e:
            print(f"Error making prediction: {e}")
    # Conclusions function
    def show_conclusions(self):
        if not self.content_frame.winfo_exists():
            return
        try:
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Frame for the conclusions
            conclusions_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            conclusions_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(conclusions_frame, text="KEY FINDINGS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            # Just the conclusions we made with the results 
            findings = [
                f"1. {self.better_model} performed best with {self.better_accuracy*100:.1f}% accuracy",
                "2. HbA1c levels and blood glucose are the most critical predictors of diabetes",
                "3. Cluster analysis revealed distinct patient groups with different risk profiles",
                "4. The model could be improved with more lifestyle factors and genetic data"]
            
            for finding in findings:
                ctk.CTkLabel(conclusions_frame, text=finding, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=5)
            
            # Frame for the recommendations
            rec_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            rec_frame.pack(fill="x", padx=10, pady=10)
            ctk.CTkLabel(rec_frame, text="RECOMMENDATIONS", font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            # The recommendations I found out that would help
            recommendations = [
                "• Collect more data on patients' lifestyle and genetic factors",
                "• Include continuous glucose monitoring data for better prediction",
                "• Consider implementing regular HbA1c screening for high-risk clusters",
                "• Develop targeted interventions for different risk groups identified by clustering"]
            
            for rec in recommendations:
                ctk.CTkLabel(rec_frame, text=rec, font=("Arial", 12), anchor="w").pack(fill="x", padx=20, pady=2)
  
        except Exception as e:
            print(f"Error showing conclusions: {e}")
            self.cleanup_previous_tab()

    def on_close(self):
        try:
            # Clean up resources
            for after_id in self._after_ids:
                try:
                    self.root.after_cancel(after_id)
                except:
                    pass
            self.cleanup_previous_tab()
            
            # Destroy the window if it exists
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                try:
                    self.root.destroy()
                except:
                    pass
                
        except Exception as e:
            print(f"Error during close: {e}")
        finally:
            sys.exit(0)

def main():
    try:
        root = ctk.CTk()
        app = DiabetesPredictorApp(root)
        
        def on_closing():
            try:
                app.on_close()
            except Exception as e:
                print(f"Error during closing: {e}")
                try:
                    root.destroy()
                except:
                    pass
                sys.exit(0)
                
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
                
    except Exception as e:
        print(f"Application startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()