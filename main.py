import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import gc

# Configure seaborn
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
        self._after_ids = []  # Track pending after events
        
        # Configure appearance
        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model()
        
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
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def train_model(self):
        """Train the machine learning model"""
        try:
            features = self.df.drop(columns=['Outcome'])
            target = self.df['Outcome']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features, target, test_size=0.2, random_state=42)
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
        except Exception as e:
            print(f"Error training model: {e}")
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
            "Correlation": self.tabview.add("Correlation"),
            "Performance": self.tabview.add("Performance"),
            "Confusion": self.tabview.add("Confusion"),
            "Features": self.tabview.add("Features")
        }
        
        # Initialize first tab
        self.safe_update_tab("Correlation")
        
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
            
            if tab_name == "Correlation":
                self.show_correlation()
            elif tab_name == "Performance":
                self.show_performance()
            elif tab_name == "Confusion":
                self.show_confusion_matrix()
            elif tab_name == "Features":
                self.show_feature_importance()
                
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

    def show_correlation(self):
        """Display correlation heatmap"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            self.current_figure = plt.Figure(figsize=(11, 9))
            ax = self.current_figure.add_subplot(111)
            
            corr = self.df.corr()
            
            # Create mask only for upper triangle (excluding diagonal)
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            
            # Create heatmap
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
                annot_kws={
                    "size": 10,
                    "ha": "center",
                    "va": "center"
                },
                cbar_kws={
                    "shrink": 0.8,
                    "label": "Correlation Coefficient"
                }
            )
            
            # Highlight Outcome column
            outcome_idx = list(corr.columns).index('Outcome')
            for i in range(len(corr.columns)):
                if i != outcome_idx:
                    ax.text(outcome_idx + 0.5, i + 0.5, f"{corr.iloc[i, outcome_idx]:.2f}",
                           ha="center", va="center", fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Improve labels
            ax.set_xticklabels(
                corr.columns,
                rotation=45,
                ha='right',
                rotation_mode='anchor',
                fontsize=11
            )
            ax.set_yticklabels(
                corr.columns,
                rotation=0,
                fontsize=11
            )
            
            ax.set_title(
                "Feature Correlations with Diabetes Outcome\n" +
                "(Lower triangle shows correlations with Outcome)",
                fontsize=14,
                pad=20
            )
            
            self.current_figure.tight_layout()
            
            # Create canvas only if frame still exists
            if self.content_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=self.content_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
            
        except Exception as e:
            print(f"Error showing correlation: {e}")
            self.cleanup_previous_tab()

    def show_performance(self):
        """Display model performance metrics"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            report = classification_report(self.y_test, self.y_pred, output_dict=True)
            
            main_frame = ctk.CTkScrollableFrame(self.content_frame)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Accuracy display
            acc_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12,
                                    border_color="#4e8cff", fg_color=("#f0f0f0", "#2b2b2b"))
            acc_frame.pack(fill="x", padx=10, pady=10, ipady=15)
            
            ctk.CTkLabel(acc_frame, text="MODEL ACCURACY", 
                        font=("Arial", 16, "bold")).pack(pady=(5,0))
            ctk.CTkLabel(acc_frame, text=f"{accuracy*100:.2f}%", 
                        font=("Arial", 32, "bold"), 
                        text_color="#4e8cff").pack(pady=(0,10))
            
            # Classification metrics
            metrics_frame = ctk.CTkFrame(main_frame, border_width=2, corner_radius=12)
            metrics_frame.pack(fill="x", padx=10, pady=10, ipady=10)
            
            ctk.CTkLabel(metrics_frame, text="CLASSIFICATION METRICS", 
                        font=("Arial", 16, "bold")).pack(pady=(10,5))
            
            # Create a grid for metrics
            for i, (label, metric) in enumerate(report['1'].items()):
                if label in ['precision', 'recall', 'f1-score']:
                    metric_frame = ctk.CTkFrame(metrics_frame)
                    metric_frame.pack(fill="x", padx=20, pady=5)
                    
                    ctk.CTkLabel(metric_frame, text=label.title(), 
                                font=("Arial", 14), width=120, anchor="w").pack(side="left")
                    ctk.CTkLabel(metric_frame, text=f"{metric:.2f}", 
                                font=("Arial", 14, "bold")).pack(side="right")
            
        except Exception as e:
            print(f"Error showing performance: {e}")
            self.cleanup_previous_tab()

    def show_confusion_matrix(self):
        """Display confusion matrix"""
        if not self.content_frame.winfo_exists():
            return
            
        try:
            self.current_figure = plt.Figure(figsize=(8, 6))
            ax = self.current_figure.add_subplot(111)
            
            cm = confusion_matrix(self.y_test, self.y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=["No Diabetes", "Diabetes"], 
                       yticklabels=["No Diabetes", "Diabetes"], 
                       ax=ax, annot_kws={"size": 16})
            
            ax.set_title("Confusion Matrix", fontsize=16, pad=20)
            ax.set_xlabel("Predicted", fontsize=14)
            ax.set_ylabel("Actual", fontsize=14)
            
            if self.content_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=self.content_frame)
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
            self.current_figure = plt.Figure(figsize=(10, 6))
            ax = self.current_figure.add_subplot(111)
            
            importance = self.model.feature_importances_
            features = self.df.drop(columns=['Outcome']).columns
            
            sns.barplot(x=importance, y=features, color="#4e8cff", ax=ax)
            ax.set_title("Feature Importance Scores", fontsize=16, pad=20)
            ax.set_xlabel("Importance Score", fontsize=14)
            ax.set_ylabel("Features", fontsize=14)
            
            self.current_figure.tight_layout()
            
            if self.content_frame.winfo_exists():
                self.canvas = FigureCanvasTkAgg(self.current_figure, master=self.content_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
            
        except Exception as e:
            print(f"Error showing feature importance: {e}")
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
        
        # Run with error handling
        while True:
            try:
                root.mainloop()
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                if 'app' in locals():
                    app.on_close()
                break
                
    except Exception as e:
        print(f"Application startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()