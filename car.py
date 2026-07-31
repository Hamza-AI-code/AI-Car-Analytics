import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from PIL import Image, ImageTk

class AdvancedAIAnalyst:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Data Analyst Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Style moderne
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white')
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        
        # Variables de contrôle
        self.n_points = tk.IntVar(value=250)
        self.n_clusters = tk.IntVar(value=4)
        self.selected_x = tk.StringVar(value="Prix")
        self.selected_y = tk.StringVar(value="Kilométrage")
        self.selected_z = tk.StringVar(value="Age")
        self.panel_width = tk.IntVar(value=350)
        self.min_values = {key: tk.DoubleVar(value=0) for key in ['Prix', 'Kilométrage', 'Age', 'Consommation', 'Puissance', 'Valeur']}
        self.max_values = {key: tk.DoubleVar(value=100) for key in ['Prix', 'Kilométrage', 'Age', 'Consommation', 'Puissance', 'Valeur']}
        self.selected_inputs = tk.StringVar(value="Prix,Kilométrage")
        self.selected_output = tk.StringVar(value="Valeur")
        self.arima_p = tk.IntVar(value=1)
        self.arima_q = tk.IntVar(value=1)
        self.arima_d = tk.IntVar(value=1)
        
        # Ajouter des traces pour les variables de sélection
        self.selected_x.trace('w', self.update_plots)
        self.selected_y.trace('w', self.update_plots)
        self.selected_z.trace('w', self.update_plots)
        self.selected_inputs.trace('w', self.update_plots)
        self.selected_output.trace('w', self.update_plots)
        
        # Palette de couleurs moderne
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'dark': '#2c3e50',
            'light': '#ecf0f1'
        }
        
        # Génération des données
        self.generate_data()
        
        # Création de l'interface
        self.create_welcome_screen()
    
    def generate_data(self):
        np.random.seed(42)
        n = self.n_points.get()
        
        self.data = {
            'Prix': np.random.lognormal(3.5, 0.4, n).clip(self.min_values['Prix'].get(), self.max_values['Prix'].get()),
            'Kilométrage': np.random.weibull(2.0, n) * 50,
            'Age': np.random.poisson(5, n) + 1,
            'Consommation': np.random.normal(8, 2, n).clip(self.min_values['Consommation'].get(), self.max_values['Consommation'].get()),
            'Puissance': np.random.choice(range(60, 301, 10), n)
        }
        
        self.data['Kilométrage'] = np.clip(self.data['Kilométrage'], self.min_values['Kilométrage'].get(), self.max_values['Kilométrage'].get())
        self.data['Age'] = np.clip(self.data['Age'], self.min_values['Age'].get(), self.max_values['Age'].get())
        self.data['Puissance'] = np.clip(self.data['Puissance'], self.min_values['Puissance'].get(), self.max_values['Puissance'].get())
        
        self.data['Valeur'] = (
            0.8 * np.log(self.data['Prix']) + 
            0.5 * np.sqrt(self.data['Kilométrage']) - 
            2 * self.data['Age']**0.7 + 
            np.random.normal(0, 3, n)
        ).clip(self.min_values['Valeur'].get(), self.max_values['Valeur'].get())
        
        self.df = pd.DataFrame(self.data)
        
        trend = np.linspace(0, 10, n)
        seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, n))
        noise = np.random.normal(0, 2, n)
        self.time_series = 50 + trend + seasonality + noise
        self.dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    
    def create_welcome_screen(self):
        self.clear_window()
        self.root.geometry("800x500")
        main_frame = tk.Frame(self.root, bg='#e3f2fd')
        main_frame.pack(expand=True, fill='both')
        
        header = tk.Frame(main_frame, bg='#0d47a1')
        header.pack(fill='x', pady=10)
        
        try:
            original_image = Image.open("C:/Users/pc/Desktop/app_logo.png")
            resized_image = original_image.resize((150, 75), Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(resized_image)
            image_frame = tk.Frame(header, bg='#bbdefb', bd=2, relief='solid')
            image_frame.pack(side='left', padx=10)
            image_label = tk.Label(image_frame, image=self.image, bg='#e3f2fd')
            image_label.pack()
        except tk.TclError as e:
            tk.Label(header, text="Erreur: Image non trouvée", font=('Arial', 10), bg='#0d47a1', fg='white').pack(side='left', padx=10)
        except Exception as e:
            tk.Label(header, text=f"Erreur: {str(e)}", font=('Arial', 10), bg='#0d47a1', fg='white').pack(side='left', padx=10)
        
        title_frame = tk.Frame(header, bg='#0d47a1')
        title_frame.pack(expand=True, fill='x')
        tk.Label(title_frame, text="Analyse for the Cars", font=('Arial', 16, 'bold'), bg='#0d47a1', fg='white', pady=15).pack()
        tk.Label(title_frame, text="Encadré par: El Mkhalet Mouna", font=('Arial', 10, 'italic'), bg='#0d47a1', fg='white').pack()
        
        content = tk.Frame(main_frame, bg='#e3f2fd')
        content.pack(expand=True, pady=30)
        btn_frame = tk.Frame(content, bg='#e3f2fd')
        btn_frame.pack(pady=30)
        
        tk.Button(btn_frame, text="ENTRER", font=('Arial', 12, 'bold'), bg='#4caf50', fg='white', width=15, height=2, command=self.show_main_interface).pack(side='left', padx=20)
        tk.Button(btn_frame, text="QUITTER", font=('Arial', 12, 'bold'), bg='#f44336', fg='white', width=15, height=2, command=self.root.quit).pack(side='right', padx=20)
        
        footer = tk.Frame(main_frame, bg='#bbdefb')
        footer.pack(fill='x', pady=10)
        tk.Label(footer, text="Étudiant: Hamza Zaidi - 3ème Année EMSI - 2024", font=('Arial', 9), bg='#bbdefb').pack()
    
    def show_main_interface(self):
        self.clear_window()
        self.root.geometry("1400x900")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)
        
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill='both', expand=True)
        
        self.control_frame = tk.Frame(main_frame, bg='#34495e', width=self.panel_width.get())
        self.control_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        self.display_frame = tk.Frame(main_frame, bg='white')
        self.display_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        sidebar_header = tk.Frame(self.control_frame, bg='#2c3e50')
        sidebar_header.pack(fill='x', pady=15)
        tk.Label(sidebar_header, text="PANEL DE CONTRÔLE", font=('Helvetica', 12, 'bold'), bg='#2c3e50', fg='white').pack()
        
        back_btn = tk.Button(self.control_frame, text="◄ Retour à l'accueil", font=('Helvetica', 10), bg='#7f8c8d', fg='white', command=self.create_welcome_screen, relief='flat', bd=0)
        back_btn.pack(fill='x', pady=10, padx=10)
        
        width_frame = tk.Frame(self.control_frame, bg='#34495e')
        width_frame.pack(fill='x', pady=5)
        tk.Label(width_frame, text="Largeur du panneau:", font=('Helvetica', 9), bg='#34495e', fg='white').pack(side='left', padx=5)
        ttk.Entry(width_frame, textvariable=self.panel_width, width=5).pack(side='left', padx=5)
        ttk.Button(width_frame, text="Mettre à jour", command=self.update_panel_width).pack(side='left', padx=5)
        
        notebook = ttk.Notebook(self.control_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Onglet Données
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="Données")
        
        tk.Label(data_tab, text="Nombre de points:", font=('Helvetica', 9)).pack(anchor='w', pady=(10, 0))
        ttk.Entry(data_tab, textvariable=self.n_points).pack(fill='x', pady=5)
        
        for var in self.data.keys():
            frame = tk.Frame(data_tab, bg='#34495e')
            frame.pack(fill='x', pady=2)
            tk.Label(frame, text=f"{var} Min:", font=('Helvetica', 9), bg='#34495e', fg='white').pack(side='left', padx=5)
            ttk.Entry(frame, textvariable=self.min_values[var], width=8).pack(side='left', padx=5)
            tk.Label(frame, text=f"Max:", font=('Helvetica', 9), bg='#34495e', fg='white').pack(side='left', padx=5)
            ttk.Entry(frame, textvariable=self.max_values[var], width=8).pack(side='left', padx=5)
        
        ttk.Button(data_tab, text="Générer Données", command=self.update_data_and_plots).pack(fill='x', pady=10)
        
        # Onglet Visualisation
        visu_tab = ttk.Frame(notebook)
        notebook.add(visu_tab, text="Visualisation")
        
        tk.Label(visu_tab, text="Axe X:", font=('Helvetica', 9)).pack(anchor='w', pady=(10, 0))
        ttk.Combobox(visu_tab, textvariable=self.selected_x, values=list(self.data.keys()), state='readonly').pack(fill='x', pady=5)
        tk.Label(visu_tab, text="Axe Y:", font=('Helvetica', 9)).pack(anchor='w')
        ttk.Combobox(visu_tab, textvariable=self.selected_y, values=list(self.data.keys()), state='readonly').pack(fill='x', pady=5)
        tk.Label(visu_tab, text="Axe Z:", font=('Helvetica', 9)).pack(anchor='w')
        ttk.Combobox(visu_tab, textvariable=self.selected_z, values=list(self.data.keys()), state='readonly').pack(fill='x', pady=5)
        
        # Onglet Analyse avec barre de défilement
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analyse")
        
        # Créer un Canvas et une Scrollbar pour l'onglet Analyse
        canvas = tk.Canvas(analysis_tab, bg='#34495e')
        scrollbar = ttk.Scrollbar(analysis_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configurer le Canvas pour le défilement
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Placer le Canvas et la Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Ajouter les widgets dans scrollable_frame
        tk.Label(scrollable_frame, text="Nombre de clusters:", font=('Helvetica', 9)).pack(anchor='w', pady=(10, 0))
        ttk.Entry(scrollable_frame, textvariable=self.n_clusters).pack(fill='x', pady=5)
        
        tk.Label(scrollable_frame, text="Variables d'entrée (séparer par des virgules):", font=('Helvetica', 9)).pack(anchor='w', pady=(10, 0))
        ttk.Entry(scrollable_frame, textvariable=self.selected_inputs).pack(fill='x', pady=5)
        tk.Label(scrollable_frame, text="Variable de sortie:", font=('Helvetica', 9)).pack(anchor='w')
        ttk.Combobox(scrollable_frame, textvariable=self.selected_output, values=list(self.data.keys()), state='readonly').pack(fill='x', pady=5)
        
        # Paramètres ARIMA
        tk.Label(scrollable_frame, text="Paramètres ARIMA", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(10, 0))
        tk.Label(scrollable_frame, text="p (0-2):", font=('Helvetica', 9)).pack(anchor='w')
        ttk.Entry(scrollable_frame, textvariable=self.arima_p, width=5).pack(fill='x', pady=5)
        tk.Label(scrollable_frame, text="q (0-2):", font=('Helvetica', 9)).pack(anchor='w')
        ttk.Entry(scrollable_frame, textvariable=self.arima_q, width=5).pack(fill='x', pady=5)
        tk.Label(scrollable_frame, text="d (0-2):", font=('Helvetica', 9)).pack(anchor='w')
        ttk.Entry(scrollable_frame, textvariable=self.arima_d, width=5).pack(fill='x', pady=5)
        
        # Boutons des algorithmes
        algorithms = [
            ("Régression Linéaire", self.show_linear_regression),
            ("Clustering K-means", self.show_clustering),
            ("Random Forest", self.show_random_forest),
            ("Séries Temporelles", self.show_time_series),
            ("Validation Croisée", self.show_cross_validation),
            ("Visualisation 3D", self.show_3d_visualization)
        ]
        
        for text, command in algorithms:
            btn = ttk.Button(scrollable_frame, text=f"► {text}", command=command)
            btn.pack(fill='x', pady=3)
        
        self.current_algorithm = None
        self.show_welcome_message()
    
    def update_panel_width(self):
        try:
            new_width = max(200, min(500, self.panel_width.get()))
            self.panel_width.set(new_width)
            self.control_frame.config(width=new_width)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une largeur valide (200-500)")
            self.panel_width.set(350)
    
    def update_data_and_plots(self):
        try:
            n = self.n_points.get()
            if n < 10 or n > 1000:
                messagebox.showerror("Erreur", "Nombre de points doit être entre 10 et 1000")
                return
            
            for var in self.data.keys():
                if self.min_values[var].get() >= self.max_values[var].get():
                    messagebox.showerror("Erreur", f"Le minimum de {var} doit être inférieur au maximum")
                    return
            
            self.generate_data()
            if self.current_algorithm:
                self.current_algorithm()
            messagebox.showinfo("Succès", f"Données mises à jour ({n} points)")
            self.show_data_summary(n)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre valide")
    
    def show_data_summary(self, n):
        stats = {col: {'Min': self.df[col].min(), 'Max': self.df[col].max(), 'Mean': self.df[col].mean()} for col in self.data.keys()}
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Résumé des Données Générées")
        summary_window.geometry("400x300")
        summary_frame = ttk.Frame(summary_window)
        summary_frame.pack(padx=10, pady=10, fill='both', expand=True)
        tk.Label(summary_frame, text=f"Résumé pour {n} points générés:", font=('Helvetica', 12, 'bold')).pack(pady=5)
        for col, values in stats.items():
            info = f"{col}: Min={values['Min']:.2f}, Max={values['Max']:.2f}, Moyenne={values['Mean']:.2f}"
            tk.Label(summary_frame, text=info, font=('Helvetica', 10)).pack()
    
    def update_plots(self, *args):
        if self.current_algorithm and self.current_algorithm != self.show_welcome_message:
            self.current_algorithm()
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def clear_display(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()
    
    def show_welcome_message(self):
        self.clear_display()
        self.current_algorithm = None
        welcome_frame = tk.Frame(self.display_frame, bg='white')
        welcome_frame.pack(expand=True, fill='both')
        tk.Label(welcome_frame, text="BIENVENUE DANS AI DATA ANALYST PRO", font=('Helvetica', 18, 'bold'), bg='white', fg='#2c3e50').pack(pady=20)
        tk.Label(welcome_frame, text="Sélectionnez un algorithme dans le panel de contrôle\npour commencer votre analyse de données", font=('Helvetica', 12), bg='white', fg='#7f8c8d').pack()
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        ax.scatter(x, y, color=self.colors['primary'], alpha=0.6)
        ax.plot(x, np.sin(x), color=self.colors['danger'], linewidth=2)
        ax.set_title("Exemple d'analyse de données", pad=20)
        ax.grid(True, linestyle='--', alpha=0.6)
        canvas = FigureCanvasTkAgg(fig, master=welcome_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=30)
    
    def show_linear_regression(self):
        self.clear_display()
        self.current_algorithm = self.show_linear_regression
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        input_vars = self.selected_inputs.get().split(',')
        y_var = self.selected_output.get()
        
        if not all(var in self.data.keys() for var in input_vars) or y_var not in self.data.keys():
            messagebox.showerror("Erreur", "Variables sélectionnées non valides")
            return
        
        X = self.df[input_vars].values
        y = self.df[y_var].values
        
        # Ajuster le modèle de régression linéaire
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        
        # Afficher la droite de régression
        if len(input_vars) == 1:
            # Cas d'une seule variable : utiliser sns.regplot pour la droite de régression
            sns.regplot(x=input_vars[0], y=y_var, data=self.df, 
                        scatter_kws={'color': self.colors['primary'], 'alpha': 0.6},
                        line_kws={'color': self.colors['danger'], 'linewidth': 2},
                        ax=ax)
            ax.set_xlabel(input_vars[0], labelpad=10)
        else:
            # Cas de plusieurs variables : utiliser la première variable pour afficher la droite
            primary_var = input_vars[0]  # Prendre la première variable d'entrée
            X_primary = self.df[primary_var].values.reshape(-1, 1)
            
            # Ajuster un modèle simple sur la première variable pour la droite
            simple_model = LinearRegression()
            simple_model.fit(X_primary, y)
            x_range = np.linspace(X_primary.min(), X_primary.max(), 100).reshape(-1, 1)
            y_range = simple_model.predict(x_range)
            
            # Nuage de points pour la première variable
            ax.scatter(self.df[primary_var], y, color=self.colors['primary'], alpha=0.6, label='Valeurs réelles')
            # Droite de régression
            ax.plot(x_range, y_range, color=self.colors['danger'], linewidth=2, label='Droite de régression')
            ax.set_xlabel(primary_var, labelpad=10)
        
        ax.set_ylabel(y_var, labelpad=10)
        ax.set_title(f'Régression Linéaire: {", ".join(input_vars)} vs {y_var}', pad=20)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        # Afficher l'équation et la MSE
        if len(input_vars) == 1:
            equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
        else:
            equation = f'y = {" + ".join([f"{coef:.2f}x{i+1}" for i, coef in enumerate(model.coef_)])} + {model.intercept_:.2f}'
        ax.text(0.05, 0.95, f'{equation}\nMSE: {mse:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.embed_plot(fig)
    
    def show_clustering(self):
        self.clear_display()
        self.current_algorithm = self.show_clustering
        self.n_clusters.trace('w', self.update_clustering)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        x_var = self.selected_x.get()
        y_var = self.selected_y.get()
        z_var = self.selected_z.get()
        X = self.df[[x_var, y_var, z_var]].values
        n_clusters = self.n_clusters.get()
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            scatter = ax.scatter(self.df[x_var], self.df[y_var], self.df[z_var], c=clusters, cmap='viridis', s=50, alpha=0.7)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200, c='red', marker='X', label='Centroïdes')
            ax.set_xlabel(x_var, labelpad=10)
            ax.set_ylabel(y_var, labelpad=10)
            ax.set_zlabel(z_var, labelpad=10)
            ax.set_title(f'Clustering K-means ({n_clusters} clusters)', pad=20)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Cluster', rotation=270, labelpad=15)
            self.embed_plot(fig)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de clustering:\n{str(e)}")
            self.show_welcome_message()
    
    def update_clustering(self, *args):
        if self.current_algorithm == self.show_clustering:
            self.show_clustering()
    
    def show_random_forest(self):
        self.clear_display()
        self.current_algorithm = self.show_random_forest
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        features = [col for col in self.data.keys() if col != self.selected_output.get()]
        X = self.df[features].values
        y = self.df[self.selected_output.get()].values
        model = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        bars = ax.barh(np.array(features)[indices], importance[indices], 
                      color=[self.colors['primary'], self.colors['secondary'], self.colors['warning'], self.colors['danger'], '#9b59b6'])
        ax.set_title(f'Importance des Variables (Random Forest) - {self.selected_output.get()}', pad=20)
        ax.set_xlabel('Importance')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., f'{width:.2f}', va='center', ha='left')
        self.embed_plot(fig)
    
    def show_time_series(self):
        self.clear_display()
        self.current_algorithm = self.show_time_series
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Récupérer et valider les paramètres ARIMA
        p = self.arima_p.get()
        q = self.arima_q.get()
        d = self.arima_d.get()
        
        if not (0 <= p <= 2 and 0 <= q <= 2 and 0 <= d <= 2):
            messagebox.showerror("Erreur", "Les paramètres p, q et d doivent être entre 0 et 2")
            return
        
        try:
            # Modèle ARIMA avec paramètres p, d, q
            model = ARIMA(self.time_series, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.predict(start=0, end=len(self.time_series)-1, typ='levels')
            
            # Graphique de série temporelle
            ax.plot(self.dates, self.time_series, label='Valeurs réelles', color=self.colors['primary'], linewidth=1.5)
            ax.plot(self.dates, forecast, color=self.colors['danger'], linewidth=2, label=f'Prédiction ARIMA({p},{d},{q})')
            
            ax.set_xlabel('Date', labelpad=10)
            ax.set_ylabel('Valeur', labelpad=10)
            ax.set_title(f'Analyse des Séries Temporelles (ARIMA {p},{d},{q})', pad=20)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            self.embed_plot(fig)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'exécution du modèle ARIMA:\n{str(e)}")
            self.show_welcome_message()
    
    def show_cross_validation(self):
        self.clear_display()
        self.current_algorithm = self.show_cross_validation
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        features = [col for col in self.data.keys() if col != self.selected_output.get()]
        X = self.df[features].values
        y = self.df[self.selected_output.get()].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # MSE positive, cv=5 par défaut
        mean_mse = np.mean(scores)
        best_mse = np.min(scores)
        
        # Graphique à barres pour MSE moyenne et meilleure MSE
        bars = ax.bar(['MSE Moyenne', 'Meilleure MSE'], [mean_mse, best_mse], 
                      color=[self.colors['primary'], self.colors['secondary']])
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'Validation Croisée - MSE ({self.selected_output.get()})', pad=20)
        ax.set_ylabel('MSE', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(0, max(mean_mse, best_mse) * 1.2)  # Ajuster l'échelle pour la lisibilité
        
        self.embed_plot(fig)
    
    def show_3d_visualization(self):
        self.clear_display()
        self.current_algorithm = self.show_3d_visualization
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        x_var = self.selected_x.get()
        y_var = self.selected_y.get()
        z_var = self.selected_z.get()
        scatter = ax.scatter(self.df[x_var], self.df[y_var], self.df[z_var], c=self.df[self.selected_output.get()], cmap='plasma', s=50, alpha=0.7)
        ax.set_xlabel(x_var, labelpad=10)
        ax.set_ylabel(y_var, labelpad=10)
        ax.set_zlabel(z_var, labelpad=10)
        ax.set_title('Visualisation 3D des Données', pad=20)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(self.selected_output.get(), rotation=270, labelpad=15)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.view_init(elev=25, azim=45)
        self.embed_plot(fig)
    
    def embed_plot(self, figure):
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(figure, master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedAIAnalyst(root)
    root.mainloop()