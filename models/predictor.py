import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

class LaLigaPredictor:
    """Modelo para predecir resultados de partidos de La Liga"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.accuracy = None
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Entrena modelo Random Forest"""
        print("\n🌳 Entrenando Random Forest...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        
        print(f"✓ Accuracy: {self.accuracy:.4f}")
        
        return self.model
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Entrena modelo Gradient Boosting"""
        print("\n📈 Entrenando Gradient Boosting...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        
        print(f"✓ Accuracy: {self.accuracy:.4f}")
        
        return self.model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("\n⚡ Entrenando XGBoost...")
        
        # Convertir objetivo a valores 0, 1, 2 (necesario para XGBoost multiclass)
        # Mapear: -1 -> 0 (Away Win), 0 -> 1 (Draw), 1 -> 2 (Home Win)
        y_train_encoded = y_train.map({-1: 0, 0: 1, 1: 2})
        y_test_encoded = y_test.map({-1: 0, 0: 1, 1: 2})
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            num_class=3  # Especificar número de clases
        )
        
        self.model.fit(X_train, y_train_encoded)
        self.accuracy = self.model.score(X_test, y_test_encoded)
        
        print(f"✓ Accuracy: {self.accuracy:.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo"""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None
        
        print("\n📊 Evaluación del modelo:")
        
        # Codificar y_test igual que durante el entrenamiento
        y_test_encoded = y_test.map({-1: 0, 0: 1, 1: 2})
        y_pred = self.model.predict(X_test)
        
        print(f"\nAccuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test_encoded, y_pred))
        print("\nReporte de clasificación:")
        
        try:
            print(classification_report(y_test_encoded, y_pred, 
                                    target_names=['Away Win', 'Draw', 'Home Win'],
                                    labels=[0, 1, 2]))
        except ValueError as e:
            print(f"Error en reporte: {e}")
        
        return y_pred
    
    def predict_match(self, features):
        """Predice el resultado de un partido"""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result_map = {-1: 'Away Win', 0: 'Draw', 1: 'Home Win'}
        
        return {
            'prediction': result_map.get(prediction, 'Unknown'),
            'home_win_prob': probability[-1] if len(probability) >= 3 else probability[0],
            'draw_prob': probability[1] if len(probability) >= 3 else 0,
            'away_win_prob': probability[0],
        }
    
    def feature_importance(self, feature_names, top_n=15):
        """Muestra importancia de features"""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            print(f"\n🎯 Top {top_n} Features más importantes:")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return importances
    
    def save_model(self, path='models/laliga_predictor.pkl'):
        """Guarda el modelo entrenado"""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'accuracy': self.accuracy
            }, f)
        
        print(f"✓ Modelo guardado en {path}")
    
    def load_model(self, path='models/laliga_predictor.pkl'):
        """Carga un modelo entrenado"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.accuracy = data['accuracy']
            
            print(f"✓ Modelo cargado desde {path}")
            print(f"  Accuracy guardado: {self.accuracy:.4f}")
        
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {path}")


if __name__ == "__main__":
    from scripts.load_data import DataLoader
    from scripts.preprocess import FeatureEngineer
    
    # Cargar datos
    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()
    
    # Preparar features
    engineer = FeatureEngineer(loader.df)
    X, y, df_features = engineer.prepare_features()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    predictor = LaLigaPredictor()
    predictor.train_xgboost(X_train, y_train, X_test, y_test)
    predictor.evaluate(X_test, y_test)
    predictor.feature_importance(X.columns.tolist())
    predictor.save_model()