import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# LightGBM es opcional; si no está instalado, se omite del comparador
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LeaguePredictor:
    """Modelo para predecir resultados de partidos de una liga."""

    def __init__(self, league_code: str = "SP1"):
        self.league_code = league_code
        self.model_path = Path(f"models/{league_code}_predictor.pkl")
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.accuracy = None
        self.model_name = None
        self._target_map = {-1: 0, 0: 1, 1: 2}  # Para modelos que necesitan 0,1,2
        self._inverse_target_map = {0: -1, 1: 0, 2: 1}

    def _encode_target(self, y):
        """Codifica el target de {-1,0,1} a {0,1,2} para XGBoost/LightGBM."""
        return y.map(self._target_map)

    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo Random Forest."""
        print("\n🌳 Entrenando Random Forest...")

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        self.model_name = 'Random Forest'

        if X_test is not None and y_test is not None:
            self.accuracy = self.model.score(X_test, y_test)
            print(f"✓ Accuracy: {self.accuracy:.4f}")

        return self.model

    def train_gradient_boosting(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo Gradient Boosting."""
        print("\n📈 Entrenando Gradient Boosting...")

        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            random_state=42
        )

        self.model.fit(X_train, y_train)
        self.model_name = 'Gradient Boosting'

        if X_test is not None and y_test is not None:
            self.accuracy = self.model.score(X_test, y_test)
            print(f"✓ Accuracy: {self.accuracy:.4f}")

        return self.model

    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo XGBoost."""
        print("\n⚡ Entrenando XGBoost...")

        y_train_encoded = self._encode_target(y_train)

        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            num_class=3
        )

        self.model.fit(X_train, y_train_encoded)
        self.model_name = 'XGBoost'

        if X_test is not None and y_test is not None:
            y_test_encoded = self._encode_target(y_test)
            self.accuracy = self.model.score(X_test, y_test_encoded)
            print(f"✓ Accuracy: {self.accuracy:.4f}")

        return self.model

    def train_lightgbm(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo LightGBM."""
        if not LIGHTGBM_AVAILABLE:
            print("\n⚠️  LightGBM no está instalado. Saltando...")
            return None

        print("\n🌿 Entrenando LightGBM...")

        y_train_encoded = self._encode_target(y_train)

        self.model = LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        self.model.fit(X_train, y_train_encoded)
        self.model_name = 'LightGBM'

        if X_test is not None and y_test is not None:
            y_test_encoded = self._encode_target(y_test)
            self.accuracy = self.model.score(X_test, y_test_encoded)
            print(f"✓ Accuracy: {self.accuracy:.4f}")

        return self.model

    def compare_models(self, X_train, y_train, X_test, y_test):
        """
        Entrena y compara Random Forest, Gradient Boosting, XGBoost y LightGBM.
        Selecciona el mejor modelo según accuracy en el conjunto de test.

        Retorna
        -------
        dict
            Diccionario con los resultados de cada modelo.
        """
        print("\n" + "=" * 60)
        print("🏆 COMPARACIÓN DE MODELOS")
        print("=" * 60)

        results = {}
        models_to_train = [
            ('Random Forest', self.train_random_forest),
            ('Gradient Boosting', self.train_gradient_boosting),
            ('XGBoost', self.train_xgboost),
        ]

        if LIGHTGBM_AVAILABLE:
            models_to_train.append(('LightGBM', self.train_lightgbm))
        else:
            print("\n⚠️  LightGBM no disponible. Instálalo con: pip install lightgbm")

        for name, train_func in models_to_train:
            train_func(X_train, y_train, X_test, y_test)
            results[name] = {
                'accuracy': self.accuracy,
                'model': self.model,
                'model_name': name
            }

        # Seleccionar el mejor modelo
        best_name = max(results, key=lambda k: results[k]['accuracy'])
        best = results[best_name]

        print("\n📊 Resultados de la comparación:")
        for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            marker = "⭐" if name == best_name else "  "
            print(f"{marker} {name:<18} Accuracy: {res['accuracy']:.4f}")

        # Restaurar el mejor modelo como modelo activo
        self.model = best['model']
        self.model_name = best['model_name']
        self.accuracy = best['accuracy']

        print(f"\n✓ Mejor modelo seleccionado: {best_name} ({self.accuracy:.4f})")

        return results

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo activo."""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None

        print(f"\n📊 Evaluación del modelo ({self.model_name}):")

        # Determinar si el modelo fue entrenado con target codificado (0,1,2)
        # o con target original (-1,0,1)
        if self.model_name in ['XGBoost', 'LightGBM']:
            y_test_encoded = self._encode_target(y_test)
        else:
            y_test_encoded = y_test

        y_pred = self.model.predict(X_test)

        print(f"\nAccuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")
        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test_encoded, y_pred))
        print("\nReporte de clasificación:")

        try:
            print(classification_report(
                y_test_encoded, y_pred,
                target_names=['Away Win', 'Draw', 'Home Win'],
                labels=[0, 1, 2]
            ))
        except ValueError as e:
            print(f"Error en reporte: {e}")

        return y_pred

    def predict_match(self, features):
        """
        Predice el resultado de un partido.

        Retorna un diccionario con la predicción y probabilidades.
        """
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None

        # Asegurar que features sea 2D
        if isinstance(features, pd.Series):
            features = features.values.reshape(1, -1)
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)

        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        # Mapear predicción al formato original {-1, 0, 1}
        if self.model_name in ['XGBoost', 'LightGBM']:
            prediction = self._inverse_target_map.get(prediction, prediction)

        result_map = {-1: 'Away Win', 0: 'Draw', 1: 'Home Win'}

        return {
            'prediction': result_map.get(prediction, 'Unknown'),
            'home_win_prob': probability[2] if len(probability) >= 3 else probability[0],
            'draw_prob': probability[1] if len(probability) >= 3 else 0,
            'away_win_prob': probability[0],
        }

    def feature_importance(self, feature_names, top_n=15):
        """Muestra importancia de features."""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]

            print(f"\n🎯 Top {top_n} Features más importantes ({self.model_name}):")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

            return importances

        print("⚠️  El modelo actual no proporciona importancia de features")
        return None

    def save_model(self, path=None):
        """Guarda el modelo entrenado."""
        if self.model is None:
            print("❌ Modelo no entrenado")
            return

        if path is None:
            path = self.model_path

        os.makedirs(os.path.dirname(str(path)), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'accuracy': self.accuracy,
                'model_name': self.model_name,
                'league_code': self.league_code,
            }, f)

        print(f"✓ Modelo guardado en {path}")

    def load_model(self, path=None):
        """Carga un modelo entrenado."""
        if path is None:
            path = self.model_path

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.accuracy = data['accuracy']
                self.model_name = data.get('model_name', 'Unknown')
                self.league_code = data.get('league_code', self.league_code)

            print(f"✓ Modelo cargado desde {path}")
            print(f"  Modelo: {self.model_name}")
            if self.accuracy is not None:
                print(f"  Accuracy guardado: {self.accuracy:.4f}")
            else:
                print(f"  Accuracy: no disponible")

        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.load_data import DataLoader
    from scripts.preprocess import FeatureEngineer
    from scripts.model_evaluation import temporal_split

    loader = DataLoader()
    loader.load_csv_files()
    loader.clean_data()

    engineer = FeatureEngineer(loader.df)
    X, y, df_features = engineer.prepare_features()

    X_train, X_test, y_train, y_test, _, _ = temporal_split(X, y, df_features, test_size=0.2)

    predictor = LeaguePredictor()
    predictor.compare_models(X_train, y_train, X_test, y_test)
    predictor.evaluate(X_test, y_test)
    predictor.feature_importance(X.columns.tolist())
    predictor.save_model()


# Alias retrocompatible para no romper imports existentes
LaLigaPredictor = LeaguePredictor
