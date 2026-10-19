import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

from core.errors import ModelNotTrainedError

# Directorio de los .pkl. Constante de módulo para que los tests puedan
# redirigirla y no escribir modelos reales al ejecutar la suite.
MODELS_DIR = Path("models")

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
        self.model_path = MODELS_DIR / f"{league_code}_predictor.pkl"
        self.model = None
        self.feature_names = None
        self.accuracy = None
        self.model_name = None
        self._target_map = {-1: 0, 0: 1, 1: 2}  # Para modelos que necesitan 0,1,2
        self._inverse_target_map = {0: -1, 1: 0, 2: 1}

    def _encode_target(self, y):
        """Codifica el target de {-1,0,1} a {0,1,2}."""
        return y.map(self._target_map)

    def _train(self, model, model_name, X_train, y_train, X_test=None, y_test=None):
        """Punto único de entrenamiento.

        Codifica el target a {0,1,2} SIEMPRE, para los cuatro modelos, no solo
        para XGBoost/LightGBM. Antes Random Forest y Gradient Boosting se
        entrenaban con el target sin codificar ({-1,0,1}), lo que obligaba a un
        `if self.model_name in [...]` disperso por evaluate()/predict_match()
        para saber qué codificación tenía cada modelo — y ese `if` estaba mal
        en evaluate(), que forzaba labels=[0,1,2] sobre etiquetas {-1,0,1} y
        hacía desaparecer la clase 'victoria visitante' del informe para RF/GB.
        """
        y_train_encoded = self._encode_target(y_train)
        model.fit(X_train, y_train_encoded)
        self.model = model
        self.model_name = model_name

        if X_test is not None and y_test is not None:
            y_test_encoded = self._encode_target(y_test)
            self.accuracy = model.score(X_test, y_test_encoded)
            print(f"✓ Accuracy: {self.accuracy:.4f}")

        return model

    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo Random Forest."""
        print("\n🌳 Entrenando Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
        return self._train(model, 'Random Forest', X_train, y_train, X_test, y_test)

    def train_gradient_boosting(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo Gradient Boosting."""
        print("\n📈 Entrenando Gradient Boosting...")
        model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            min_samples_split=5, random_state=42,
        )
        return self._train(model, 'Gradient Boosting', X_train, y_train, X_test, y_test)

    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo XGBoost."""
        print("\n⚡ Entrenando XGBoost...")
        model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42,
            n_jobs=-1, eval_metric='mlogloss', num_class=3,
        )
        return self._train(model, 'XGBoost', X_train, y_train, X_test, y_test)

    def train_lightgbm(self, X_train, y_train, X_test=None, y_test=None):
        """Entrena modelo LightGBM."""
        if not LIGHTGBM_AVAILABLE:
            print("\n⚠️  LightGBM no está instalado. Saltando...")
            return None

        print("\n🌿 Entrenando LightGBM...")
        model = LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        return self._train(model, 'LightGBM', X_train, y_train, X_test, y_test)

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

        # Todos los modelos se entrenan con el target codificado {0,1,2} (ver _train)
        y_test_encoded = self._encode_target(y_test)
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

    RESULT_NAMES = {-1: 'Away Win', 0: 'Draw', 1: 'Home Win'}

    @staticmethod
    def _as_matrix(features):
        """Normaliza la entrada a una matriz 2D.

        Pasar un ndarray en vez de un DataFrame de una fila es ~14x más rápido:
        la conversión pandas→XGBoost dominaba el coste del Monte Carlo, que hace
        una llamada por partido simulado.
        """
        if isinstance(features, pd.DataFrame):
            return features.to_numpy(dtype=np.float32)
        if isinstance(features, pd.Series):
            return features.to_numpy(dtype=np.float32).reshape(1, -1)
        features = np.asarray(features, dtype=np.float32)
        return features.reshape(1, -1) if features.ndim == 1 else features

    def _probs_by_outcome(self, proba_row) -> dict:
        """Probabilidades indexadas por resultado {-1, 0, 1}.

        Se usa `model.classes_` en lugar de la posición: si una clase faltara en
        el entrenamiento, indexar por posición asignaría la probabilidad al
        resultado equivocado en silencio. Todos los modelos entrenan en
        {0,1,2} (ver _train), así que la traducción es siempre la misma.
        """
        return {
            self._inverse_target_map[int(clase)]: float(valor)
            for clase, valor in zip(self.model.classes_, proba_row)
        }

    def predict_match(self, features):
        """Predice un partido. Devuelve la predicción y las tres probabilidades."""
        if self.model is None:
            raise ModelNotTrainedError(
                f"No hay modelo entrenado para {self.league_code}. "
                f"Pulsa 'Entrenar modelo' primero."
            )

        # Una sola llamada al modelo: la predicción sale del argmax de las
        # probabilidades, así que llamar además a predict() era duplicar el coste.
        proba = self.model.predict_proba(self._as_matrix(features))[0]
        p = self._probs_by_outcome(proba)

        away, draw, home = p.get(-1, 0.0), p.get(0, 0.0), p.get(1, 0.0)
        prediction = max((home, 1), (draw, 0), (away, -1))[1]

        return {
            'prediction': self.RESULT_NAMES[prediction],
            'home_win_prob': home,
            'draw_prob': draw,
            'away_win_prob': away,
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
                'accuracy': self.accuracy,
                'model_name': self.model_name,
                'league_code': self.league_code,
            }, f)

        print(f"✓ Modelo guardado en {path}")

    def load_model(self, path=None):
        """Carga un modelo entrenado.

        Lanza ModelNotTrainedError si no existe el fichero. Antes se tragaba el
        FileNotFoundError y dejaba self.model a None, lo que reaparecía mucho
        más tarde como 'NoneType object is not subscriptable' en mitad de una
        simulación.
        """
        if path is None:
            path = self.model_path

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError as exc:
            raise ModelNotTrainedError(
                f"No hay modelo entrenado para {self.league_code}. "
                f"Pulsa 'Entrenar modelo' primero."
            ) from exc

        self.model = data['model']
        # Un modelo cargado se usa para predecir fila a fila (simulación,
        # próxima jornada). Con n_jobs=-1, XGBoost reparte cada predicción de
        # una fila entre todos los núcleos y la sobrecarga de hilos domina:
        # 0,40 ms frente a 0,11 ms con un solo hilo, y la CPU queda libre.
        if "n_jobs" in self.model.get_params():
            self.model.set_params(n_jobs=1)
        self.accuracy = data.get('accuracy')
        self.model_name = data.get('model_name', 'Unknown')
        self.league_code = data.get('league_code', self.league_code)

        print(f"✓ Modelo cargado desde {path}")
        print(f"  Modelo: {self.model_name}")
        if self.accuracy is not None:
            print(f"  Accuracy guardado: {self.accuracy:.4f}")
        else:
            print("  Accuracy: no disponible")

        return self.model
