"""
Script para ejecutar el pipeline completo:
Cargar -> Procesar -> Entrenar -> Evaluar
"""

import sys
import os
from sklearn.model_selection import train_test_split

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.load_data import DataLoader
from scripts.preprocess import FeatureEngineer
from models.predictor import LaLigaPredictor

def main():
    print("=" * 60)
    print("🎯 PIPELINE COMPLETO - LM_LaLiga")
    print("=" * 60)
    
    # 1. CARGAR DATOS
    print("\n[1/5] CARGANDO DATOS...")
    try:
        loader = DataLoader()
        loader.load_csv_files()
        loader.clean_data()
        loader.save_processed_data()
        loader.get_summary()
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False
    
    # 2. PREPARAR CARACTERÍSTICAS
    print("\n[2/5] PREPARANDO CARACTERÍSTICAS...")
    try:
        engineer = FeatureEngineer(loader.df)
        X, y, df_features = engineer.prepare_features()
        print(f"✓ Features preparados: {X.shape[1]} características, {X.shape[0]} muestras")
    except Exception as e:
        print(f"❌ Error preparando características: {e}")
        return False
    
    # 3. DIVIDIR DATOS
    print("\n[3/5] DIVIDIENDO DATOS...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    except Exception as e:
        print(f"❌ Error dividiendo datos: {e}")
        return False
    
    # 4. ENTRENAR MODELO
    print("\n[4/5] ENTRENANDO MODELOS...")
    try:
        predictor = LaLigaPredictor()
        
        # Probar XGBoost (mejor rendimiento)
        predictor.train_xgboost(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"❌ Error entrenando modelo: {e}")
        return False
    
    # 5. EVALUAR
    print("\n[5/5] EVALUANDO MODELO...")
    try:
        predictor.evaluate(X_test, y_test)
        predictor.feature_importance(X.columns.tolist(), top_n=15)
        predictor.save_model()
    except Exception as e:
        print(f"❌ Error evaluando modelo: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ¡Pipeline completado exitosamente!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)