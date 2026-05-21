"""
DentaScan — Script de entrenamiento de modelos ML.
Ejecutar: python scripts/train_model.py --dataset_dir input/balanceado

Entrena SVM y Random Forest sobre el dataset de radiografías.
Guarda los modelos en models_ml/ para que el servidor los cargue.

Pipeline de entrenamiento:
    input/ (JPG/PNG) → conversión PNG → balance por clase
    → extracción de 12 features (OpenCV)
    → división 70/15/15 estratificada
    → GridSearchCV (5-fold)
    → guardado de mejor modelo
"""

import sys
import argparse
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score

from app.config import get_settings
from app.core.preprocessor import ImagePreprocessor
from app.core.feature_extractor import FeatureExtractor
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger("train_model")

# ── Clases del dataset ─────────────────────────────────────────────────────────
CLASS_MAP = {
    "diente_sano": 0,
    "caries_incipiente": 1,
    "caries_avanzada": 2,
    "absceso_periapical": 3,
    "lesion_osea": 4,
}

RANDOM_SEED = 42


def load_dataset(dataset_dir: Path, samples_per_class: int = 47) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga imágenes del dataset y extrae features.
    Estructura esperada:
        dataset_dir/
            diente_sano/        *.jpg | *.png
            caries_incipiente/  *.jpg | *.png
            ...

    Aplica balanceo por submuestreo aleatorio (seed=42).
    """
    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()

    X, y = [], []
    random.seed(RANDOM_SEED)

    for class_name, class_idx in CLASS_MAP.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            logger.warning("Directorio no encontrado: %s — omitiendo clase.", class_dir)
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            logger.warning("No se encontraron imágenes en %s", class_dir)
            continue

        # Balanceo por submuestreo
        if len(images) > samples_per_class:
            images = random.sample(images, samples_per_class)

        logger.info("Clase '%s': %d imágenes", class_name, len(images))

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            processed = preprocessor.preprocess(img)
            features = extractor.extract(processed)
            X.append([features[k] for k in [
                "media", "std", "min_px", "max_px",
                "bordes_mean", "sobel_mean",
                "zona_tl", "zona_tr", "zona_bl", "zona_br",
                "prop_oscuros", "asimetria",
            ]])
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cargando dataset desde: %s", dataset_dir)

    X, y = load_dataset(dataset_dir)
    if len(X) == 0:
        logger.error("Dataset vacío. Verifica la estructura de directorios.")
        return

    logger.info("Dataset cargado: %d muestras, %d features, %d clases",
                len(X), X.shape[1], len(np.unique(y)))

    # División 70/15/15 estratificada
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    logger.info("Split — train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # Normalización Min-Max — ajustar solo en train
    scaler = MinMaxScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n = scaler.transform(X_val)
    X_test_n = scaler.transform(X_test)
    joblib.dump(scaler, output_dir / "scaler.pkl")
    logger.info("Scaler guardado.")

    # ── Random Forest ──────────────────────────────────────────────────────────
    logger.info("Entrenando Random Forest con GridSearchCV...")
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
    }
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    rf_cv = GridSearchCV(rf_base, rf_params, cv=5, scoring="f1_macro", n_jobs=-1)
    rf_cv.fit(X_train_n, y_train)

    best_rf = rf_cv.best_estimator_
    rf_val_f1 = f1_score(y_val, best_rf.predict(X_val_n), average="macro")
    rf_test_f1 = f1_score(y_test, best_rf.predict(X_test_n), average="macro")

    logger.info("RF — mejores params: %s", rf_cv.best_params_)
    logger.info("RF — F1-macro val=%.4f  test=%.4f", rf_val_f1, rf_test_f1)
    print("\nRandom Forest — Reporte en validación:")
    print(classification_report(y_val, best_rf.predict(X_val_n),
                                 target_names=list(CLASS_MAP.keys()), zero_division=0))

    joblib.dump(best_rf, output_dir / "rf_model.pkl")
    logger.info("Random Forest guardado.")

    # ── SVM ───────────────────────────────────────────────────────────────────
    logger.info("Entrenando SVM con GridSearchCV...")
    svm_params = {
        "C": [1, 10, 100],
        "kernel": ["rbf", "poly", "linear"],
        "gamma": ["scale", "auto"],
    }
    svm_base = SVC(class_weight="balanced", random_state=RANDOM_SEED, probability=True)
    svm_cv = GridSearchCV(svm_base, svm_params, cv=5, scoring="f1_macro", n_jobs=-1)
    svm_cv.fit(X_train_n, y_train)

    best_svm = svm_cv.best_estimator_
    svm_val_f1 = f1_score(y_val, best_svm.predict(X_val_n), average="macro")
    svm_test_f1 = f1_score(y_test, best_svm.predict(X_test_n), average="macro")

    logger.info("SVM — mejores params: %s", svm_cv.best_params_)
    logger.info("SVM — F1-macro val=%.4f  test=%.4f", svm_val_f1, svm_test_f1)
    print("\nSVM — Reporte en validación:")
    print(classification_report(y_val, best_svm.predict(X_val_n),
                                  target_names=list(CLASS_MAP.keys()), zero_division=0))

    joblib.dump(best_svm, output_dir / "svm_model.pkl")
    logger.info("SVM guardado.")

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE ENTRENAMIENTO")
    print("="*60)
    winner = "Random Forest" if rf_val_f1 >= svm_val_f1 else "SVM"
    print(f"Mejor modelo: {winner}")
    print(f"  RF  — F1-macro val={rf_val_f1:.4f}  test={rf_test_f1:.4f}")
    print(f"  SVM — F1-macro val={svm_val_f1:.4f}  test={svm_test_f1:.4f}")
    print(f"\nModelos guardados en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DentaScan — Entrenamiento de modelos ML")
    parser.add_argument("--dataset_dir", type=Path,
                        default=Path("input/balanceado"),
                        help="Directorio raíz del dataset (con carpetas por clase)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("models_ml"),
                        help="Directorio de salida para los modelos pkl")
    args = parser.parse_args()

    train(args.dataset_dir, args.output_dir)
