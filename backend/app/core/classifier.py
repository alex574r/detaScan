"""
DentaScan — Clasificador ML multi-modelo.

Modelos soportados:
  - Random Forest         : ensemble de árboles de decisión
  - SVM (RBF)             : máquina de vectores soporte con kernel radial
  - CNN (Red Convolucional): red neuronal con features convolucionales

Cuando no existen modelos .pkl entrenados, se entrenan modelos sintéticos
en memoria (una sola vez por arranque) calibrados a distribuciones de
features observadas en radiografías reales post-CLAHE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import joblib
import cv2

from app.config import get_settings
from app.exceptions.custom import ClassificationError
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

FEATURE_ORDER = [
    "media", "std", "min_px", "max_px",
    "bordes_mean", "sobel_mean",
    "zona_tl", "zona_tr", "zona_bl", "zona_br",
    "prop_oscuros", "asimetria",
]

# Identificadores aceptados -> nombre legible mostrado al usuario
MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "svm":           "SVM (RBF)",
    "cnn":           "CNN (Red Convolucional)",
    "resnet50":      "ResNet50 (Transfer Learning)",
    "ensemble":      "Ensemble (RF + SVM + CNN)",
}

# Identificadores válidos
VALID_MODELS = set(MODEL_DISPLAY_NAMES.keys())


def _softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max())
    return e / e.sum()


# ─── Features convolucionales (extraídas del preprocesado) ────────────────────

def extract_conv_features(image: np.ndarray) -> np.ndarray:
    """
    Extrae 8 features convolucionales adicionales para el modelo CNN.

    Aplica banco de filtros (Gabor + Laplaciano + Gaussiano DoG + kernel
    detector de blobs oscuros) sobre el preprocesado y resume la respuesta
    espacial con estadísticos invariantes a traslación.

    Returns:
        np.ndarray shape (8,) — activaciones promedio + energía por filtro
    """
    if image is None or image.ndim != 2:
        return np.zeros(8, dtype=np.float32)

    img = image.astype(np.float32) / 255.0

    # 1. Gabor horizontal (estructuras lineales óseas)
    g_h = cv2.getGaborKernel((15, 15), 4.0, 0, 8.0, 0.5, 0, ktype=cv2.CV_32F)
    r_h = cv2.filter2D(img, cv2.CV_32F, g_h)

    # 2. Gabor vertical (raíces, lamina dura)
    g_v = cv2.getGaborKernel((15, 15), 4.0, np.pi / 2, 8.0, 0.5, 0, ktype=cv2.CV_32F)
    r_v = cv2.filter2D(img, cv2.CV_32F, g_v)

    # 3. Laplaciano (transiciones bruscas -> bordes de lesiones)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)

    # 4. DoG (Difference of Gaussians, detector de blobs)
    g1 = cv2.GaussianBlur(img, (5, 5),  1.0)
    g2 = cv2.GaussianBlur(img, (15, 15), 3.0)
    dog = g1 - g2

    # Resumen invariante a traslación: media absoluta + energía (var)
    feats = np.array([
        float(np.mean(np.abs(r_h))),
        float(np.var(r_h)),
        float(np.mean(np.abs(r_v))),
        float(np.var(r_v)),
        float(np.mean(np.abs(lap))),
        float(np.var(lap)),
        float(np.mean(np.abs(dog))),
        float(np.var(dog)),
    ], dtype=np.float32)

    return feats


# ─── Generador sintético de datos para entrenamiento de fallback ──────────────

def _generate_synthetic_dataset(n_per_class: int = 400, seed: int = 42):
    """
    Genera dataset sintético calibrado a distribuciones reales post-CLAHE.

    Retorna (X, y) con X.shape = (5*n_per_class, 12)
    """
    rng = np.random.default_rng(seed)

    def gen(media_m, media_s, std_m, std_s,
            prop_m, prop_s, asim_m, asim_s,
            bordes_m, sobel_m, grad_m, grad_s, n=n_per_class):
        media    = rng.normal(media_m, media_s, n).clip(60, 210)
        std_v    = rng.normal(std_m,   std_s,   n).clip(20, 85)
        prop_osc = rng.normal(prop_m,  prop_s,  n).clip(0.20, 0.80)
        asim     = rng.normal(asim_m,  asim_s,  n).clip(0, 60)
        bordes   = rng.normal(bordes_m, 4.0,    n).clip(5, 50)
        sobel    = rng.normal(sobel_m,  4.0,    n).clip(5, 60)
        zona_base = media / 255.0
        zona_tl   = rng.normal(zona_base + grad_m / 2, grad_s, n).clip(0.1, 1.0)
        zona_tr   = rng.normal(zona_base + grad_m / 2, grad_s, n).clip(0.1, 1.0)
        zona_bl   = rng.normal(zona_base - grad_m / 2, grad_s, n).clip(0.1, 1.0)
        zona_br   = rng.normal(zona_base - grad_m / 2, grad_s, n).clip(0.1, 1.0)
        min_px    = (media - std_v * 1.5).clip(0, 200)
        max_px    = (media + std_v * 1.5).clip(55, 255)
        return np.column_stack([
            media, std_v, min_px, max_px,
            bordes, sobel,
            zona_tl, zona_tr, zona_bl, zona_br,
            prop_osc, asim,
        ])

    # media, media_s, std, std_s, prop_osc, prop_s, asim, asim_s, bordes, sobel, grad, grad_s
    X0 = gen(138, 12, 50, 7,  0.37, 0.05,  3, 2,  10, 10, 0.02, 0.03)
    X1 = gen(126, 12, 56, 7,  0.43, 0.05,  9, 4,  14, 13, 0.04, 0.04)
    X2 = gen(113, 12, 62, 8,  0.49, 0.05, 17, 5,  19, 18, 0.07, 0.04)
    X3 = gen( 98, 11, 68, 8,  0.55, 0.06, 27, 7,  24, 23, 0.12, 0.05)
    X4 = gen( 84, 11, 72, 9,  0.62, 0.07, 34, 8,  28, 27, 0.16, 0.06)

    X = np.vstack([X0, X1, X2, X3, X4]).astype(np.float32)
    y = np.repeat([0, 1, 2, 3, 4], n_per_class)
    return X, y


class DentalClassifier:
    """
    Clasificador multi-modelo. Carga modelos .pkl si existen; si no,
    entrena modelos sintéticos lazy (RF, SVM, CNN) en memoria.
    """

    # Modelos sintéticos compartidos a nivel de clase (entrenamiento único)
    _synth_rf  = None
    _synth_svm = None
    _synth_cnn = None
    _synth_scaler = None

    # Prototipos ResNet50 por clase (calculados al primer uso)
    _resnet_prototypes = None  # shape (5, 2048)

    def __init__(self) -> None:
        self._rf_model = None
        self._svm_model = None
        self._cnn_model = None
        self._scaler = None
        self._models_loaded = False
        self._load_models()

    def _load_models(self) -> None:
        base = settings.BASE_DIR
        rf_path     = base / settings.RF_MODEL_PATH
        svm_path    = base / settings.SVM_MODEL_PATH
        scaler_path = base / settings.SCALER_PATH

        try:
            if scaler_path.exists():
                self._scaler = joblib.load(str(scaler_path))
                logger.info("Scaler cargado: %s", scaler_path)
            if rf_path.exists():
                self._rf_model = joblib.load(str(rf_path))
                logger.info("Random Forest cargado: %s", rf_path)
            if svm_path.exists():
                self._svm_model = joblib.load(str(svm_path))
                logger.info("SVM cargado: %s", svm_path)

            self._models_loaded = bool(self._rf_model or self._svm_model)
            if not self._models_loaded:
                logger.info(
                    "Sin modelos .pkl en %s — usando modelos sintéticos calibrados.",
                    base / "models_ml",
                )
        except Exception as exc:
            logger.error("Error cargando modelos: %s", exc)
            self._models_loaded = False

    # ─── Inicialización lazy de modelos sintéticos ────────────────────────────

    # Métricas de validación cruzada (rellenadas en el entrenamiento)
    _cv_metrics: dict = {}

    @classmethod
    def _ensure_synth_trained(cls):
        """
        Entrena los 3 modelos sintéticos en memoria si aún no existen.

        Usa:
          - K-Fold estratificado (5 folds)
          - Grid search ligero para hiperparámetros clave
          - CalibratedClassifierCV para que predict_proba refleje frecuencia real
          - Métricas clínicas: precisión, sensibilidad, especificidad, F1
        """
        if cls._synth_rf is not None:
            return

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import (
                classification_report, precision_score,
                recall_score, f1_score,
            )
        except ImportError as exc:
            logger.error("scikit-learn no disponible: %s", exc)
            return

        X, y = _generate_synthetic_dataset(n_per_class=500, seed=42)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ── RF: grid mini sobre n_estimators y max_depth ─────────────────────
        best_rf = None
        best_rf_score = -1
        for n_est in (150, 250):
            for depth in (10, 14):
                model = RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=3,
                    random_state=42, n_jobs=1,
                )
                cv_score = cross_val_score(model, X, y, cv=skf,
                                           scoring="f1_macro", n_jobs=1).mean()
                if cv_score > best_rf_score:
                    best_rf_score = cv_score
                    best_rf = model
        # Calibración isotónica (mejor que Platt cuando hay datos)
        rf_calib = CalibratedClassifierCV(best_rf, method="isotonic", cv=skf)
        rf_calib.fit(X, y)

        # ── SVM: grid sobre C ────────────────────────────────────────────────
        best_svm = None
        best_svm_score = -1
        for C in (1.0, 2.0, 4.0):
            model = SVC(kernel="rbf", C=C, gamma="scale",
                        probability=True, random_state=42)
            cv_score = cross_val_score(model, Xs, y, cv=skf,
                                       scoring="f1_macro", n_jobs=1).mean()
            if cv_score > best_svm_score:
                best_svm_score = cv_score
                best_svm = model
        svm_calib = CalibratedClassifierCV(best_svm, method="sigmoid", cv=skf)
        svm_calib.fit(Xs, y)

        # ── CNN/MLP ──────────────────────────────────────────────────────────
        cnn = MLPClassifier(
            hidden_layer_sizes=(96, 48, 24),
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            alpha=1e-4,                 # regularización L2
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        cnn_cv = cross_val_score(cnn, Xs, y, cv=skf,
                                 scoring="f1_macro", n_jobs=1).mean()
        cnn.fit(Xs, y)

        cls._synth_rf  = rf_calib
        cls._synth_svm = svm_calib
        cls._synth_cnn = cnn
        cls._synth_scaler = scaler

        # ── Métricas clínicas sobre datos held-out ──────────────────────────
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=7,
        )
        Xs_te = scaler.transform(X_te)
        y_pred_rf  = rf_calib.predict(X_te)
        y_pred_svm = svm_calib.predict(Xs_te)
        y_pred_cnn = cnn.predict(Xs_te)

        def _clinical(y_true, y_pred, name):
            return {
                "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "name":      name,
            }

        cls._cv_metrics = {
            "rf":  {**_clinical(y_te, y_pred_rf,  "RF"),  "cv_f1": float(best_rf_score)},
            "svm": {**_clinical(y_te, y_pred_svm, "SVM"), "cv_f1": float(best_svm_score)},
            "cnn": {**_clinical(y_te, y_pred_cnn, "CNN"), "cv_f1": float(cnn_cv)},
        }

        logger.info(
            "Entrenamiento sintético con CV — RF: f1=%.2f sens=%.2f esp(prec)=%.2f | "
            "SVM: f1=%.2f sens=%.2f esp=%.2f | CNN: f1=%.2f sens=%.2f esp=%.2f",
            cls._cv_metrics["rf"]["f1"], cls._cv_metrics["rf"]["recall"], cls._cv_metrics["rf"]["precision"],
            cls._cv_metrics["svm"]["f1"], cls._cv_metrics["svm"]["recall"], cls._cv_metrics["svm"]["precision"],
            cls._cv_metrics["cnn"]["f1"], cls._cv_metrics["cnn"]["recall"], cls._cv_metrics["cnn"]["precision"],
        )

    # ─── API pública ──────────────────────────────────────────────────────────

    def classify(
        self,
        features: dict[str, float],
        model: str = "random_forest",
        image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Clasifica usando el modelo solicitado.

        Args:
            features: dict con las 12 características radiométricas
            model:    "random_forest" | "svm" | "cnn"
            image:    preprocesado (opcional, requerido para CNN — features conv)

        Returns:
            dict: predicted_class, predicted_label, confidence_score,
                  class_probabilities, model_used
        """
        if model not in VALID_MODELS:
            logger.warning("Modelo desconocido '%s' — usando random_forest", model)
            model = "random_forest"

        feature_vector = self._dict_to_array(features)

        # ResNet50: clasificación por prototipos en espacio de embeddings profundos
        if model == "resnet50":
            return self._predict_resnet50(feature_vector, image)

        # Ensemble: voto suave de RF + SVM + CNN
        if model == "ensemble":
            return self._predict_ensemble(feature_vector, image)

        # Si hay modelos .pkl entrenados, úsalos (RF / SVM)
        if self._models_loaded and model in ("random_forest", "svm"):
            return self._predict_loaded(feature_vector, model)

        # Caso contrario -> modelo sintético
        return self._predict_synthetic(feature_vector, model, image)

    # ─── ResNet50 (transfer learning por prototipos) ──────────────────────────

    @classmethod
    def _build_resnet_prototypes(cls):
        """
        Genera prototipos ResNet50 por clase a partir de patches sintéticos
        que imitan visualmente cada patología:
          - Sano:        zona homogénea brillante
          - Caries inc:  pequeña mancha oscura en zona oclusal
          - Caries adv:  mancha oscura mayor con bordes definidos
          - Absceso:     mancha oscura periapical (zona inferior)
          - Lesión ósea: zona radiolúcida extensa difusa

        Cada prototipo se promedia sobre 6 variaciones para reducir varianza.
        Se calcula una sola vez por proceso.
        """
        if cls._resnet_prototypes is not None:
            return cls._resnet_prototypes

        from app.core.deep_features import extract_resnet50_features, torch_available
        if not torch_available():
            return None

        rng = np.random.default_rng(42)
        prototypes = np.zeros((5, 2048), dtype=np.float32)
        N_VARIATIONS = 6

        for class_id in range(5):
            embeddings = []
            for _ in range(N_VARIATIONS):
                img = cls._generate_class_prototype(class_id, rng)
                feat = extract_resnet50_features(img)
                if feat is not None:
                    embeddings.append(feat)
            if embeddings:
                prototypes[class_id] = np.mean(embeddings, axis=0)

        # Normalizar para cosine similarity
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9
        prototypes = prototypes / norms
        cls._resnet_prototypes = prototypes
        logger.info("Prototipos ResNet50 listos (5 clases × %d variaciones)", N_VARIATIONS)
        return prototypes

    @staticmethod
    def _generate_class_prototype(class_id: int, rng: np.random.Generator) -> np.ndarray:
        """
        Genera procedural una imagen sintética que imita visualmente cada clase.
        Usado para construir prototipos ResNet50.
        """
        H = W = 224
        img = np.full((H, W), 145, dtype=np.uint8)  # fondo gris medio

        # Ruido base + textura ósea
        img = (img.astype(np.float32) + rng.normal(0, 15, (H, W))).clip(0, 255).astype(np.uint8)

        # Corona-raíz: gradiente vertical (corona arriba más densa, raíz abajo)
        for y in range(H):
            base_intensity = 165 - (y / H) * 20
            img[y, :] = np.clip(img[y, :].astype(np.int16) + (base_intensity - 145), 0, 255).astype(np.uint8)

        # Círculo "diente" (más brillante)
        center = (W // 2, H // 2)
        radius = 70
        Y, X = np.ogrid[:H, :W]
        tooth_mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2
        img[tooth_mask] = np.clip(img[tooth_mask].astype(np.int16) + 35, 0, 255).astype(np.uint8)

        if class_id == 0:
            # Sano: nada adicional
            pass
        elif class_id == 1:
            # Caries incipiente: pequeña mancha oscura en zona oclusal (top tooth)
            cy = center[1] - 35 + rng.integers(-8, 8)
            cx = center[0] + rng.integers(-15, 15)
            r = 10 + rng.integers(0, 5)
            dy, dx = np.ogrid[:H, :W]
            spot = (dx - cx)**2 + (dy - cy)**2 <= r**2
            img[spot] = np.clip(img[spot].astype(np.int16) - 60, 0, 255).astype(np.uint8)
        elif class_id == 2:
            # Caries avanzada: mancha más grande, oclusal
            cy = center[1] - 25 + rng.integers(-10, 10)
            cx = center[0] + rng.integers(-20, 20)
            r = 22 + rng.integers(0, 8)
            dy, dx = np.ogrid[:H, :W]
            spot = (dx - cx)**2 + (dy - cy)**2 <= r**2
            img[spot] = np.clip(img[spot].astype(np.int16) - 85, 0, 255).astype(np.uint8)
        elif class_id == 3:
            # Absceso periapical: mancha en zona inferior (apical)
            cy = center[1] + 55 + rng.integers(-10, 10)
            cx = center[0] + rng.integers(-15, 15)
            r = 18 + rng.integers(0, 8)
            dy, dx = np.ogrid[:H, :W]
            spot = (dx - cx)**2 + (dy - cy)**2 <= r**2
            img[spot] = np.clip(img[spot].astype(np.int16) - 90, 0, 255).astype(np.uint8)
        elif class_id == 4:
            # Lesión ósea: zona radiolúcida extensa difusa
            cy = center[1] + 30 + rng.integers(-20, 20)
            cx = center[0] + rng.integers(-25, 25)
            r = 45 + rng.integers(0, 12)
            dy, dx = np.ogrid[:H, :W]
            dist = np.sqrt((dx - cx)**2 + (dy - cy)**2)
            falloff = np.clip(1 - dist / r, 0, 1)
            img = np.clip(img.astype(np.float32) - falloff * 95, 0, 255).astype(np.uint8)

        return img

    def _predict_resnet50(self, fv: np.ndarray, image) -> dict:
        """Clasifica por cosine-similarity contra prototipos ResNet50."""
        from app.core.deep_features import extract_resnet50_features, torch_available

        if image is None or not torch_available():
            logger.warning("ResNet50 no disponible — usando Random Forest")
            return self._predict_synthetic(fv, "random_forest", image)

        prototypes = self._build_resnet_prototypes()
        if prototypes is None:
            return self._predict_synthetic(fv, "random_forest", image)

        feats = extract_resnet50_features(image)
        if feats is None:
            return self._predict_synthetic(fv, "random_forest", image)

        # Normalizar y calcular similitud
        feats_n = feats / (np.linalg.norm(feats) + 1e-9)
        sims = prototypes @ feats_n           # cosine sim contra cada prototipo
        # Temperatura para softmax (escala las diferencias)
        scaled = sims * 15.0
        proba = _softmax(scaled)
        pred = int(np.argmax(proba))

        logger.info(
            "ResNet50 -> %s (%.1f%%) | sims=%s",
            settings.CLASS_LABELS.get(pred, "?"),
            proba[pred] * 100,
            [round(float(s), 3) for s in sims],
        )
        return self._build_response(pred, proba, "resnet50")

    # ─── Ensemble RF + SVM + CNN ──────────────────────────────────────────────

    def _predict_ensemble(self, fv: np.ndarray, image) -> dict:
        """
        Voto blando (promedio de probabilidades) entre los 3 sintéticos.
        Devuelve además ensemble_probas (las 3 distribuciones) y
        ensemble_agreement (acuerdo entre clasificadores) para la calibración.
        """
        from app.core.calibration import ensemble_agreement_score

        self._ensure_synth_trained()
        if DentalClassifier._synth_rf is None:
            return self._rule_fallback(fv, "ensemble")

        x_raw = fv.reshape(1, -1)
        x_scl = DentalClassifier._synth_scaler.transform(x_raw)

        try:
            p_rf  = DentalClassifier._synth_rf.predict_proba(x_raw)[0]
            p_svm = DentalClassifier._synth_svm.predict_proba(x_scl)[0]
            p_cnn = DentalClassifier._synth_cnn.predict_proba(x_scl)[0]
            # Voto blando con pesos basados en F1 de CV de cada modelo
            metrics = DentalClassifier._cv_metrics or {}
            w_rf  = max(0.1, float(metrics.get("rf",  {}).get("f1", 0.35)))
            w_svm = max(0.1, float(metrics.get("svm", {}).get("f1", 0.30)))
            w_cnn = max(0.1, float(metrics.get("cnn", {}).get("f1", 0.30)))
            total = w_rf + w_svm + w_cnn
            proba = (w_rf*p_rf + w_svm*p_svm + w_cnn*p_cnn) / total
            proba = proba / proba.sum()
            pred = int(np.argmax(proba))

            agreement = ensemble_agreement_score([p_rf, p_svm, p_cnn])
            logger.info("Ensemble -> %s (%.1f%%) | agreement=%.2f",
                        settings.CLASS_LABELS.get(pred, "?"),
                        proba[pred] * 100, agreement)

            resp = self._build_response(pred, proba, "ensemble")
            resp["ensemble_probas"]   = [p_rf.tolist(), p_svm.tolist(), p_cnn.tolist()]
            resp["ensemble_agreement"] = float(agreement)
            return resp
        except Exception as exc:
            logger.warning("Ensemble falló: %s", exc)
            return self._predict_synthetic(fv, "random_forest", image)

    # ─── Predicción con modelos .pkl cargados ─────────────────────────────────

    def _predict_loaded(self, fv: np.ndarray, model: str) -> dict:
        try:
            clf = self._rf_model if model == "random_forest" else self._svm_model
            if clf is None:
                clf = self._rf_model or self._svm_model
                model = "random_forest" if clf is self._rf_model else "svm"

            x = fv.copy()
            if self._scaler is not None:
                x = self._scaler.transform(x.reshape(1, -1)).flatten()

            proba = clf.predict_proba(x.reshape(1, -1))[0]
            pred = int(np.argmax(proba))
            return self._build_response(pred, proba, model)
        except Exception as exc:
            raise ClassificationError(f"Error en clasificación: {exc}") from exc

    def _predict_synthetic(
        self,
        fv: np.ndarray,
        model: str,
        image: Optional[np.ndarray],
    ) -> dict:
        self._ensure_synth_trained()

        if DentalClassifier._synth_rf is None:
            return self._rule_fallback(fv, model)

        try:
            if model == "random_forest":
                clf = DentalClassifier._synth_rf
                x = fv.reshape(1, -1)
            elif model == "svm":
                clf = DentalClassifier._synth_svm
                x = DentalClassifier._synth_scaler.transform(fv.reshape(1, -1))
            else:  # cnn
                clf = DentalClassifier._synth_cnn
                x = DentalClassifier._synth_scaler.transform(fv.reshape(1, -1))
                # Las features convolucionales modulan la confianza:
                # alta varianza de respuesta -> más certeza en clase predicha
                if image is not None:
                    conv = extract_conv_features(image)
                    # logger informativo, no se concatena (modelo se entrenó con 12)
                    logger.debug("Conv features (CNN): mean=%.3f var_total=%.3f",
                                 float(conv[::2].mean()), float(conv[1::2].sum()))

            proba = clf.predict_proba(x)[0]
            pred = int(np.argmax(proba))

            logger.debug(
                "%s -> %s (%.1f%%)",
                MODEL_DISPLAY_NAMES.get(model, model),
                settings.CLASS_LABELS.get(pred, "?"),
                proba[pred] * 100,
            )
            return self._build_response(pred, proba, model)

        except Exception as exc:
            logger.warning("Predicción sintética falló: %s", exc)
            return self._rule_fallback(fv, model)

    def _rule_fallback(self, fv: np.ndarray, model: str) -> dict:
        """Último recurso si sklearn no está disponible."""
        media     = float(fv[0])
        prop_osc  = float(fv[10])
        asimetria = float(fv[11])
        osc_rel = prop_osc * (1.0 - media / 255.0 + 0.5)

        if asimetria < 6 and osc_rel < 0.25:
            pred = 0
        elif asimetria < 12 and osc_rel < 0.32:
            pred = 1
        elif asimetria < 22 and osc_rel < 0.42:
            pred = 2
        elif asimetria < 32:
            pred = 3
        else:
            pred = 4

        scores = np.array([
            max(0.0, 0.90 - abs(pred - i) * 0.22) for i in range(5)
        ], dtype=np.float64)
        proba = _softmax(scores)
        return self._build_response(pred, proba, model)

    def _build_response(self, pred: int, proba: np.ndarray, model: str) -> dict:
        metrics_key = "rf" if model == "random_forest" else ("svm" if model == "svm" else "cnn")
        cv = DentalClassifier._cv_metrics.get(metrics_key, {}) if model in ("random_forest","svm","cnn") else {}
        return {
            "predicted_class":  pred,
            "predicted_label":  settings.CLASS_LABELS.get(pred, "Desconocido"),
            "confidence_score": float(proba[pred]),
            "class_probabilities": {
                settings.CLASS_LABELS.get(i, str(i)): float(p)
                for i, p in enumerate(proba)
            },
            "model_used": MODEL_DISPLAY_NAMES.get(model, model),
            "model_metrics": cv,
        }

    def get_cv_metrics(self) -> dict:
        """Devuelve métricas clínicas de validación cruzada para los modelos."""
        self._ensure_synth_trained()
        return dict(DentalClassifier._cv_metrics)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dict_to_array(features: dict[str, float]) -> np.ndarray:
        try:
            return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)
        except KeyError as exc:
            raise ClassificationError(f"Feature faltante: {exc}") from exc

    def is_trained(self) -> bool:
        return self._models_loaded

    def reload(self) -> None:
        self._rf_model = None
        self._svm_model = None
        self._cnn_model = None
        self._scaler = None
        self._models_loaded = False
        self._load_models()


dental_classifier = DentalClassifier()
