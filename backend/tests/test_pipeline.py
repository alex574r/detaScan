"""
DentaScan — Tests del pipeline de procesamiento de imágenes.
Ejecutar: pytest tests/ -v
"""

import numpy as np
import pytest
import cv2


@pytest.fixture
def sample_image():
    """Imagen sintética de 256x256 para tests del pipeline."""
    img = np.zeros((256, 256), dtype=np.uint8)
    # Simular estructura dental: fondo oscuro + zona brillante central
    cv2.ellipse(img, (128, 128), (80, 60), 0, 0, 360, 200, -1)
    # Zona radiolúcida (simula caries)
    cv2.circle(img, (100, 100), 15, 30, -1)
    return img


def test_gaussian_filter(sample_image):
    from app.core.preprocessor import ImagePreprocessor
    p = ImagePreprocessor()
    result = p.apply_gaussian(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    # El filtro gaussiano debe reducir la varianza
    assert np.std(result.astype(float)) < np.std(sample_image.astype(float)) + 5


def test_median_filter(sample_image):
    from app.core.preprocessor import ImagePreprocessor
    p = ImagePreprocessor()
    result = p.apply_median(sample_image)
    assert result.shape == sample_image.shape


def test_clahe(sample_image):
    from app.core.preprocessor import ImagePreprocessor
    p = ImagePreprocessor()
    result = p.apply_clahe(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_full_preprocess_pipeline(sample_image):
    from app.core.preprocessor import ImagePreprocessor
    p = ImagePreprocessor()
    result = p.preprocess(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8


def test_canny_edges(sample_image):
    from app.core.segmentor import ImageSegmentor
    s = ImageSegmentor()
    edges = s.apply_canny(sample_image)
    assert edges.shape == sample_image.shape
    # Deben existir bordes en la imagen sintética
    assert edges.max() > 0


def test_sobel_edges(sample_image):
    from app.core.segmentor import ImageSegmentor
    s = ImageSegmentor()
    sobel = s.apply_sobel(sample_image)
    assert sobel.shape == sample_image.shape
    assert sobel.max() > 0


def test_otsu_threshold(sample_image):
    from app.core.segmentor import ImageSegmentor
    s = ImageSegmentor()
    binary = s.apply_otsu_threshold(sample_image)
    # Resultado debe ser binario (solo 0 y 255)
    unique_vals = np.unique(binary)
    assert set(unique_vals).issubset({0, 255})


def test_radiolucent_detection(sample_image):
    from app.core.segmentor import ImageSegmentor
    s = ImageSegmentor()
    mask, regions = s.detect_radiolucent_regions(sample_image, darkness_threshold=80)
    assert mask.shape == sample_image.shape
    assert isinstance(regions, list)
    # La imagen tiene zona oscura (30 gris), debe detectar al menos una región
    assert len(regions) >= 1


def test_feature_extraction(sample_image):
    from app.core.feature_extractor import FeatureExtractor
    e = FeatureExtractor()
    features = e.extract(sample_image)

    expected_keys = [
        "media", "std", "min_px", "max_px",
        "bordes_mean", "sobel_mean",
        "zona_tl", "zona_tr", "zona_bl", "zona_br",
        "prop_oscuros", "asimetria",
    ]
    for key in expected_keys:
        assert key in features, f"Feature '{key}' no encontrada"
        assert isinstance(features[key], float)

    # Verificar rangos
    assert 0 <= features["media"] <= 255
    assert 0 <= features["std"] <= 128
    assert 0 <= features["prop_oscuros"] <= 1


def test_feature_array_shape(sample_image):
    from app.core.feature_extractor import FeatureExtractor
    e = FeatureExtractor()
    arr = e.extract_to_array(sample_image)
    assert arr.shape == (12,)
    assert arr.dtype == np.float32


def test_classifier_demo_mode(sample_image):
    """El clasificador en modo demo debe retornar una estructura válida."""
    from app.core.feature_extractor import FeatureExtractor
    from app.core.classifier import DentalClassifier

    e = FeatureExtractor()
    features = e.extract(sample_image)

    clf = DentalClassifier()
    result = clf.classify(features)

    assert "predicted_class" in result
    assert "predicted_label" in result
    assert "confidence_score" in result
    assert "class_probabilities" in result
    assert 0 <= result["predicted_class"] <= 4
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_normalize_to_8bit():
    from app.core.loader import ImageLoader
    loader = ImageLoader()
    img_16bit = np.array([[0, 32768, 65535]], dtype=np.uint16)
    result = loader._normalize_to_8bit(img_16bit)
    assert result.dtype == np.uint8
    assert result[0, 0] == 0
    assert result[0, 2] == 255


def test_resize_periapical(sample_image):
    from app.core.preprocessor import ImagePreprocessor
    p = ImagePreprocessor()
    resized = p.resize_standard(sample_image, "periapical")
    assert resized.shape == (512, 512)


def test_preprocessor_rejects_3d_image():
    from app.core.preprocessor import ImagePreprocessor
    from app.exceptions.custom import ProcessingError
    p = ImagePreprocessor()
    rgb_img = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ProcessingError):
        p.preprocess(rgb_img)
