"""
DentaScan — Extractor de features profundos (ResNet50 ImageNet).

Carga lazy de ResNet50 pre-entrenado en ImageNet (descarga 1ª vez ~98MB).
Extrae 2048 features profundos por imagen para alimentar al clasificador.

Notas:
  - ImageNet no es dental, pero los pesos generan features visuales
    genéricos (texturas, bordes, formas) muy potentes para transfer learning.
  - Si torch no está instalado, las funciones retornan None con un warning.
"""

from __future__ import annotations

from typing import Optional
import threading

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

_torch_available: Optional[bool] = None
_model = None
_preprocess = None
_device = None
_lock = threading.Lock()


def torch_available() -> bool:
    """Detecta si torch/torchvision están instalados (cacheado)."""
    global _torch_available
    if _torch_available is not None:
        return _torch_available
    try:
        import torch                          # noqa: F401
        import torchvision                    # noqa: F401
        _torch_available = True
    except ImportError as exc:
        logger.warning("torch/torchvision no disponibles: %s", exc)
        _torch_available = False
    return _torch_available


def _load_model():
    """Carga lazy de ResNet50 con cabeza removida. Thread-safe."""
    global _model, _preprocess, _device
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from torchvision.models import ResNet50_Weights

        _device = torch.device("cpu")
        logger.info("Cargando ResNet50 ImageNet (esto puede tardar la 1ª vez)...")

        weights = ResNet50_Weights.IMAGENET1K_V2
        m = models.resnet50(weights=weights)
        # Reemplaza la cabeza FC por Identity → salida de 2048 features
        m.fc = nn.Identity()
        m.eval()
        m.to(_device)

        # Preprocesamiento estándar de ImageNet
        _preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        _model = m
        logger.info("ResNet50 listo (2048 features, device=%s).", _device)
        return _model


def extract_resnet50_features(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extrae un vector de 2048 features profundos.

    Args:
        image: ndarray 2D (escala de grises) o 3D HxWxC

    Returns:
        np.ndarray shape (2048,) o None si torch no está disponible.
    """
    if not torch_available():
        return None

    import torch

    # Convertir gris → RGB (ResNet espera 3 canales)
    if image.ndim == 2:
        img_rgb = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        img_rgb = np.concatenate([image, image, image], axis=-1)
    else:
        img_rgb = image

    img_rgb = img_rgb.astype(np.uint8)

    try:
        model = _load_model()
        tensor = _preprocess(img_rgb).unsqueeze(0).to(_device)
        with torch.no_grad():
            feats = model(tensor).cpu().numpy().flatten()
        return feats.astype(np.float32)
    except Exception as exc:
        logger.exception("Error extrayendo features ResNet50: %s", exc)
        return None
