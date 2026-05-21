"""
DentaScan — Excepciones personalizadas del dominio.
"""


class DentaScanException(Exception):
    """Base de todas las excepciones del sistema."""
    pass


class ImageLoadError(DentaScanException):
    """Error al cargar o leer un archivo de imagen."""
    pass


class UnsupportedFormatError(DentaScanException):
    """Formato de archivo no soportado por el sistema."""
    pass


class FileTooLargeError(DentaScanException):
    """El archivo supera el límite de tamaño permitido."""
    pass


class ProcessingError(DentaScanException):
    """Error durante el pipeline de procesamiento de imágenes."""
    pass


class ModelNotFoundError(DentaScanException):
    """El modelo ML solicitado no existe o no está entrenado."""
    pass


class FeatureExtractionError(DentaScanException):
    """Error al extraer características de la imagen."""
    pass


class ClassificationError(DentaScanException):
    """Error durante la clasificación con el modelo ML."""
    pass


class DicomReadError(DentaScanException):
    """Error al leer o parsear un archivo DICOM."""
    pass


class AnalysisNotFoundError(DentaScanException):
    """Análisis no encontrado en la base de datos."""
    pass


class UnauthorizedError(DentaScanException):
    """El usuario no tiene permisos para esta operación."""
    pass
