"""
Configuraciones centralizadas para todos los datasets del TFG
==============================================================

Este archivo contiene las configuraciones para cada dataset utilizado
en el proyecto de Curriculum Learning.

Uso:
    from dataset_configs import get_config
    
    config = get_config('digits')  
    batch_size = config.batch_size
"""

import os


class SuffixedName(str):
    """
    Subclase de str pensada para usarse como `config.name`.

    Mantiene un atributo interno `_base_name` que se usa en TODAS las
    comparaciones de igualdad (==, !=, in, hash). En interpolaciones
    string (f"...{name}", str(name), concatenación), se renderiza con
    el sufijo aplicado.

    Esto permite añadir un sufijo de ejecución (p.ej. "_noiseA_10") sin
    romper código que compare config.name contra strings literales.
    """

    def __new__(cls, full_name, base_name=None):
        instance = super().__new__(cls, full_name)
        instance._base_name = base_name if base_name is not None else full_name
        return instance

    def __eq__(self, other):
        if isinstance(other, str):
            return self._base_name == other
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return hash(self._base_name)


def _get_noise_suffix():
    """Lee la variable de entorno NOISE_SUFFIX. Devuelve '' si no está."""
    return os.environ.get('NOISE_SUFFIX', '').strip()


def _apply_noise_suffix(config):
    """
    Aplica el sufijo de NOISE_SUFFIX (si existe) a output_path y name.
    Modifica el config in-place. No tiene efecto si NOISE_SUFFIX no está
    definida o está vacía.
    """
    suffix = _get_noise_suffix()
    if not suffix:
        return config

    # Aplicar sufijo a output_path (carpeta del dataset organizado)
    if config.output_path is not None:
        config.output_path = config.output_path + suffix

    # Aplicar sufijo a name como SuffixedName: las comparaciones usarán el
    # nombre base original, pero las interpolaciones en strings (RESULTS_DIR)
    # incluirán el sufijo, lo que separa los directorios de resultados.
    if config.name is not None:
        config.name = SuffixedName(config.name + suffix, base_name=config.name)

    return config


class DatasetConfig:
    """Clase base para configuración de datasets"""
    
    def __init__(self):
        # Identificación
        self.name = None
        
        # Paths
        self.input_path = None
        self.output_path = None
        
        # Propiedades de imagen
        self.num_classes = None
        self.image_size = None      # Tamaño original
        self.resize_to = None        # Tamaño para entrenamiento
        self.channels = None
        
        # Parámetros de clustering
        self.n_clusters_per_class = None
        self.n_clusters_global = None
        
        # Parámetros de entrenamiento
        self.batch_size = None
        self.learning_rate = None
        self.epochs = None           # Para autoencoder
        self.latent_dim = None


class DigitsConfig(DatasetConfig):
    """
    Configuración para dataset Digits (dígitos manuscritos)
    
    Dataset: ~10,160 imágenes de dígitos 0-9
    Resolución: 128x128
    Clases: 10 (dígitos 0-9)
    """
    
    def __init__(self):
        super().__init__()
        
        # Identificación
        self.name = "digits"
        
        # Paths
        self.input_path = "./datos/digits_updated"
        self.output_path = "./datos_procesados/dataset_digits_organized"
        
        # Propiedades de imagen
        self.num_classes = 10
        self.image_size = 128
        self.resize_to = 128         # No necesita resize
        self.channels = 3            # RGB
        
        # Parámetros de clustering
        self.n_clusters_per_class = 15
        self.n_clusters_global = 10
        
        # Parámetros de entrenamiento
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 50             # Autoencoder
        self.latent_dim = 32


class CIFAR10Config(DatasetConfig):
    """
    Configuración para CIFAR-10
    
    Dataset: 60,000 imágenes (50,000 train + 10,000 test)
    Selección: 5000 imágenes aleatorias del conjunto completo
    Resolución: 32x32 → resize a 128x128
    Clases: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    Conversión: RGB → Escala de grises (para coherencia con Digits)
    """
    
    def __init__(self):
        super().__init__()
        
        # Identificación
        self.name = "cifar10"
        
        # Paths
        self.input_path = "./datos/cifar-10-batches-py"  # Directorio con pickles
        self.output_path = "./datos_procesados/dataset_cifar10_organized"
        
        # Propiedades de imagen
        self.num_classes = 10
        self.image_size = 32         # Original
        self.resize_to = 128         # Resize para coherencia con Digits
        self.channels = 1            # Convertido a escala de grises
        
        # Selección de datos
        self.n_samples = 20000        # Número de imágenes a seleccionar aleatoriamente
        
        # Parámetros de clustering
        self.n_clusters_per_class = 15  # Igual que Digits
        self.n_clusters_global = 10
        
        # Parámetros de entrenamiento
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 50
        self.latent_dim = 32


class CIFAR10RGBConfig(DatasetConfig):
    """
    Configuración para CIFAR-10 en RGB (SIN conversión a grayscale)
    
    Dataset: 60,000 imágenes (50,000 train + 10,000 test)
    Selección: 5000 imágenes aleatorias del conjunto completo
    Resolución: 32x32 → resize a 128x128
    Clases: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    Color: RGB NATIVO (sin conversión a grayscale)
    
    DIFERENCIA CON CIFAR10Config:
    - Mantiene 3 canales RGB en vez de convertir a grayscale
    - Usa normalización específica de CIFAR-10 RGB
    - Output en carpeta separada para no sobrescribir versión grayscale
    """
    
    def __init__(self):
        super().__init__()
        
        # Identificación
        self.name = "cifar10rgb"
        
        # Paths
        self.input_path = "./datos/cifar-10-batches-py"  # Mismo dataset original
        self.output_path = "./datos_procesados/dataset_cifar10rgb_organized"  # Nueva carpeta
        
        # Propiedades de imagen
        self.num_classes = 10
        self.image_size = 32         # Original
        self.resize_to = 128         # Resize para coherencia con Digits
        self.channels = 3            # RGB verdadero (sin conversión)
        
        # Selección de datos
        self.n_samples = 10000        # Número de imágenes a seleccionar aleatoriamente
        
        # Parámetros de clustering
        self.n_clusters_per_class = 15  # Igual que Digits
        self.n_clusters_global = 10
        
        # Parámetros de entrenamiento
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 50
        self.latent_dim = 32


# =============================================================================
# REGISTRO DE DATASETS DISPONIBLES
# =============================================================================

DATASET_CONFIGS = {
    'digits': DigitsConfig,
    'cifar10': CIFAR10Config,
    'cifar10rgb': CIFAR10RGBConfig,
}


def get_config(dataset_name):
    """
    Obtener configuración para un dataset específico
    
    Args:
        dataset_name (str): Nombre del dataset
                           Opciones: 'digits'
    
    Returns:
        DatasetConfig: Instancia de la configuración correspondiente
    
    Raises:
        ValueError: Si el dataset no está registrado
    
    Ejemplo:
        >>> config = get_config('digits')
        >>> print(config.batch_size)
        64
        >>> print(config.image_size)
        128
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' no reconocido.\n"
            f"Datasets disponibles: {available}"
        )

    config = DATASET_CONFIGS[dataset_name]()
    # Aplicar sufijo de la variable de entorno NOISE_SUFFIX si existe.
    # Si no está definida o está vacía, esto es un no-op.
    config = _apply_noise_suffix(config)
    return config


def list_available_datasets():
    """
    Listar todos los datasets disponibles
    
    Returns:
        list: Lista de nombres de datasets disponibles
    """
    return list(DATASET_CONFIGS.keys())


def print_config_summary(dataset_name):
    """
    Imprimir resumen de configuración de un dataset
    
    Args:
        dataset_name (str): Nombre del dataset
    """
    config = get_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"CONFIGURACIÓN: {config.name.upper()}")
    print(f"{'='*60}")
    print(f"\n PATHS:")
    print(f"   Input:  {config.input_path}")
    print(f"   Output: {config.output_path}")
    print(f"\n  IMÁGENES:")
    print(f"   Tamaño original: {config.image_size}×{config.image_size}")
    print(f"   Resize para entrenamiento: {config.resize_to}×{config.resize_to}")
    print(f"   Canales: {config.channels}")
    if config.num_classes:
        print(f"   Clases: {config.num_classes}")
    print(f"\n CLUSTERING:")
    print(f"   Clusters por clase: {config.n_clusters_per_class}")
    print(f"   Clusters globales: {config.n_clusters_global}")
    print(f"   Dimensión latente: {config.latent_dim}")
    print(f"\n ENTRENAMIENTO:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs (autoencoder): {config.epochs}")
    print(f"{'='*60}\n")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de configuraciones
    """
    print("\n" + "="*60)
    print("EJEMPLO DE USO: dataset_configs.py")
    print("="*60)
    
    # Listar datasets disponibles
    print("\n Datasets disponibles:")
    for dataset in list_available_datasets():
        print(f"   - {dataset}")
    
    # Mostrar configuración de Digits
    print_config_summary('digits')
    
    
    # Ejemplo de uso programático
    print("\n Ejemplo de uso programático:")
    print("="*60)
    
    config = get_config('digits')
    print(f"\nconfig = get_config('digits')")
    print(f"config.batch_size = {config.batch_size}")
    print(f"config.learning_rate = {config.learning_rate}")
    print(f"config.input_path = '{config.input_path}'")
    
    print("\n" + "="*60)
