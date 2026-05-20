"""
Sistema de Clustering con Autoencoder + K-Means para Curriculum Learning
=========================================================================

Este script organiza el dataset en grupos de dificultad configurables
usando dos niveles de granularidad:

Framework: PyTorch
"""

import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Usar backend sin interfaz gráfica
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
import argparse
from dataset_configs import get_config

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración centralizada del pipeline"""
    
    @staticmethod
    def from_dataset_config(dataset_config):
        """Crear Config desde DatasetConfig"""
        config = Config()
        
        # Paths
        config.INPUT_PATH = dataset_config.input_path
        config.OUTPUT_PATH = dataset_config.output_path
        config.METADATA_FILE = "clustering_metadata.json"
        
        # Parámetros de imágenes
        config.IMAGE_SIZE = dataset_config.resize_to  # IMPORTANTE: usar resize_to
        
        # NUEVO: Detectar si es RGB
        config.IS_RGB = dataset_config.name in ['cifar10rgb']
        config.INPUT_CHANNELS = 3 if config.IS_RGB else 1
        config.DATASET_NAME = dataset_config.name
        
        # Parámetros de entrenamiento
        config.BATCH_SIZE = dataset_config.batch_size
        config.LEARNING_RATE = dataset_config.learning_rate
        config.EPOCHS = dataset_config.epochs
        config.LATENT_DIM = dataset_config.latent_dim
        
        # Parámetros de clustering
        config.N_CLUSTERS_PER_DIGIT = dataset_config.n_clusters_per_class
        config.N_CLUSTERS_GLOBAL = dataset_config.n_clusters_global
        
        # Device
        config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        config.DIGITS = list(range(10))
        
        return config


# ============================================================================
# AUTOENCODER
# ============================================================================

class ConvAutoencoder(nn.Module):
    """
    Autoencoder Convolucional adaptable para diferentes tamaños de imagen.
    Soporta 128×128 (Digits, CIFAR-10 redimensionado)
    NUEVO: Soporta 1 canal (grayscale) o 3 canales (RGB).
    """
    def __init__(self, latent_dim=32, input_size=128, input_channels=1):
        super(ConvAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.input_channels = input_channels
        
        # Calcular tamaño después de 4 capas con stride=2
        # 128×128 -> 8×8
        # 256×256 -> 16×16
        final_size = input_size // 16
        
        # Encoder: reduce progresivamente
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # /4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # /8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # /16
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Flatten(),
            nn.Linear(256 * final_size * final_size, latent_dim)
        )
        
        # Decoder: reconstruye desde latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * final_size * final_size),
            nn.ReLU(),
            
            nn.Unflatten(1, (256, final_size, final_size)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # ×2
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # ×4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # ×8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),    # ×16
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


# ============================================================================
# DATASET
# ============================================================================

def load_cifar10_batch(filepath):
    """Carga un batch de CIFAR-10 desde archivo pickle"""
    import pickle
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_and_select_cifar10(cifar_dir, n_samples=5000, seed=42):
    """
    Carga CIFAR-10 desde archivos pickle y selecciona n_samples aleatorias.
    
    Args:
        cifar_dir: Directorio con archivos data_batch_* y test_batch
        n_samples: Número de imágenes a seleccionar
        seed: Semilla para reproducibilidad
    
    Returns:
        images: Array numpy (n_samples, 32, 32, 3)
        labels: Array numpy (n_samples,)
    """
    np.random.seed(seed)
    
    print(f"\n Cargando CIFAR-10 desde {cifar_dir}...")
    
    all_images = []
    all_labels = []
    
    # Cargar batches de entrenamiento
    for i in range(1, 6):
        batch_file = Path(cifar_dir) / f'data_batch_{i}'
        if batch_file.exists():
            batch = load_cifar10_batch(batch_file)
            images = batch[b'data']
            labels = batch[b'labels']
            
            # Reshape: (10000, 3072) → (10000, 32, 32, 3)
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            
            all_images.append(images)
            all_labels.extend(labels)
    
    # Cargar batch de test
    test_file = Path(cifar_dir) / 'test_batch'
    if test_file.exists():
        batch = load_cifar10_batch(test_file)
        images = batch[b'data']
        labels = batch[b'labels']
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        all_images.append(images)
        all_labels.extend(labels)
    
    # Concatenar todo
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)
    
    total = len(all_images)
    print(f"   Total cargado: {total} imágenes")
    
    # Seleccionar aleatoriamente n_samples
    if n_samples < total:
        indices = np.random.choice(total, size=n_samples, replace=False)
        selected_images = all_images[indices]
        selected_labels = all_labels[indices]
        print(f"   Seleccionadas {n_samples} imágenes aleatorias")
    else:
        selected_images = all_images
        selected_labels = all_labels
        print(f"   Usando todas las {total} imágenes")
    
    return selected_images, selected_labels


class DigitDataset(Dataset):
    """
    Dataset unificado para Digits y CIFAR-10.
    - Digits: Carga desde carpetas 0-9
    - CIFAR-10: Usa imágenes pre-cargadas desde pickle
    """
    
    def __init__(self, root_dir, transform=None, specific_digit=None, 
                 cifar_images=None, cifar_labels=None):
        """
        Args:
            root_dir: Path al directorio (para Digits)
            transform: Transformaciones
            specific_digit: Filtrar por un dígito específico (para Digits)
            cifar_images: Array numpy con imágenes CIFAR-10 (opcional)
            cifar_labels: Array numpy con labels CIFAR-10 (opcional)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.is_cifar = cifar_images is not None
        
        if self.is_cifar:
            # Modo CIFAR-10: usar arrays pre-cargados
            self.cifar_images = cifar_images
            self.cifar_labels = cifar_labels
            
            if specific_digit is not None:
                # Filtrar por clase específica
                mask = self.cifar_labels == specific_digit
                self.indices = np.where(mask)[0]
            else:
                self.indices = np.arange(len(self.cifar_labels))
            
            print(f"Cargadas {len(self.indices)} imágenes (CIFAR-10)")
            
        else:
            # Modo Digits: cargar desde carpetas
            digits = [specific_digit] if specific_digit is not None else Config.DIGITS
            
            for digit in digits:
                digit_dir = self.root_dir / str(digit)
                if not digit_dir.exists():
                    continue
                
                for img_path in digit_dir.glob("*.png"):
                    self.images.append(str(img_path))
                    self.labels.append(digit)
            
            print(f"Cargadas {len(self.images)} imágenes")
    
    def __len__(self):
        if self.is_cifar:
            return len(self.indices)
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.is_cifar:
            # CIFAR-10: obtener de arrays
            real_idx = self.indices[idx]
            image = self.cifar_images[real_idx]  # (32, 32, 3) numpy array
            label = int(self.cifar_labels[real_idx])
            
            # Convertir a PIL Image
            image = Image.fromarray(image.astype('uint8'), mode='RGB')
            
            # NUEVO: Solo convertir a grayscale si NO es RGB
            if not Config.IS_RGB:
                image = image.convert('L')
            
            # Path dummy (para compatibilidad)
            img_path = f"cifar10_class{label}_idx{real_idx}"
            
        else:
            # Digits: cargar desde archivo
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # NUEVO: Cargar RGB o grayscale según configuración
            if Config.IS_RGB:
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


# ============================================================================
# ENTRENAMIENTO DEL AUTOENCODER
# ============================================================================

def train_autoencoder(model, dataloader, epochs=50):
    """Entrena el autoencoder"""
    
    print("\n Entrenando Autoencoder...")
    print(f"   Device: {Config.DEVICE}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    
    model = model.to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, _, _ in progress_bar:
            images = images.to(Config.DEVICE)
            
            # Forward pass
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    print("Autoencoder entrenado!")
    return model, losses


# ============================================================================
# EXTRACCIÓN DE EMBEDDINGS
# ============================================================================

def extract_embeddings(model, dataloader):
    """Extrae los embeddings (representaciones latentes) de todas las imágenes"""
    
    print("\n Extrayendo embeddings...")
    
    model.eval()
    embeddings = []
    labels = []
    paths = []
    
    with torch.no_grad():
        for images, lbls, img_paths in tqdm(dataloader, desc="Extrayendo embeddings"):
            images = images.to(Config.DEVICE)
            _, z = model(images)
            
            embeddings.append(z.cpu().numpy())
            labels.extend(lbls.numpy())
            paths.extend(img_paths)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    print(f" Embeddings extraídos: {embeddings.shape}")
    
    return embeddings, labels, paths


# ============================================================================
# RUIDO EN ETIQUETAS
# ============================================================================

def apply_label_noise(labels_clean, noise_level, n_classes, seed=42):
    """
    Aplica ruido aleatorio uniforme a un porcentaje de las etiquetas.

    Para cada imagen seleccionada (con probabilidad noise_level), su etiqueta
    se reemplaza por una elegida uniformemente al azar entre las (n_classes-1)
    clases restantes. Es decir: si una imagen es seleccionada como ruidosa,
    su etiqueta CAMBIA con seguridad (no puede coincidir con la original).

    Args:
        labels_clean: array (N,) con las etiquetas originales.
        noise_level:  fracción de etiquetas a corromper (0.0 a 1.0).
        n_classes:    número total de clases del dataset.
        seed:         semilla para reproducibilidad.

    Returns:
        labels_noisy: array (N,) con las etiquetas corruptas/originales.
        is_noisy:     array booleano (N,) que marca qué imágenes fueron ruidificadas.
    """
    rng = np.random.default_rng(seed)
    N = len(labels_clean)

    if noise_level <= 0.0:
        return labels_clean.copy(), np.zeros(N, dtype=bool)

    # Selección aleatoria de qué imágenes son ruidificadas
    n_noisy = int(round(N * noise_level))
    noisy_indices = rng.choice(N, size=n_noisy, replace=False)
    is_noisy = np.zeros(N, dtype=bool)
    is_noisy[noisy_indices] = True

    # Para cada imagen ruidificada, elegir una etiqueta DIFERENTE a la original
    labels_noisy = labels_clean.copy()
    for idx in noisy_indices:
        original = int(labels_clean[idx])
        # Choose uniformly among the other (n_classes - 1) classes
        candidates = [c for c in range(n_classes) if c != original]
        labels_noisy[idx] = rng.choice(candidates)

    # Verificación: todas las imágenes ruidosas tienen etiqueta distinta
    assert np.all(labels_noisy[is_noisy] != labels_clean[is_noisy]), \
        "Inconsistencia: alguna etiqueta ruidosa coincide con la original."

    print(f"   Ruido aplicado: {n_noisy}/{N} etiquetas modificadas "
          f"({100*n_noisy/N:.2f}% real)")

    return labels_noisy, is_noisy


# ============================================================================
# CLUSTERING
# ============================================================================

def perform_clustering(embeddings, labels, paths,
                       labels_clean=None, labels_noisy=None,
                       is_noisy=None, noise_mode='clean'):
    """
    K-Means + Ordenar por distancia - NIVELES BALANCEADOS
    
    Proceso:
    1. K-Means POR CLASE (para agrupar por similaridad)
    2. Dentro de cada clase, ordenar por distancia al centroide del cluster
    3. Dividir en 10 niveles BALANCEADOS (mismo número de imágenes por nivel)
    
    Para CIFAR-10 (5000 imágenes, 10 clases):
    - 500 imágenes por clase
    - 50 imágenes por nivel por clase
    - 500 imágenes por nivel TOTAL (perfectamente balanceado)
    
    Ventajas:
    - Niveles PERFECTAMENTE BALANCEADOS
    - Cada clase contribuye por igual en cada nivel
    - Combina agrupación (K-Means) + ordenamiento (distancia)
    """
    
    print("\n MÉTODO: K-Means + Distancia (Balanceado)")
    print("   Características: Niveles balanceados, combina agrupación y orden")
    
    metadata = []
    unique_labels = np.unique(labels)
    is_multiclass = len(unique_labels) > 1
    
    digit_clusters = {}
    kmeans_models = {}
    
    # ========================================================================
    # Clustering POR CLASE + Ordenar por distancia
    # ========================================================================
    
    print(f"\n   Clustering por clase con ordenamiento por distancia")
    
    for clase in unique_labels:
        # Filtrar embeddings de esta clase
        class_mask = labels == clase
        class_embeddings = embeddings[class_mask]
        class_paths = np.array(paths)[class_mask]
        class_indices = np.where(class_mask)[0]
        
        if len(class_embeddings) == 0:
            continue
        
        # K-Means para agrupar por similaridad
        n_clusters_class = min(10, len(class_embeddings) // 10)
        n_clusters_class = max(2, n_clusters_class)
        
        kmeans = KMeans(n_clusters=n_clusters_class, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)
        
        # Calcular distancias al centroide de cada cluster
        distances = np.zeros(len(class_embeddings))
        for cluster_id in range(n_clusters_class):
            cluster_mask_local = cluster_labels == cluster_id
            cluster_centroid = kmeans.cluster_centers_[cluster_id]
            cluster_distances = np.linalg.norm(
                class_embeddings[cluster_mask_local] - cluster_centroid, axis=1
            )
            distances[cluster_mask_local] = cluster_distances
        
        # Ordenar por distancia (fácil → difícil)
        sorted_indices = np.argsort(distances)
        
        # Dividir en 10 niveles balanceados
        n_images = len(class_embeddings)

        # PROTECCIÓN: Si hay muy pocas imágenes
        if n_images < Config.N_CLUSTERS_GLOBAL:
            print(f"      Advertencia: Clase {int(clase)} tiene solo {n_images} imágenes")
            print(f"         (menos que {Config.N_CLUSTERS_GLOBAL} niveles). Usando niveles simples.")
            # Asignar niveles simples sin división
            difficulty_levels_class = np.arange(n_images) % Config.N_CLUSTERS_GLOBAL
        else:
            # Cálculo normal
            images_per_level = n_images // Config.N_CLUSTERS_GLOBAL
            
            # Asignar nivel de dificultad
            difficulty_levels_class = np.zeros(n_images, dtype=int)
            for i, sorted_idx in enumerate(sorted_indices):
                level = min(i // images_per_level, Config.N_CLUSTERS_GLOBAL - 1)
                difficulty_levels_class[sorted_idx] = level
        
        # Guardar info
        digit_clusters[int(clase)] = {
            'cluster_labels': cluster_labels,
            'distances': distances,
            'paths': class_paths,
            'difficulty_levels': difficulty_levels_class,
            'indices': class_indices
        }
        
        kmeans_models[int(clase)] = kmeans
        
        print(f"   Clase {clase}: {n_clusters_class} clusters K-Means, "
              f"{n_images} imágenes = {images_per_level} por nivel")
    
    # ========================================================================
    # Train/val split estratificado
    # ========================================================================
    
    print("\n   Generando metadata con split train/val...")
    
    train_indices_global = []
    val_indices_global = []
    
    np.random.seed(42)
    
    for clase in unique_labels:
        if int(clase) not in digit_clusters:
            continue
        
        class_indices = digit_clusters[int(clase)]['indices']
        
        # Shuffle
        shuffled = class_indices.copy()
        np.random.shuffle(shuffled)
        
        # Split 80/20
        split_idx = int(len(shuffled) * 0.8)
        train_indices_global.extend(shuffled[:split_idx])
        val_indices_global.extend(shuffled[split_idx:])
    
    train_indices_set = set(train_indices_global)
    val_indices_set = set(val_indices_global)
    
    print(f"   Split: {len(train_indices_global)} train, {len(val_indices_global)} val")
    
    # ========================================================================
    # Crear metadata
    # ========================================================================

    # Si labels_clean no se ha pasado, asumimos que coincide con labels (modo clean)
    if labels_clean is None:
        labels_clean = labels
    if labels_noisy is None:
        labels_noisy = labels
    if is_noisy is None:
        is_noisy = np.zeros(len(labels), dtype=bool)

    for i, (path, label) in enumerate(zip(paths, labels)):
        
        if int(label) in digit_clusters:
            cluster_info = digit_clusters[int(label)]
            
            # Encontrar índice dentro de la clase
            class_path_idx = np.where(cluster_info['paths'] == path)[0][0]
            
            digit_cluster = int(cluster_info['cluster_labels'][class_path_idx])
            digit_distance = float(cluster_info['distances'][class_path_idx])
            difficulty_level = int(cluster_info['difficulty_levels'][class_path_idx])
        else:
            digit_cluster = 0
            digit_distance = 0.0
            difficulty_level = 0
        
        is_train = i in train_indices_set
        split = 'train' if is_train else 'val'

        # Etiqueta efectiva que se usará para guardar el archivo en su carpeta
        # de clase. Reglas:
        #   - clean          → siempre la etiqueta limpia
        #   - pre_clustering → ruidosa SOLO si split=='train' Y is_noisy[i].
        #                      val siempre limpio (requisito: ruido solo en train).
        #                      El clustering ya operó con etiquetas ruidosas, lo
        #                      que afecta a la asignación de niveles, pero la
        #                      etiqueta visible al clasificador en val es limpia.
        clean_label = int(labels_clean[i])
        is_noisy_i = bool(is_noisy[i])
        if noise_mode in ('pre_clustering', 'train_only'):
            if split == 'train' and is_noisy_i:
                effective_label = int(labels_noisy[i])
            else:
                effective_label = clean_label
        else:  # clean
            effective_label = clean_label

        metadata.append({
            'path': path,
            'digit': effective_label,           # Etiqueta efectiva (la usada al guardar y entrenar)
            'digit_clean': clean_label,         # Etiqueta original limpia (siempre)
            'is_noisy': is_noisy_i,             # ¿La etiqueta fue ruidificada en origen?
            'noise_mode': noise_mode,           # Para trazabilidad
            'split': split,
            'digit_cluster': digit_cluster,
            'digit_distance': digit_distance,
            'global_cluster': digit_cluster,    # Por compatibilidad
            'global_distance': digit_distance,
            'difficulty_level': difficulty_level,
            'embedding': embeddings[i].tolist()
        })
    
    print(f" Metadata generada para {len(metadata)} imágenes")
    
    # Mostrar distribución
    print("\n    Distribución por nivel de dificultad:")
    level_counts = {}
    for level in range(Config.N_CLUSTERS_GLOBAL):
        count = sum(1 for item in metadata if item['difficulty_level'] == level)
        level_counts[level] = count
        percentage = (count / len(metadata)) * 100
        bar = '-' * max(1, int(percentage / 2))
        print(f"      Level {level:02d}: {count:5d} imágenes ({percentage:5.2f}%) {bar}")
    
    # Calcular estadísticas de uniformidad
    counts = list(level_counts.values())
    avg_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = np.min(counts)
    max_count = np.max(counts)
    print(f"\n    Estadísticas de distribución:")
    print(f"      Promedio por nivel: {avg_count:.1f} imágenes")
    print(f"      Desviación estándar: {std_count:.1f}")
    print(f"      Rango: {min_count} - {max_count} imágenes")
    print(f"      Coeficiente de variación: {(std_count/avg_count)*100:.1f}% "
          f"(cuanto menor, más uniforme)")
    


    # ================================================================
    # GUARDAR METADATA PARA VISUALIZACIONES
    # ================================================================

    print("\n Guardando metadata del clustering para visualizaciones...")

    # Extraer arrays de metadata ya creada
    difficulty_levels_array = np.array([item['difficulty_level'] for item in metadata])
    cluster_labels_array = np.array([item['digit_cluster'] for item in metadata])
    distances_array = np.array([item['digit_distance'] for item in metadata])

    # Preparar metadata completa para visualizaciones
    metadata_for_vis = {
        'embeddings': embeddings,
        'labels': labels,
        'paths': paths,
        'cluster_labels': cluster_labels_array,
        'distances': distances_array,
        'difficulty_levels': difficulty_levels_array,
        'kmeans_models': kmeans_models,
        'digit_clusters': digit_clusters,
        'n_clusters_per_class': Config.N_CLUSTERS_PER_DIGIT,
    }

    # Guardar metadata
    import pickle
    metadata_path = Path(Config.OUTPUT_PATH) / 'clustering_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_for_vis, f)

    print(f"    Metadata guardada en: {metadata_path}")

    # ================================================================


    return metadata, digit_clusters, kmeans_models


# ============================================================================
# ORGANIZACIÓN DE ARCHIVOS
# ============================================================================

def organize_dataset(metadata, cifar_images=None, cifar_labels=None):
    """
    Organiza el dataset en carpetas train/val según nivel de dificultad.
    - Multi-clase (Digits): level_XX/digito/
    
    Args:
        metadata: Lista con información de cada imagen
        cifar_images: Array numpy con imágenes CIFAR-10 (opcional)
        cifar_labels: Array numpy con labels CIFAR-10 (opcional)
    """
    
    print("\n  Organizando dataset en train/val...")
    
    output_path = Path(Config.OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Detectar si es CIFAR-10
    is_cifar = cifar_images is not None
    
    # Detectar si es multi-clase o mono-clase
    unique_labels = set(item['digit'] for item in metadata)
    is_multiclass = len(unique_labels) > 1
    
    if is_multiclass:
        print(f"    Multi-clase detectado → Estructura: level_XX/digito/")
        # Crear estructura con subcarpetas por dígito
        for split in ['train', 'val']:
            for level in range(Config.N_CLUSTERS_GLOBAL):
                for digit in unique_labels:
                    split_dir = output_path / split / f"level_{level:02d}" / str(digit)
                    split_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"    Mono-clase detectado → Estructura: level_XX/")
        # Crear estructura sin subcarpetas por dígito
        for split in ['train', 'val']:
            for level in range(Config.N_CLUSTERS_GLOBAL):
                split_dir = output_path / split / f"level_{level:02d}"
                split_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar imágenes
    copied_train = 0
    copied_val = 0
    
    for i, item in enumerate(tqdm(metadata, desc="Guardando imágenes")):
        level = item['difficulty_level']
        digit = item['digit']
        split = item['split']
        
        if is_cifar:
            # ================================================================
            # MODO CIFAR-10: Guardar imagen desde array numpy
            # ================================================================
            
            # Extraer índice real de CIFAR-10 desde el path
            # Format: "cifar10_class{label}_idx{real_idx}"
            path_str = item['path']
            real_idx = int(path_str.split('_idx')[1])
            
            # Obtener imagen del array
            img_array = cifar_images[real_idx]  # (32, 32, 3)
            
            # Convertir a PIL Image
            img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
            
            # NUEVO: Solo convertir a grayscale si NO es RGB
            if not Config.IS_RGB:
                img = img.convert('L')
            
            # Redimensionar a 128×128
            img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.Resampling.LANCZOS)
            
            # Crear nombre de archivo
            filename = f"cifar10_class{digit}_idx{real_idx}.png"
            
            # Crear ruta destino
            if is_multiclass:
                dst_path = output_path / split / f"level_{level:02d}" / str(digit) / filename
            else:
                dst_path = output_path / split / f"level_{level:02d}" / filename
            
            # Guardar imagen
            img.save(dst_path)
            
        else:
            # ================================================================
            # MODO DIGITS Copiar desde archivo existente
            # ================================================================
            
            src_path = Path(item['path'])
            
            if not src_path.exists():
                print(f"   No se encontró: {src_path}")
                continue
            
            # NUEVO: Si es RGB, necesitamos asegurar que la imagen se guarde en RGB
            if Config.IS_RGB:
                # Cargar imagen y asegurar que está en RGB
                img = Image.open(src_path).convert('RGB')
                img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.Resampling.LANCZOS)
                
                # Crear nombre único
                filename = src_path.name
                
                # Crear ruta destino según tipo de dataset
                if is_multiclass:
                    dst_path = output_path / split / f"level_{level:02d}" / str(digit) / filename
                else:
                    dst_path = output_path / split / f"level_{level:02d}" / filename
                
                # Guardar imagen en RGB
                img.save(dst_path)
            else:
                # Modo grayscale: copiar directamente
                # Crear nombre único
                filename = src_path.name
                
                # Crear ruta destino según tipo de dataset
                if is_multiclass:
                    dst_path = output_path / split / f"level_{level:02d}" / str(digit) / filename
                else:
                    dst_path = output_path / split / f"level_{level:02d}" / filename
                
                # Copiar imagen
                shutil.copy2(src_path, dst_path)
        
        if split == 'train':
            copied_train += 1
        else:
            copied_val += 1
    
    print(f"\n Dataset organizado correctamente:")
    print(f"    Train: {copied_train} imágenes")
    print(f"    Val: {copied_val} imágenes")
    print(f"    Total: {copied_train + copied_val} imágenes")
    print(f"    Guardado en: {output_path}")
    
    # Mostrar distribución por split
    print("\n Distribución por nivel de dificultad:")
    print("\n   TRAIN:")
    for level in range(Config.N_CLUSTERS_GLOBAL):
        count = sum(1 for item in metadata if item['difficulty_level'] == level and item['split'] == 'train')
        percentage = (count / copied_train) * 100 if copied_train > 0 else 0
        bar = '_' * int(percentage / 2)
        print(f"      Level {level:02d}: {count:5d} imágenes ({percentage:5.2f}%) {bar}")
    
    print("\n   VALIDATION:")
    for level in range(Config.N_CLUSTERS_GLOBAL):
        count = sum(1 for item in metadata if item['difficulty_level'] == level and item['split'] == 'val')
        percentage = (count / copied_val) * 100 if copied_val > 0 else 0
        bar = '_' * int(percentage / 2)
        print(f"      Level {level:02d}: {count:5d} imágenes ({percentage:5.2f}%) {bar}")


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def visualize_results(embeddings, labels, metadata, save_path):
    """Crea visualizaciones de los resultados del clustering"""
    
    print("\n Generando visualizaciones...")
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. t-SNE de embeddings coloreado por dígito
    # ========================================================================
    
    print("   Generando t-SNE por dígito...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings[:5000])  # Subsample para velocidad
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels[:5000], cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Dígito')
    plt.title('t-SNE de Embeddings - Coloreado por Dígito', fontsize=16)
    plt.xlabel('t-SNE Dimensión 1')
    plt.ylabel('t-SNE Dimensión 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'tsne_by_digit.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 2. t-SNE coloreado por nivel de dificultad
    # ========================================================================
    
    print("   Generando t-SNE por dificultad...")
    
    difficulty_levels = np.array([item['difficulty_level'] for item in metadata[:5000]])
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=difficulty_levels, cmap='RdYlGn_r', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Nivel de Dificultad')
    plt.title('t-SNE de Embeddings - Coloreado por Dificultad', fontsize=16)
    plt.xlabel('t-SNE Dimensión 1')
    plt.ylabel('t-SNE Dimensión 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'tsne_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 3. Distribución de dificultad por dígito
    # ========================================================================
    
    print("   Generando distribución de dificultad...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for digit in Config.DIGITS:
        digit_metadata = [item for item in metadata if item['digit'] == digit]
        difficulties = [item['difficulty_level'] for item in digit_metadata]
        
        axes[digit].hist(difficulties, bins=Config.N_CLUSTERS_GLOBAL, 
                        range=(0, Config.N_CLUSTERS_GLOBAL), 
                        edgecolor='black', alpha=0.7)
        axes[digit].set_title(f'Dígito {digit}', fontsize=12, fontweight='bold')
        axes[digit].set_xlabel('Nivel de Dificultad')
        axes[digit].set_ylabel('Frecuencia')
        axes[digit].grid(True, alpha=0.3)
    
    plt.suptitle('Distribución de Niveles de Dificultad por Dígito', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / 'difficulty_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Visualizaciones guardadas en {save_path}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todo el pipeline"""
    
    # Declarar Config como global para que las funciones puedan acceder
    global Config
    
    # ========================================================================
    # PARSEAR ARGUMENTOS
    # ========================================================================
    
    # Obtener lista de datasets disponibles
    from dataset_configs import list_available_datasets
    available_datasets = list_available_datasets()
    
    parser = argparse.ArgumentParser(
        description='Clustering para Curriculum Learning'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='digits',
        choices=available_datasets,
        help=f'Dataset a procesar. Disponibles: {", ".join(available_datasets)}'
    )
    parser.add_argument(
        '--noise_level',
        type=float,
        default=0.0,
        help='Fracción de etiquetas de TRAIN a corromper con ruido aleatorio uniforme. '
             'Ejemplos: 0.01 (1%%), 0.10 (10%%), 0.15 (15%%). Por defecto 0.0 (sin ruido).'
    )
    parser.add_argument(
        '--noise_mode',
        type=str,
        default='clean',
        choices=['clean', 'pre_clustering', 'train_only'],
        help='Modo de aplicación del ruido en etiquetas. En todos los modos, '
             'val mantiene SIEMPRE las etiquetas limpias. Modos: '
             '"clean" sin ruido (default); '
             '"pre_clustering" (Opción A: el K-Means usa etiquetas ruidosas, '
             'lo que afecta a la asignación de niveles; el ruido es visible al '
             'clasificador en train); '
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default=None,
        help='Sufijo opcional para el directorio de salida. Si se omite, se genera '
             'automáticamente a partir de noise_mode y noise_level (ej. _noiseA_10).'
    )
    args = parser.parse_args()

    # =========================================================================
    # VALIDACIÓN DE ARGUMENTOS DE RUIDO
    # =========================================================================
    if not (0.0 <= args.noise_level <= 1.0):
        raise ValueError(f"--noise_level debe estar en [0.0, 1.0], recibido {args.noise_level}")
    if args.noise_level > 0.0 and args.noise_mode == 'clean':
        # Coherencia: si hay ruido pero el modo es clean, error explícito
        raise ValueError(
            "Inconsistencia: --noise_level > 0 requiere --noise_mode pre_clustering "
            "o train_only. Si quieres ejecutar sin ruido, deja --noise_level 0.0."
        )
    if args.noise_level == 0.0 and args.noise_mode != 'clean':
        # Coherencia inversa: modo de ruido pero nivel 0 → forzar clean
        print(" noise_level=0.0 con noise_mode != clean: se forzará modo clean.")
        args.noise_mode = 'clean'

    # Sufijo del directorio de salida: ej. "_noiseA_10" o "_noiseB_15"
    if args.output_suffix is not None:
        output_suffix = args.output_suffix
    elif args.noise_mode == 'clean':
        output_suffix = ''
    else:
        mode_letter = 'A' if args.noise_mode == 'pre_clustering' else 'B'
        pct = int(round(args.noise_level * 100))
        output_suffix = f'_noise{mode_letter}_{pct:02d}'
    
    # Cargar configuración del dataset
    dataset_config = get_config(args.dataset)
    
    # ========================================================================
    # SEEDS PARA REPRODUCIBILIDAD COMPLETA
    # ========================================================================
    SEED = 42
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    
    # Crear Config desde dataset_config y hacerlo global
    Config = Config.from_dataset_config(dataset_config)

    # Aplicar sufijo de ruido al OUTPUT_PATH (si lo hay) sin tocar dataset_configs.py
    if output_suffix:
        original_output = Config.OUTPUT_PATH
        Config.OUTPUT_PATH = original_output + output_suffix
        print(f" OUTPUT_PATH redirigido por modo de ruido: {Config.OUTPUT_PATH}")

    print("="*80)
    print("SISTEMA DE CLUSTERING PARA CURRICULUM LEARNING")
    print("="*80)
    print(f"\n  Dataset: {dataset_config.name}")
    print(f"  Modo: {'RGB (3 canales)' if Config.IS_RGB else 'Grayscale (1 canal)'}")
    print(f"  Reproducibilidad: Seed fija = {SEED}")
    print(f"  Input: {Config.INPUT_PATH}")
    print(f"  Output: {Config.OUTPUT_PATH}")
    print(f"  Clusters por clase: {Config.N_CLUSTERS_PER_DIGIT}")
    print(f"  Clusters globales: {Config.N_CLUSTERS_GLOBAL}")
    print(f"  Niveles de dificultad: {Config.N_CLUSTERS_GLOBAL}")
    print(f"  Device: {Config.DEVICE}")
    if args.noise_mode == 'clean':
        print(f"  Ruido en etiquetas: NINGUNO (clean)")
    else:
        mode_label = ('Opción A: pre_clustering (afecta clustering Y entrenamiento)'
                      if args.noise_mode == 'pre_clustering'
                      else 'Opción B: train_only (clustering limpio, ruido solo en train)')
        print(f"   Ruido en etiquetas: {args.noise_level*100:.1f}% — {mode_label}")
    
    # ========================================================================
    # 1. Preparar datos
    # ========================================================================
    
    # NUEVO: Transform dinámico según RGB o Grayscale
    if Config.IS_RGB:
        print(f" Usando transform RGB ({Config.INPUT_CHANNELS} canales)")
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    else:
        print(f" Usando transform Grayscale ({Config.INPUT_CHANNELS} canal)")
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    # Detectar si es CIFAR-10 o Digits
    is_cifar10 = dataset_config.name in ['cifar10', 'cifar10rgb']
    
    # Variables para CIFAR-10
    cifar_images_global = None
    cifar_labels_global = None
    
    if is_cifar10:
        # ====================================================================
        # MODO CIFAR-10: Cargar desde pickles
        # ====================================================================
        
        if not Path(Config.INPUT_PATH).exists():
            print(f"\n  Error: No se encontró el directorio CIFAR-10 en {Config.INPUT_PATH}")
            print("    Descarga CIFAR-10 y extrae los archivos en ese directorio.")
            return
        
        # Cargar y seleccionar 5000 aleatorias
        cifar_images, cifar_labels = load_and_select_cifar10(
            Config.INPUT_PATH, 
            n_samples=dataset_config.n_samples,
            seed=42
        )
        
        # Guardar globalmente para organize_dataset
        cifar_images_global = cifar_images
        cifar_labels_global = cifar_labels
        
        # Crear dataset
        dataset = DigitDataset(
            Config.INPUT_PATH, 
            transform=transform,
            cifar_images=cifar_images,
            cifar_labels=cifar_labels
        )
        
    else:
        # ====================================================================
        # MODO DIGITS Cargar desde carpetas
        # ====================================================================
        
        if not Path(Config.INPUT_PATH).exists():
            print(f"\n  Error: No se encontró el dataset en {Config.INPUT_PATH}")
            print("    Verifica la ruta y vuelve a intentar.")
            return
        
        dataset = DigitDataset(Config.INPUT_PATH, transform=transform)
    
    if len(dataset) == 0:
        print("\n  Error: No se encontraron imágenes en el dataset")
        return
    
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=True, num_workers=4)
    
    # ========================================================================
    # 2. Entrenar Autoencoder
    # ========================================================================
    
    # NUEVO: Crear autoencoder con número de canales dinámico
    print(f"\n Creando autoencoder con {Config.INPUT_CHANNELS} canales de entrada...")
    model = ConvAutoencoder(
        latent_dim=Config.LATENT_DIM, 
        input_size=Config.IMAGE_SIZE,
        input_channels=Config.INPUT_CHANNELS
    )
    model, losses = train_autoencoder(model, dataloader, epochs=Config.EPOCHS)
    
    # Guardar modelo
    model_path = Path(Config.OUTPUT_PATH) / "autoencoder_model.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n  Modelo guardado en: {model_path}")
    
    # Guardar curva de loss
    print("  Generando gráfica de entrenamiento...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, color='#2E86AB')
    plt.title('Entrenamiento del Autoencoder - Loss por Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = Path(Config.OUTPUT_PATH) / "autoencoder_training_loss.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfica de loss guardada en: {loss_plot_path}")
    
    # ========================================================================
    # 3. Extraer embeddings
    # ========================================================================
    
    dataloader_eval = DataLoader(dataset, batch_size=Config.BATCH_SIZE, 
                                 shuffle=False, num_workers=4)
    embeddings, labels, paths = extract_embeddings(model, dataloader_eval)

    # ========================================================================
    # 3b. Aplicar ruido en etiquetas (si procede)
    # ========================================================================
    #
    # labels (devuelto por extract_embeddings) son las etiquetas LIMPIAS leídas
    # del dataset original. A partir de aquí preparamos:
    #   - labels_clean : copia inmutable de las etiquetas limpias
    #   - labels_noisy : versión ruidosa (idéntica a labels_clean si noise_level=0)
    #   - is_noisy     : máscara booleana de qué muestras fueron ruidificadas
    # Y, según noise_mode, decidimos qué array pasamos como `labels` al clustering:
    #   - clean  labels_clean (clustering opera con etiquetas limpias)
    #   - pre_clustering     labels_noisy (clustering opera con etiquetas ruidosas)
    # ------------------------------------------------------------------------

    labels_clean = np.asarray(labels).astype(int).copy()
    n_classes = int(np.unique(labels_clean).size)

    labels_noisy, is_noisy = apply_label_noise(
        labels_clean=labels_clean,
        noise_level=args.noise_level,
        n_classes=max(n_classes, 10),  # 10 clases asumidas en este pipeline
        seed=SEED
    )

    if args.noise_mode == 'pre_clustering':
        labels_for_clustering = labels_noisy
        print(f"\n    noise_mode=pre_clustering: K-Means usará etiquetas RUIDOSAS")
    else:
        labels_for_clustering = labels_clean
        if args.noise_mode == 'train_only':
            print(f"\n    noise_mode=train_only: K-Means usará etiquetas LIMPIAS, "
                  f"ruido se inyectará al guardar imágenes de train")
        else:
            print(f"\n    noise_mode=clean: K-Means con etiquetas limpias, sin ruido")

    # ========================================================================
    # 4. Realizar clustering
    # ========================================================================
    
    metadata, digit_clusters, kmeans_global = perform_clustering(
        embeddings, labels_for_clustering, paths,
        labels_clean=labels_clean,
        labels_noisy=labels_noisy,
        is_noisy=is_noisy,
        noise_mode=args.noise_mode
    )

    # Guardar también un noise_metadata.json compacto con la traza del ruido
    if args.noise_mode != 'clean':
        noise_meta_path = Path(Config.OUTPUT_PATH) / 'noise_metadata.json'
        noise_meta_path.parent.mkdir(parents=True, exist_ok=True)
        noise_summary = {
            'noise_level_requested': args.noise_level,
            'noise_level_actual': float(is_noisy.sum() / len(is_noisy)),
            'noise_mode': args.noise_mode,
            'n_total': int(len(is_noisy)),
            'n_noisy': int(is_noisy.sum()),
            'seed': SEED,
            'description': (
                'pre_clustering: ruido aplicado antes de K-Means. Afecta tanto '
                'a la asignación de niveles como a las etiquetas que ve el '
                'clasificador en train. Validación queda con etiquetas limpias.'
                if args.noise_mode == 'pre_clustering'
                else
                'train_only: K-Means usa etiquetas limpias (niveles correctos). '
                'El ruido se inyecta solo al guardar las imágenes de train en sus '
                'carpetas de clase. Validación queda con etiquetas limpias.'
            )
        }
        with open(noise_meta_path, 'w') as f:
            json.dump(noise_summary, f, indent=2)
        print(f"  Resumen de ruido guardado en: {noise_meta_path}")
    
    # Guardar metadata
    metadata_path = Path(Config.OUTPUT_PATH) / Config.METADATA_FILE
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata guardada en: {metadata_path}")
    
    # ========================================================================
    # 5. Organizar dataset
    # ========================================================================
    
    organize_dataset(metadata, cifar_images=cifar_images_global, cifar_labels=cifar_labels_global)
    
    # ========================================================================
    # 6. Visualizar resultados
    # ========================================================================
    
    viz_path = Path(Config.OUTPUT_PATH) / "visualizations"
    visualize_results(embeddings, labels, metadata, viz_path)
    
    # ========================================================================
    # Resumen final
    # ========================================================================
    
    print("\n" + "="*80)
    print(" PROCESO COMPLETADO CON ÉXITO")
    print("="*80)
    print(f"\n Archivos generados:")
    print(f"   • Dataset train: {Config.OUTPUT_PATH}/train/level_XX/")
    print(f"   • Dataset val: {Config.OUTPUT_PATH}/val/level_XX/")
    print(f"   • Metadata: {metadata_path}")
    print(f"   • Modelo: {model_path}")
    print(f"   • Visualizaciones: {viz_path}/")
    
    # Calcular totales
    train_count = sum(1 for item in metadata if item['split'] == 'train')
    val_count = sum(1 for item in metadata if item['split'] == 'val')
    
    print(f"\n Estadísticas:")
    print(f"   Total de imágenes: {len(metadata)}")
    print(f"   Train: {train_count} ({train_count/len(metadata)*100:.1f}%)")
    print(f"   Validation: {val_count} ({val_count/len(metadata)*100:.1f}%)")
    print(f"   Niveles de dificultad: {Config.N_CLUSTERS_GLOBAL} (level_00 a level_{Config.N_CLUSTERS_GLOBAL-1:02d})")
    print(f"   Dimensión latente: {Config.LATENT_DIM}")
    print("\n Próximo paso:")
    print("   Usa los datasets train/ y val/ para entrenar tu red neuronal con curriculum learning!")
    print("="*80)


if __name__ == "__main__":
    main()
