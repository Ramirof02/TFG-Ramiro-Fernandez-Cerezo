"""
Experimento 6: Overlap Aleatorio Completo
=========================================
Fase actual + 30% completamente aleatorio de fases anteriores.

Ejecutar: python exp6_solape_aleatorio_completo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import random
import os
import argparse
from dataset_configs import get_config

# ============================================================================
# PARSEAR ARGUMENTOS
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    type=str,
    default='digits',
    choices=['digits', 'cifar10', 'cifar10rgb'],
    help='Dataset a usar'
)
args = parser.parse_args()

# Cargar configuración del dataset
config = get_config(args.dataset)

# Fijar seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Detectar directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Asume que scripts están en TFG/Codigos/

# Rutas relativas al proyecto
TRAIN_DATASET = f"{config.output_path}/train"
VAL_DATASET = f"{config.output_path}/val"
RESULTS_DIR = f"./resultados/exp6_results_{config.name}"

BATCH_SIZE = config.batch_size
LEARNING_RATE = config.learning_rate
NUM_CLASSES = 10
IMAGE_SIZE = config.resize_to
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_PHASES = 5
EPOCHS_PER_PHASE = 10  #Variable modificable en funcion de la fase del proyecto
TOTAL_EPOCHS = N_PHASES * EPOCHS_PER_PHASE

# Niveles por fase
PHASE_LEVELS = [[0,1], [2,3], [4,5], [6,7], [8,9]]  # 10 niveles en 5 fases
# ============================================================================
# MODELO CNN
# ============================================================================

class DigitCNN(nn.Module):
    """CNN para clasificación - Optimizada para 128x128
    
    """
    
    def __init__(self, num_classes=10, input_size=128, input_channels=3):
        super(DigitCNN, self).__init__()
        
        # Calcular tamaño final después de 5 MaxPool2d (reduce por 2^5 = 32)
        final_size = input_size // 32
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * final_size * final_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# DATASET
# ============================================================================

class LevelBasedDataset(Dataset):
    """
    Dataset adaptable que carga imágenes desde train/ o val/ filtrado por niveles.
    
    Soporta dos estructuras:
    - Multi-clase (Digits): level_XX/0/, level_XX/1/, ... level_XX/9/
    """
    
    def __init__(self, root_dir, levels=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data = []
        
        # Si no se especifican niveles, cargar todos
        if levels is None:
            level_dirs = sorted([d for d in self.root_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('level_')])
            levels = [int(d.name.split('_')[1]) for d in level_dirs]
        
        for level in levels:
            level_dir = self.root_dir / f'level_{level:02d}'
            
            if not level_dir.exists():
                continue
            
            # Detectar si tiene subcarpetas por dígito o imágenes directas
            has_digit_subdirs = any((level_dir / str(i)).exists() for i in range(10))
            
            if has_digit_subdirs:
                # Estructura multi-clase: level_XX/digito/*.png
                for digit in range(10):
                    digit_dir = level_dir / str(digit)
                    
                    if not digit_dir.exists():
                        continue
                    
                    for img_path in digit_dir.glob('*.png'):
                        self.data.append({
                            'path': str(img_path),
                            'digit': digit,
                            'level': level
                        })
            else:
                # Estructura mono-clase: level_XX/*.png
                for img_path in level_dir.glob('*.png'):
                    self.data.append({
                        'path': str(img_path),
                        'digit': 0,  # Etiqueta dummy
                        'level': level
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # NUEVO: Cargar RGB o grayscale según dataset
        if config.name in ['cifar10rgb']:
            image = Image.open(item['path']).convert('RGB')
        else:
            image = Image.open(item['path']).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = item['digit']
        return image, label

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total


def save_plots(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()




# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print(f"EXPERIMENTO 6: Overlap Aleatorio Completo - {config.name}")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Results dir: {RESULTS_DIR}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # DESPUÉS (detecta automáticamente):
    if config.name in ['cifar10rgb']:
        # RGB: mantener colores
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    else:
        # Grayscale: convertir a gris (digits, cifar10)
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    print("Cargando dataset...")
    val_dataset = LevelBasedDataset(VAL_DATASET, levels=None, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"   Validación: {len(val_dataset)} imágenes")
    
    print("Creando modelo...")
    model = DigitCNN(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'phase_info': []
    }
    
    print("Estrategia: Fase actual + 30% aleatorio de TODAS las fases anteriores")
    print(f"Distribución: {PHASE_LEVELS}")
    print()
    
    start_time = time.time()
    
    for phase in range(N_PHASES):
        print(f"--- FASE {phase+1}/{N_PHASES} ---")
        
        current_levels = PHASE_LEVELS[phase]
        current_dataset = LevelBasedDataset(TRAIN_DATASET, levels=current_levels, transform=transform)
        
        if phase == 0:
            train_dataset = current_dataset
        else:
            # TODOS los niveles anteriores
            previous_levels = []
            for p in range(phase):
                previous_levels.extend(PHASE_LEVELS[p])
            
            prev_dataset = LevelBasedDataset(TRAIN_DATASET, levels=previous_levels, transform=transform)
            
            # Muestreo ALEATORIO 30%
            random_indices = random.sample(range(len(prev_dataset.data)), 
                                          int(len(prev_dataset.data) * 0.3))
            random_data = [prev_dataset.data[i] for i in random_indices]
            
            class RandomDataset(Dataset):
                def __init__(self, data, transform):
                    self.data = data
                    self.transform = transform
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    item = self.data[idx]
                    # NUEVO: Cargar RGB o grayscale según dataset
                    if config.name in ['cifar10rgb']:
                        image = Image.open(item['path']).convert('RGB')
                    else:
                        image = Image.open(item['path']).convert('L')
                    if self.transform:
                        image = self.transform(image)
                    return image, item['digit']
            
            random_dataset = RandomDataset(random_data, transform)
            train_dataset = ConcatDataset([current_dataset, random_dataset])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        print(f"Niveles actuales: {current_levels}")
        print(f"Entrenando con {len(train_dataset)} imágenes")
        
        history['phase_info'].append({
            'phase': phase + 1,
            'current_levels': current_levels,
            'n_samples': len(train_dataset)
        })
        
        phase_start = phase * EPOCHS_PER_PHASE + 1
        phase_end = (phase + 1) * EPOCHS_PER_PHASE
        
        for epoch in range(phase_start, phase_end + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if epoch % 5 == 0 or epoch == TOTAL_EPOCHS:
                print(f"Epoch {epoch}/{TOTAL_EPOCHS} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        print()
    
    training_time = time.time() - start_time
    
    print("Guardando resultados...")
    torch.save(model.state_dict(), Path(RESULTS_DIR) / 'model_final.pth')
    
    with open(Path(RESULTS_DIR) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    save_plots(history, RESULTS_DIR)
    
    results = {
        'experiment': 'exp6_solape_aleatorio_completo',
        'strategy': 'Overlap: fase actual + 30% aleatorio total',
        'phase_levels': PHASE_LEVELS,
        'total_epochs': TOTAL_EPOCHS,
        'training_time': training_time,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'device': str(DEVICE)
    }
    
    with open(Path(RESULTS_DIR) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {training_time/3600:.2f} horas")
    print(f"Val Accuracy final: {history['val_acc'][-1]:.2f}%")
    print(f"Mejor Val Accuracy: {max(history['val_acc']):.2f}%")
    print("="*70)



if __name__ == "__main__":
    main()
