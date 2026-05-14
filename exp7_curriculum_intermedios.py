"""
Experimento 7: Curriculum Intermedios Progresivo
================================================
Curriculum learning con SOLO niveles intermedios (3-7).
Progresión nivel a nivel sin overlap.

Descarta niveles extremos:
- Niveles 0-2 (muy fáciles)
- Niveles 8-9 (muy difíciles)

Estructura:
  Fase 1: Solo nivel [3]
  Fase 2: Solo nivel [4]
  Fase 3: Solo nivel [5]
  Fase 4: Solo nivel [6]
  Fase 5: Solo nivel [7]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import argparse
from dataset_configs import get_config
import json
from pathlib import Path
import time

# ============================================================================
# PARSEAR ARGUMENTOS PRIMERO
# ============================================================================
parser = argparse.ArgumentParser(description='Experimento 7: Curriculum Intermedios Progresivo')
parser.add_argument('--dataset', type=str, required=True,
                    choices=['digits', 'cifar10', 'cifar10rgb'],
                    help='Dataset a utilizar')
args = parser.parse_args()

# Obtener configuración del dataset
config = get_config(args.dataset)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
TRAIN_DATASET = f"{config.output_path}/train"
VAL_DATASET = f"{config.output_path}/val"
IMAGE_SIZE = config.resize_to

N_PHASES = 5
EPOCHS_PER_PHASE = 10 #Variable modificable en funcion de la fase del proyecto
TOTAL_EPOCHS = N_PHASES * EPOCHS_PER_PHASE  # 25

BATCH_SIZE = config.batch_size
LEARNING_RATE = config.learning_rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONFIGURACIÓN DEL EXPERIMENTO: Curriculum progresivo solo niveles intermedios
PHASE_LEVELS = [
    [3],      # Fase 1: Solo nivel 3 (más fácil de intermedios)
    [4],      # Fase 2: Solo nivel 4
    [5],      # Fase 3: Solo nivel 5 (medio)
    [6],      # Fase 4: Solo nivel 6
    [7],      # Fase 5: Solo nivel 7 (más difícil de intermedios)
]

OUTPUT_DIR = f"./resultados/exp7_results_{config.name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Transform para convertir grayscale a RGB (3 canales)
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

# ============================================================================
# DATASET PERSONALIZADO
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
# MODELO CNN
# ============================================================================

class DigitCNN(nn.Module):
    """CNN para clasificación - Optimizada para 128x128
    
    Arquitectura unificada (5 capas conv + 4 FC, ~5.93M parámetros):
    32 -> 64 -> 128 -> 256 -> 512 channels.
    Idéntica al resto de experimentos para garantizar comparabilidad.
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
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por un epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Valida el modelo"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# ============================================================================
# ENTRENAMIENTO PRINCIPAL
# ============================================================================

def main():
    print("="*80)
    print(f"EXPERIMENTO 7: Curriculum Intermedios Progresivo - {config.name}")
    print("="*80)
    print(f"Dataset train: {TRAIN_DATASET}")
    print(f"Dataset val: {VAL_DATASET}")
    print(f"Niveles usados: 3, 4, 5, 6, 7 (SOLO intermedios)")
    print(f"Niveles descartados: 0, 1, 2 (muy fáciles) y 8, 9 (muy difíciles)")
    print(f"Estrategia: Curriculum progresivo sin overlap")
    print(f"Fases: {N_PHASES}")
    print(f"Epochs por fase: {EPOCHS_PER_PHASE}")
    print(f"Total epochs: {TOTAL_EPOCHS}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # Iniciar timer
    start_time = time.time()
    
    # Modelo, loss, optimizer
    model = DigitCNN(num_classes=10, input_size=IMAGE_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Dataset de validación (COMPLETO - niveles 0-9)
    val_dataset = LevelBasedDataset(
        VAL_DATASET,
        levels=list(range(10)),  # Validación con TODOS los niveles
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n IMPORTANTE: Validación con TODOS los niveles (0-9)")
    print(f"Validación: {len(val_dataset)} imágenes")
    print(f"Se espera baja accuracy en niveles no entrenados (0-2, 8-9)\n")
    
    # Historial
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'phase_info': []
    }
    
    # ========================================================================
    # ENTRENAMIENTO POR FASES
    # ========================================================================
    
    for phase in range(N_PHASES):
        levels = PHASE_LEVELS[phase]
        
        print(f"\n{'='*80}")
        print(f"FASE {phase + 1}/{N_PHASES}")
        print(f"Niveles: {levels}")
        print('='*80)
        
        # Cargar dataset de la fase
        train_dataset = LevelBasedDataset(
            TRAIN_DATASET,
            levels=levels,
            transform=transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        print(f"Train: {len(train_dataset)} imágenes")
        
        # Guardar info de la fase
        history['phase_info'].append({
            'phase': phase + 1,
            'levels': levels,
            'num_samples': len(train_dataset)
        })
        
        # Epochs de esta fase
        phase_start = phase * EPOCHS_PER_PHASE + 1
        phase_end = (phase + 1) * EPOCHS_PER_PHASE
        
        for epoch in range(phase_start, phase_end + 1):
            # Entrenar
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            
            # Validar
            val_loss, val_acc = validate(
                model, val_loader, criterion, DEVICE
            )
            
            # Guardar métricas
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}/{TOTAL_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    
    # Calcular tiempo total
    training_time = time.time() - start_time
    
    # Guardar modelo
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_best.pth")
    
    # Guardar historial
    with open(f"{OUTPUT_DIR}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Guardar resultados finales
    results = {
        'experiment': 'exp7_curriculum_intermedios',
        'dataset': config.name,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'training_time': training_time,
        'total_epochs': TOTAL_EPOCHS,
        'phases': N_PHASES,
        'epochs_per_phase': EPOCHS_PER_PHASE,
        'levels_used': [3, 4, 5, 6, 7],
        'strategy': 'curriculum_intermedios_progresivo'
    }
    
    with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Accuracy final en validación: {history['val_acc'][-1]:.2f}%")
    print(f"Mejor accuracy en validación: {max(history['val_acc']):.2f}%")
    print(f"Tiempo de entrenamiento: {training_time/3600:.2f} horas")
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print("\n NOTA: La validación incluye niveles no entrenados (0-2, 8-9)")
    print("   Se espera baja accuracy en esos niveles.")
    print("="*80)

if __name__ == "__main__":
    main()
