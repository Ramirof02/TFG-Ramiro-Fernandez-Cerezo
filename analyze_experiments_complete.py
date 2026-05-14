"""
Análisis Comparativo Completo de Experimentos de Curriculum Learning
====================================================================

Versión consolidada que incluye:
- AUC (Area Under the Curve)
- Métricas tradicionales
- Análisis por grupos
- Rankings múltiples
- Reportes detallados
- Gráficas comparativas

Uso:
    python analyze_experiments_complete.py --dataset digits
    python analyze_experiments_complete.py --dataset cifar10
    python analyze_experiments_complete.py --dataset cifar10rgb

Soporte para experimentos con ruido (variable de entorno NOISE_SUFFIX):
    export NOISE_SUFFIX=_noiseA_01
    python analyze_experiments_complete.py --dataset cifar10rgb
    # → analiza ./resultados/expN_results_cifar10rgb_noiseA_01/
    # → guarda en ./analisis/analysis_results_cifar10rgb_noiseA_01/

    Si NOISE_SUFFIX no está definida o está vacía, se comporta como antes
    (analiza las carpetas sin sufijo).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI - debe estar ANTES de import pyplot
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
from scipy import integrate
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    type=str,
    default='digits',
    choices=['digits', 'cifar10','cifar10rgb'],
    help='Dataset analizado'
)
args = parser.parse_args()

DATASET = args.dataset

# Leer la variable de entorno NOISE_SUFFIX (mismo mecanismo que dataset_configs.py).
# Si está definida (p.ej. "_noiseA_01"), se aplica al nombre del dataset
# para construir las rutas de los experimentos y del directorio de salida.
NOISE_SUFFIX = os.environ.get('NOISE_SUFFIX', '').strip()
DATASET_TAG = DATASET + NOISE_SUFFIX  # ej. "cifar10rgb" o "cifar10rgb_noiseA_01"

if NOISE_SUFFIX:
    print(f" NOISE_SUFFIX activo: {NOISE_SUFFIX!r}")
    print(f"    Buscando experimentos en: ./resultados/expN_results_{DATASET_TAG}/")
    print(f"    Guardando análisis en:    ./analisis/analysis_results_{DATASET_TAG}/")
else:
    print(f" Sin sufijo de ruido. Analizando experimentos limpios.")


def _dataset_label():
    """
    Devuelve una etiqueta descriptiva del dataset analizado para usar
    en cabeceras de reportes y prints. Decodifica el sufijo NOISE_SUFFIX
    en una descripción legible.
        cifar10rgb            -> 'CIFAR10RGB'
        cifar10rgb_noiseA_01  -> 'CIFAR10RGB (ruido modo A, 1%)'
        cifar10rgb_noiseB_15  -> 'CIFAR10RGB (ruido modo B, 15%)'
    """
    base = DATASET.upper()
    if not NOISE_SUFFIX:
        return base
    # Patrón esperado: _noise{A|B}_{NN}
    s = NOISE_SUFFIX.lstrip('_')
    if s.startswith('noise') and len(s) >= 8:
        try:
            mode = s[5]                  # 'A' o 'B'
            pct = int(s.split('_')[1])   # número
            mode_desc = ('pre-clustering' if mode == 'A'
                         else 'train-only' if mode == 'B'
                         else mode)
            return f"{base} (ruido modo {mode}: {mode_desc}, {pct}%)"
        except (IndexError, ValueError):
            pass
    # Si no encaja con el patrón esperado, devolver el sufijo crudo
    return f"{base} ({NOISE_SUFFIX})"

DATASET_LABEL = _dataset_label()

# Configuración de experimentos
EXPERIMENTS = {
    'exp1': {
        'name': 'Tradicional (Baseline)',
        'short_name': 'Baseline',
        'dir': f'./resultados/exp1_results_{DATASET_TAG}',
        'group': 'baseline'
    },
    'exp2': {
        'name': 'Etapas Temporales',
        'short_name': 'Temporal',
        'dir': f'./resultados/exp2_results_{DATASET_TAG}',
        'group': 'baseline'
    },
    'exp3': {
        'name': 'Por Dificultad',
        'short_name': 'Por Dificultad',
        'dir': f'./resultados/exp3_results_{DATASET_TAG}',
        'group': 'completo'
    },
    'exp4': {
        'name': 'Overlap Simple',
        'short_name': 'Overlap Full',
        'dir': f'./resultados/exp4_results_{DATASET_TAG}',
        'group': 'completo'
    },
    'exp5': {
        'name': 'Hard Mining',
        'short_name': 'Hard Mining',
        'dir': f'./resultados/exp5_results_{DATASET_TAG}',
        'group': 'completo'
    },
    'exp6': {
        'name': 'Aleatorio Completo',
        'short_name': 'Rand Total',
        'dir': f'./resultados/exp6_results_{DATASET_TAG}',
        'group': 'completo'
    },
    'exp7': {
        'name': 'Intermedios Progresivo',
        'short_name': 'Inter Prog',
        'dir': f'./resultados/exp7_results_{DATASET_TAG}',
        'group': 'intermedios'
    },
    'exp8': {
        'name': 'Intermedios Acumulativo',
        'short_name': 'Inter Acum',
        'dir': f'./resultados/exp8_results_{DATASET_TAG}',
        'group': 'intermedios'
    },
    'exp9': {
        'name': 'Intermedios + Hard Mining',
        'short_name': 'Inter+Hard',
        'dir': f'./resultados/exp9_results_{DATASET_TAG}',
        'group': 'intermedios'
    },
    'exp10': {
        'name': 'Intermedios + Aleatorio',
        'short_name': 'Inter+Rand',
        'dir': f'./resultados/exp10_results_{DATASET_TAG}',
        'group': 'intermedios'
    },
}

OUTPUT_DIR = f'./analisis/analysis_results_{DATASET_TAG}'
Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================================
# FUNCIONES DE CÁLCULO DE MÉTRICAS
# ============================================================================

def calculate_auc(accuracy_values):
    """
    Calcula el AUC (Area Under the Curve) normalizado.
    
    Args:
        accuracy_values: Lista de valores de accuracy por epoch
    
    Returns:
        float: AUC normalizado (0-100)
    """
    epochs = np.arange(len(accuracy_values))
    auc = integrate.trapezoid(accuracy_values, epochs)
    auc_normalized = auc / len(accuracy_values)
    return auc_normalized


def calculate_convergence_epoch(accuracy_values, threshold=0.95):
    """
    Calcula en qué epoch el modelo alcanza el threshold del mejor accuracy.
    
    Args:
        accuracy_values: Lista de accuracy por epoch
        threshold: Porcentaje del mejor accuracy (default 0.95 = 95%)
    
    Returns:
        int: Epoch de convergencia (None si no converge)
    """
    best_acc = max(accuracy_values)
    target = best_acc * threshold
    
    for epoch, acc in enumerate(accuracy_values, 1):
        if acc >= target:
            return epoch
    
    return None


def calculate_stability(accuracy_values, window=5):
    """
    Calcula la estabilidad del entrenamiento.
    
    Args:
        accuracy_values: Lista de accuracy
        window: Ventana de últimos epochs a considerar
    
    Returns:
        float: Desviación estándar (menor = más estable)
    """
    if len(accuracy_values) < window:
        return np.std(accuracy_values)
    
    last_values = accuracy_values[-window:]
    return np.std(last_values)


def calculate_improvement(accuracy_values):
    """Calcula la mejora total desde el primer epoch"""
    return accuracy_values[-1] - accuracy_values[0]


# ============================================================================
# FUNCIONES DE CARGA Y ANÁLISIS
# ============================================================================

def load_experiment_data(exp_dir):
    """
    Carga los datos de un experimento.
    
    Args:
        exp_dir: Directorio del experimento
    
    Returns:
        dict: Datos del experimento (None si no existe)
    """
    history_file = Path(exp_dir) / 'history.json'
    results_file = Path(exp_dir) / 'results.json'
    
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    results = None
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    return {
        'history': history,
        'results': results
    }


def analyze_experiment(exp_id, exp_info):
    """
    Analiza un experimento y calcula todas las métricas.
    
    Args:
        exp_id: ID del experimento
        exp_info: Info del experimento
    
    Returns:
        dict: Métricas calculadas
    """
    data = load_experiment_data(exp_info['dir'])
    
    if data is None:
        return None
    
    history = data['history']
    results = data['results']
    
    val_acc = history['val_acc']
    train_acc = history['train_acc']
    
    metrics = {
        'experiment': exp_id,
        'name': exp_info['name'],
        'short_name': exp_info['short_name'],
        'group': exp_info['group'],
        
        # AUC
        'auc_val': calculate_auc(val_acc),
        'auc_train': calculate_auc(train_acc),
        
        # Accuracy
        'final_val_acc': val_acc[-1],
        'best_val_acc': max(val_acc),
        'final_train_acc': train_acc[-1],
        
        # Generalization gap
        'gen_gap': train_acc[-1] - val_acc[-1],
        
        # Convergencia
        'convergence_epoch': calculate_convergence_epoch(val_acc),
        
        # Estabilidad
        'stability': calculate_stability(val_acc),
        
        # Mejora
        'improvement': calculate_improvement(val_acc),
        
        # Tiempo
        'training_time': results.get('training_time', None) if results else None,
        
        # Datos completos para gráficas
        'val_acc_curve': val_acc,
        'train_acc_curve': train_acc,
        'val_loss_curve': history.get('val_loss', []),
        'train_loss_curve': history.get('train_loss', []),
    }
    
    return metrics


# ============================================================================
# CREACIÓN DE TABLAS
# ============================================================================

def create_comparison_table(all_metrics):
    """Crea tabla comparativa completa"""
    df_data = []
    for exp_id, metrics in all_metrics.items():
        df_data.append({
            'Exp': exp_id.upper(),
            'Nombre': metrics['short_name'],
            'Grupo': metrics['group'],
            'AUC Val': f"{metrics['auc_val']:.2f}",
            'Best Acc': f"{metrics['best_val_acc']:.2f}%",
            'Final Acc': f"{metrics['final_val_acc']:.2f}%",
            'Gap Train-Val': f"{metrics['gen_gap']:.2f}%",
            'Convergencia': metrics['convergence_epoch'] if metrics['convergence_epoch'] else 'N/A',
            'Estabilidad': f"{metrics['stability']:.3f}",
            'Mejora': f"{metrics['improvement']:.2f}%",
            'Tiempo (h)': f"{metrics['training_time']/3600:.2f}" if metrics['training_time'] else 'N/A'
        })
    
    df = pd.DataFrame(df_data)
    return df


# ============================================================================
# RANKINGS
# ============================================================================

def create_rankings(all_metrics):
    """Crea rankings por diferentes criterios"""
    rankings = {
        'auc': sorted(all_metrics.items(), key=lambda x: x[1]['auc_val'], reverse=True),
        'best_acc': sorted(all_metrics.items(), key=lambda x: x[1]['best_val_acc'], reverse=True),
        'final_acc': sorted(all_metrics.items(), key=lambda x: x[1]['final_val_acc'], reverse=True),
        'convergence': sorted(
            [(k, v) for k, v in all_metrics.items() if v['convergence_epoch'] is not None],
            key=lambda x: x[1]['convergence_epoch']
        ),
        'stability': sorted(all_metrics.items(), key=lambda x: x[1]['stability']),
        'generalization': sorted(all_metrics.items(), key=lambda x: abs(x[1]['gen_gap'])),
        'speed': sorted(
            [(k, v) for k, v in all_metrics.items() if v['training_time'] is not None],
            key=lambda x: x[1]['training_time']
        ),
    }
    
    return rankings


# ============================================================================
# GRÁFICAS
# ============================================================================

def plot_convergence_curves(all_metrics, output_dir):
    """Gráfico de curvas de convergencia por grupo"""
    groups = {
        'baseline': 'Baseline (Sin Curriculum)',
        'completo': 'Curriculum Completo (Niveles 0-9)',
        'intermedios': 'Curriculum Intermedios (Niveles 3-7)'
    }
    
    for group_id, group_name in groups.items():
        group_metrics = {k: v for k, v in all_metrics.items() if v['group'] == group_id}
        
        if not group_metrics:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Curvas de validación
        for exp_id, metrics in group_metrics.items():
            epochs = range(1, len(metrics['val_acc_curve']) + 1)
            ax1.plot(epochs, metrics['val_acc_curve'], 
                    label=f"{exp_id.upper()}: {metrics['short_name']}", linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax1.set_title(f'{group_name} - Validación', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Curvas de entrenamiento
        for exp_id, metrics in group_metrics.items():
            epochs = range(1, len(metrics['train_acc_curve']) + 1)
            ax2.plot(epochs, metrics['train_acc_curve'],
                    label=f"{exp_id.upper()}: {metrics['short_name']}", linewidth=2)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
        ax2.set_title(f'{group_name} - Entrenamiento', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(output_dir) / f'convergence_{group_id}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  {plot_path.name}")


def plot_auc_comparison(all_metrics, output_dir):
    """Comparación AUC"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    exp_names = [m['short_name'] for m in all_metrics.values()]
    auc_values = [m['auc_val'] for m in all_metrics.values()]
    colors = ['red' if m['group'] == 'baseline' else 
              'blue' if m['group'] == 'completo' else 
              'green' for m in all_metrics.values()]
    
    bars = ax.bar(range(len(exp_names)), auc_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels([k.upper() for k in all_metrics.keys()], rotation=0)
    ax.set_ylabel('AUC (Area Under Curve)', fontsize=12)
    ax.set_title('Comparación de AUC por Experimento', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for bar, val in zip(bars, auc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Baseline'),
        Patch(facecolor='blue', alpha=0.7, label='Completo (0-9)'),
        Patch(facecolor='green', alpha=0.7, label='Intermedios (3-7)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    auc_plot = Path(output_dir) / 'auc_comparison.png'
    plt.savefig(auc_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   {auc_plot.name}")


def plot_metrics_comparison(all_metrics, output_dir):
    """Gráficos de barras comparativos de múltiples métricas"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    exp_names = [k.upper() for k in all_metrics.keys()]
    colors = ['red' if m['group'] == 'baseline' else 
              'blue' if m['group'] == 'completo' else 
              'green' for m in all_metrics.values()]
    
    # Mejor Val Acc
    ax1 = axes[0, 0]
    best_accs = [m['best_val_acc'] for m in all_metrics.values()]
    bars = ax1.bar(exp_names, best_accs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mejor Val Accuracy (%)', fontsize=11)
    ax1.set_title('Mejor Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, best_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # AUC
    ax2 = axes[0, 1]
    aucs = [m['auc_val'] for m in all_metrics.values()]
    bars = ax2.bar(exp_names, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('AUC', fontsize=11)
    ax2.set_title('Area Under Curve (AUC)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Tiempo
    ax3 = axes[1, 0]
    times = [m['training_time']/3600 if m['training_time'] else 0 for m in all_metrics.values()]
    bars = ax3.bar(exp_names, times, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Tiempo de Entrenamiento (h)', fontsize=11)
    ax3.set_title('Tiempo de Entrenamiento', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, times):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Estabilidad
    ax4 = axes[1, 1]
    stabilities = [m['stability'] for m in all_metrics.values()]
    bars = ax4.bar(exp_names, stabilities, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Estabilidad (menor = mejor)', fontsize=11)
    ax4.set_title('Estabilidad del Entrenamiento', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, stabilities):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    metrics_plot = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(metrics_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {metrics_plot.name}")


# ============================================================================
# REPORTE TEXTUAL
# ============================================================================

def save_detailed_report(all_metrics, rankings, df, output_dir):
    """Guarda reporte detallado en texto"""
    report_file = Path(output_dir) / 'detailed_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANÁLISIS COMPARATIVO COMPLETO - CURRICULUM LEARNING\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {DATASET_LABEL}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experimentos analizados: {len(all_metrics)}\n\n")
        
        # Tabla comparativa
        f.write("="*80 + "\n")
        f.write("TABLA COMPARATIVA COMPLETA\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Rankings
        f.write("="*80 + "\n")
        f.write("RANKINGS POR CRITERIO\n")
        f.write("="*80 + "\n\n")
        
        f.write(" TOP 3 - AUC (Area Under Curve):\n")
        f.write("-"*80 + "\n")
        for i, (exp_id, metrics) in enumerate(rankings['auc'][:3], 1):
            f.write(f"  {i}. {exp_id.upper()}: {metrics['name']}\n")
            f.write(f"     AUC: {metrics['auc_val']:.2f} | Best Acc: {metrics['best_val_acc']:.2f}%\n")
        f.write("\n")
        
        f.write(" TOP 3 - Mejor Accuracy:\n")
        f.write("-"*80 + "\n")
        for i, (exp_id, metrics) in enumerate(rankings['best_acc'][:3], 1):
            f.write(f"  {i}. {exp_id.upper()}: {metrics['name']}\n")
            f.write(f"     Accuracy: {metrics['best_val_acc']:.2f}%\n")
        f.write("\n")
        
        f.write("⚡ TOP 3 - Convergencia Más Rápida:\n")
        f.write("-"*80 + "\n")
        for i, (exp_id, metrics) in enumerate(rankings['convergence'][:3], 1):
            f.write(f"  {i}. {exp_id.upper()}: {metrics['name']}\n")
            f.write(f"     Epoch: {metrics['convergence_epoch']}\n")
        f.write("\n")
        
        f.write(" TOP 3 - Más Estable:\n")
        f.write("-"*80 + "\n")
        for i, (exp_id, metrics) in enumerate(rankings['stability'][:3], 1):
            f.write(f"  {i}. {exp_id.upper()}: {metrics['name']}\n")
            f.write(f"     Estabilidad: {metrics['stability']:.3f}\n")
        f.write("\n")
        
        f.write(" TOP 3 - Mejor Generalización:\n")
        f.write("-"*80 + "\n")
        for i, (exp_id, metrics) in enumerate(rankings['generalization'][:3], 1):
            f.write(f"  {i}. {exp_id.upper()}: {metrics['name']}\n")
            f.write(f"     Gap Train-Val: {metrics['gen_gap']:.2f}%\n")
        f.write("\n")
        
        # Análisis por grupos
        f.write("="*80 + "\n")
        f.write("COMPARACIÓN POR GRUPOS\n")
        f.write("="*80 + "\n\n")
        
        groups = {
            'baseline': 'Baseline (Sin Curriculum)',
            'completo': 'Curriculum Completo (0-9)',
            'intermedios': 'Curriculum Intermedios (3-7)'
        }
        
        for group_id, group_name in groups.items():
            group_exps = [v for v in all_metrics.values() if v['group'] == group_id]
            if group_exps:
                f.write(f"{group_name}:\n")
                f.write("-"*80 + "\n")
                avg_auc = np.mean([e['auc_val'] for e in group_exps])
                avg_acc = np.mean([e['best_val_acc'] for e in group_exps])
                f.write(f"  AUC promedio: {avg_auc:.2f}\n")
                f.write(f"  Accuracy promedio: {avg_acc:.2f}%\n")
                f.write(f"  Número de experimentos: {len(group_exps)}\n\n")
        
        # Podio final
        f.write("="*80 + "\n")
        f.write(" PODIO FINAL - TOP 3\n")
        f.write("="*80 + "\n\n")
        
        top3_auc = rankings['auc'][:3]
        
        if len(top3_auc) > 0:
            gold = top3_auc[0]
            f.write(" PRIMER LUGAR (Por AUC)\n")
            f.write("-"*80 + "\n")
            f.write(f"   {gold[0].upper()}: {gold[1]['name']}\n")
            f.write(f"   AUC: {gold[1]['auc_val']:.2f}\n")
            f.write(f"   Best Accuracy: {gold[1]['best_val_acc']:.2f}%\n")
            f.write(f"   Convergencia: Epoch {gold[1]['convergence_epoch']}\n")
            f.write(f"   Estabilidad: {gold[1]['stability']:.3f}\n\n")
        
        if len(top3_auc) > 1:
            silver = top3_auc[1]
            f.write(" SEGUNDO LUGAR\n")
            f.write("-"*80 + "\n")
            f.write(f"   {silver[0].upper()}: {silver[1]['name']}\n")
            f.write(f"   AUC: {silver[1]['auc_val']:.2f}\n")
            f.write(f"   Best Accuracy: {silver[1]['best_val_acc']:.2f}%\n\n")
        
        if len(top3_auc) > 2:
            bronze = top3_auc[2]
            f.write(" TERCER LUGAR\n")
            f.write("-"*80 + "\n")
            f.write(f"   {bronze[0].upper()}: {bronze[1]['name']}\n")
            f.write(f"   AUC: {bronze[1]['auc_val']:.2f}\n")
            f.write(f"   Best Accuracy: {bronze[1]['best_val_acc']:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ {report_file.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print(f"ANÁLISIS COMPARATIVO COMPLETO - {DATASET_LABEL}")
    print("="*70)
    
    # Analizar todos los experimentos
    all_metrics = {}
    
    for exp_id, exp_info in EXPERIMENTS.items():
        print(f"\nAnalizando {exp_id}: {exp_info['name']}...")
        metrics = analyze_experiment(exp_id, exp_info)
        
        if metrics is None:
            print(f"    Resultados no encontrados en {exp_info['dir']}")
        else:
            all_metrics[exp_id] = metrics
            print(f"   AUC: {metrics['auc_val']:.2f}")
            print(f"   Best Val Acc: {metrics['best_val_acc']:.2f}%")
    
    if not all_metrics:
        print("\n No se encontraron resultados. Ejecuta los experimentos primero.")
        return
    
    # Crear tabla comparativa
    print("\n" + "="*70)
    print("CREANDO TABLA COMPARATIVA")
    print("="*70)
    
    df = create_comparison_table(all_metrics)
    print("\n" + df.to_string(index=False))
    
    # Guardar CSV
    csv_path = Path(OUTPUT_DIR) / 'comparative_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Tabla guardada: {csv_path}")
    
    # Crear rankings
    print("\n" + "="*70)
    print("CREANDO RANKINGS")
    print("="*70)
    
    rankings = create_rankings(all_metrics)
    
    print("\n TOP 3 - AUC:")
    for i, (exp_id, metrics) in enumerate(rankings['auc'][:3], 1):
        print(f"  {i}. {exp_id.upper()}: {metrics['name']}")
        print(f"     AUC: {metrics['auc_val']:.2f} | Best Acc: {metrics['best_val_acc']:.2f}%")
    
    # Generar gráficas
    print("\n" + "="*70)
    print("GENERANDO GRÁFICAS")
    print("="*70)
    
    plot_convergence_curves(all_metrics, OUTPUT_DIR)
    plot_auc_comparison(all_metrics, OUTPUT_DIR)
    plot_metrics_comparison(all_metrics, OUTPUT_DIR)
    
    # Guardar reporte detallado
    print("\n" + "="*70)
    print("GENERANDO REPORTE DETALLADO")
    print("="*70)
    
    save_detailed_report(all_metrics, rankings, df, OUTPUT_DIR)
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE INSIGHTS")
    print("="*70)
    
    # Mejor por AUC
    best_auc = rankings['auc'][0]
    print(f"\n Mejor AUC: {best_auc[0].upper()} - {best_auc[1]['name']}")
    print(f"   AUC: {best_auc[1]['auc_val']:.2f}")
    
    # Mejor accuracy
    best_acc_exp = rankings['best_acc'][0]
    print(f"\n Mejor Accuracy: {best_acc_exp[0].upper()} - {best_acc_exp[1]['name']}")
    print(f"   Accuracy: {best_acc_exp[1]['best_val_acc']:.2f}%")
    
    # Más estable
    most_stable = rankings['stability'][0]
    print(f"\n Más Estable: {most_stable[0].upper()} - {most_stable[1]['name']}")
    print(f"   Estabilidad: {most_stable[1]['stability']:.3f}")
    
    # Convergencia más rápida
    if rankings['convergence']:
        fastest = rankings['convergence'][0]
        print(f"\n⚡ Convergencia Más Rápida: {fastest[0].upper()} - {fastest[1]['name']}")
        print(f"   Epoch: {fastest[1]['convergence_epoch']}")
    
    # Comparación grupos
    groups = {
        'baseline': 'Baseline',
        'completo': 'Completo (0-9)',
        'intermedios': 'Intermedios (3-7)'
    }
    
    print(f"\n Comparación por Grupo (AUC promedio):")
    for group_id, group_name in groups.items():
        group_exps = [v for v in all_metrics.values() if v['group'] == group_id]
        if group_exps:
            avg_auc = np.mean([e['auc_val'] for e in group_exps])
            print(f"   {group_name}: {avg_auc:.2f}")
    
    print("\n" + "="*70)
    print(f" Análisis completado. Resultados en: {OUTPUT_DIR}")
    print("="*70)
    
    print("\nArchivos generados:")
    print("   comparative_metrics.csv")
    print("   convergence_baseline.png")
    print("   convergence_completo.png")
    print("   convergence_intermedios.png")
    print("   auc_comparison.png")
    print("   metrics_comparison.png")
    print("   detailed_report.txt")


if __name__ == "__main__":
    main()
