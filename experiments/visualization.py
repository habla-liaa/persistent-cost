"""
Funciones para visualización de diagramas de persistencia y reportes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def format_bars_for_output(dgm):
    """
    Formatea las barras de persistencia para salida de texto.
    
    Args:
        dgm: lista de arrays numpy, uno por dimensión
    
    Returns:
        str: texto formateado
    """
    output = []
    
    for dim, bars in enumerate(dgm):
        if isinstance(bars, list):
            bars = np.array(bars) if bars else np.empty((0, 2))
        
        if len(bars) == 0:
            output.append(f"  Dimensión {dim}: (ninguna)\n")
            continue
        
        output.append(f"  Dimensión {dim}: {len(bars)} barras\n")
        for i, (birth, death) in enumerate(bars):
            persistence = death - birth if not np.isinf(death) else np.inf
            death_str = '∞' if np.isinf(death) else f"{death:.4f}"
            output.append(f"    [{i}] ({birth:.4f}, {death_str}) - persistencia: {persistence}\n")
    
    return ''.join(output)


def classify_cone_bars(dgm_cone, dgm_ker, dgm_coker):
    """
    Clasifica las barras del cono como kernel, cokernel o desapareadas.
    
    Returns:
        str: clasificación formateada
    """
    output = []
    
    # Convertir a conjuntos de tuplas para comparación
    def bars_to_set(bars_list):
        result = []
        for dim, bars in enumerate(bars_list):
            if isinstance(bars, list):
                bars = np.array(bars) if bars else np.empty((0, 2))
            for bar in bars:
                result.append((dim, tuple(bar)))
        return set(result)
    
    cone_set = bars_to_set(dgm_cone)
    ker_set = bars_to_set(dgm_ker)
    coker_set = bars_to_set(dgm_coker)
    
    matched = ker_set | coker_set
    unmatched = cone_set - matched
    
    output.append(f"Total barras en cono: {len(cone_set)}\n")
    output.append(f"Clasificadas como kernel: {len(ker_set)}\n")
    output.append(f"Clasificadas como cokernel: {len(coker_set)}\n")
    output.append(f"Desapareadas: {len(unmatched)}\n\n")
    
    if unmatched:
        output.append("Barras desapareadas:\n")
        for dim, (birth, death) in sorted(unmatched):
            death_str = '∞' if np.isinf(death) else f"{death:.4f}"
            output.append(f"  Dim {dim}: ({birth:.4f}, {death_str})\n")
    
    return ''.join(output)


def plot_single_diagram(ax, dgm, title, maxdim=2):
    """
    Plotea un único diagrama de persistencia.
    
    Args:
        ax: matplotlib axis
        dgm: lista de arrays numpy con barras por dimensión
        title: título del gráfico
        maxdim: dimensión máxima a plotear
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Determinar rango para ejes
    max_val = 0
    for dim in range(min(len(dgm), maxdim + 1)):
        bars = dgm[dim]
        if isinstance(bars, list):
            bars = np.array(bars) if bars else np.empty((0, 2))
        if len(bars) > 0:
            finite_deaths = bars[~np.isinf(bars[:, 1]), 1]
            if len(finite_deaths) > 0:
                max_val = max(max_val, finite_deaths.max())
            max_val = max(max_val, bars[:, 0].max())
    
    if max_val == 0:
        max_val = 1
    
    # Plotear diagonal
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, linewidth=1)
    
    # Plotear barras por dimensión
    for dim in range(min(len(dgm), maxdim + 1)):
        bars = dgm[dim]
        if isinstance(bars, list):
            bars = np.array(bars) if bars else np.empty((0, 2))
        
        if len(bars) == 0:
            continue
        
        color = colors[dim % len(colors)]
        
        # Separar barras finitas e infinitas
        finite_mask = ~np.isinf(bars[:, 1])
        finite_bars = bars[finite_mask]
        infinite_bars = bars[~finite_mask]
        
        # Plotear barras finitas
        if len(finite_bars) > 0:
            ax.scatter(finite_bars[:, 0], finite_bars[:, 1], 
                      c=color, label=f'H{dim}', alpha=0.6, s=50)
        
        # Plotear barras infinitas en el borde superior
        if len(infinite_bars) > 0:
            y_inf = max_val * 1.05
            ax.scatter(infinite_bars[:, 0], [y_inf] * len(infinite_bars),
                      c=color, marker='^', s=100, alpha=0.8)
    
    ax.set_xlabel('Birth', fontsize=10)
    ax.set_ylabel('Death', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim([-max_val * 0.05, max_val * 1.1])
    ax.set_ylim([-max_val * 0.05, max_val * 1.15])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)


def plot_persistence_diagrams(results, output_dir):
    """
    Genera gráficos de diagramas de persistencia para un experimento.
    
    Args:
        results: dict con resultados del experimento
        output_dir: directorio donde guardar los gráficos
    """
    experiment_name = results['experiment_name']
    
    # Para cada método que tenga resultados exitosos
    for method in ['cone', 'cone2', 'cylinder']:
        if method not in results or 'error' in results[method]:
            continue
        
        method_data = results[method]
        
        # Crear figura con 6 subplots (X, Y, Cone, Ker, Coker, Cylinder)
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # X
        if 'dgm_X' in method_data:
            ax1 = fig.add_subplot(gs[0, 0])
            plot_single_diagram(ax1, method_data['dgm_X'], 'Espacio X')
        
        # Y
        if 'dgm_Y' in method_data:
            ax2 = fig.add_subplot(gs[0, 1])
            plot_single_diagram(ax2, method_data['dgm_Y'], 'Espacio Y')
        
        # Cone/Cylinder
        if 'dgm_cone' in method_data:
            ax3 = fig.add_subplot(gs[0, 2])
            plot_single_diagram(ax3, method_data['dgm_cone'], 'Cono')
        
        # Kernel
        if 'dgm_ker' in method_data:
            ax4 = fig.add_subplot(gs[1, 0])
            plot_single_diagram(ax4, method_data['dgm_ker'], 'Kernel')
        
        # Cokernel
        if 'dgm_coker' in method_data:
            ax5 = fig.add_subplot(gs[1, 1])
            plot_single_diagram(ax5, method_data['dgm_coker'], 'Cokernel')
        
        # Cylinder (si aplica) - por ahora vacío o referencia
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.text(0.5, 0.5, f'Método: {method}', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.axis('off')
        
        fig.suptitle(f'{experiment_name} - {method.upper()} (n={results["n"]})',
                    fontsize=14, fontweight='bold')
        
        # Guardar
        filename = f'{method}_diagrams.png'
        filepath = f'{output_dir}/{filename}'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Gráfico guardado: {filepath}")


def plot_barcode(dgm, title='Persistence Barcode', max_bars=50):
    """
    Plotea un código de barras de persistencia.
    
    Args:
        dgm: lista de arrays numpy con barras por dimensión
        title: título del gráfico
        max_bars: número máximo de barras a mostrar por dimensión
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    y_pos = 0
    
    for dim, bars in enumerate(dgm):
        if isinstance(bars, list):
            bars = np.array(bars) if bars else np.empty((0, 2))
        
        if len(bars) == 0:
            continue
        
        # Limitar número de barras mostradas
        if len(bars) > max_bars:
            bars = bars[:max_bars]
        
        color = colors[dim % len(colors)]
        
        for birth, death in bars:
            if np.isinf(death):
                death = birth + (bars[:, 0].max() - birth) * 1.5
            ax.plot([birth, death], [y_pos, y_pos], color=color, linewidth=2)
            y_pos += 1
        
        # Añadir separador entre dimensiones
        y_pos += 2
    
    ax.set_xlabel('Filtration Value')
    ax.set_ylabel('Bars')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    
    return fig, ax


def create_comparison_table(all_results, output_path='results/comparison_table.txt'):
    """
    Crea una tabla comparativa de todos los experimentos.
    
    Args:
        all_results: lista de dicts con resultados
        output_path: ruta donde guardar la tabla
    """
    with open(output_path, 'w') as f:
        f.write("TABLA COMPARATIVA DE EXPERIMENTOS\n")
        f.write("="*80 + "\n\n")
        
        header = f"{'Experimento':<20} {'n':<5} {'Lipschitz':<12} {'Método':<10} {'#Ker':<8} {'#Coker':<8}\n"
        f.write(header)
        f.write("-"*80 + "\n")
        
        for result in all_results:
            exp_name = result['experiment_name']
            n = result['n']
            L = result['lipschitz_constant']
            
            for method in ['cone', 'cone2', 'cylinder']:
                if method not in result or 'error' in result[method]:
                    continue
                
                method_data = result[method]
                n_ker = sum(len(b) for b in method_data.get('dgm_ker', []))
                n_coker = sum(len(b) for b in method_data.get('dgm_coker', []))
                
                line = f"{exp_name:<20} {n:<5} {L:<12.4f} {method:<10} {n_ker:<8} {n_coker:<8}\n"
                f.write(line)
            
            f.write("\n")
    
    print(f"Tabla comparativa guardada en: {output_path}")
