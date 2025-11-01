import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import numpy as np


def diagram_stats(diagram):
    """
    Imprime estadísticas básicas del diagrama de persistencia.

    - diagram: dict
        Claves: dimensiones (int)
        Valores: listas de tuplas (nacimiento, muerte), donde muerte puede ser inf.
    """
    stats = []
    for dim, dgm in diagram.items():
        n_points = len(dgm)
        finite_points = [pt for pt in dgm if not np.isinf(pt[1])]
        n_finite = len(finite_points)
        n_infinite = n_points - n_finite

        if n_finite > 0:
            lifetimes = [pt[1] - pt[0] for pt in finite_points]
            avg_lifetime = np.mean(lifetimes)
            max_lifetime = np.max(lifetimes)
            avg_non_zero = np.mean([lt for lt in lifetimes if lt > 0]) if any(lt > 0 for lt in lifetimes) else 0.0
            ratio_non_zero = sum(1 for lt in lifetimes if lt > 0) / n_finite
        else:
            avg_lifetime = float('nan')
            max_lifetime = float('nan')
            avg_non_zero = float('nan')
            ratio_non_zero = float('nan')
        stats.append({
            "dimension": dim,
            "total_points": n_points,
            "finite_points": n_finite,
            "infinite_points": n_infinite,
            "ratio_finite_non_zero": ratio_non_zero,
            "average_lifetime_finite": avg_lifetime,
            "average_lifetime_non_zero": avg_non_zero,
            "max_lifetime_finite": max_lifetime
        })
    return stats


def print_diagram_stats(diagram):
    stats = diagram_stats(diagram)
    for stat in stats:
        print(f"Dimension {stat['dimension']}:")
        print(f"  Total points: {stat['total_points']}")
        print(f"  Finite points: {stat['finite_points']}")
        print(f"  Infinite points: {stat['infinite_points']}")
        print(
            f"  Average lifetime (finite): {stat['average_lifetime_finite']:.4f}")
        print(f"  Max lifetime (finite): {stat['max_lifetime_finite']:.4f}")
        print("")


def print_diagram(diagram):
    for dim, dgm in diagram.items():
        print(f"Dimension {dim}:")
        if len(dgm) == 0:
            print("  No points")
        else:
            for point in dgm:
                print(f"  {point}")


def plot_persistence_barcodes(barcodes_dict, tit="Barcodes de Homología Persistente por Dimensión"):
    """
    Grafica los barcodes de homología persistente a partir de un diccionario.

    - barcodes_dict: dict
        Claves: dimensiones (int)
        Valores: listas de tuplas (nacimiento, muerte), donde muerte puede ser inf.
    """
    all_dims = sorted(barcodes_dict.keys())
    max_dim = max(all_dims)

    # Recopilar todos los puntos finitos para estimar xlim
    finite_births = []
    finite_deaths = []
    for bars in barcodes_dict.values():
        for b, d in bars:
            finite_births.append(b)
            if not np.isinf(d):
                finite_deaths.append(d)

    min_birth = min(finite_births) if finite_births else 0.0
    max_death = max(finite_deaths) if finite_deaths else min_birth + 1.0
    plot_limit = max_death + 0.5  # margen extra

    colors = plt.cm.tab10
    y_pos = 0

    legend_patches = []

    for dim in range(max_dim + 1):
        bars = barcodes_dict.get(dim, [])
        dim_color = colors(dim % 10)

        # Agregar etiqueta a la leyenda (una sola vez por dimensión)
        legend_patches.append(mpatches.Patch(color=dim_color, label=f"H{dim}"))

        if not bars:
            y_pos += 1  # dejar espacio aunque no haya barras
        else:
            for birth, death in bars:
                if np.isinf(death):
                    death_plot = plot_limit
                else:
                    death_plot = death

                plt.hlines(y=y_pos, xmin=birth, xmax=death_plot,
                           colors=dim_color, linewidth=2)
                y_pos += 1

    plt.xlabel("Filtración")
    plt.yticks([])  # sin ticks en el eje y
    plt.title(tit)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.xlim(min_birth - 0.1, plot_limit)
    plt.ylim(-1, y_pos + 0.5)

    # Agregar la leyenda
    plt.legend(handles=legend_patches, title="Dimensión", loc='upper right')

    plt.tight_layout()
    plt.show()
