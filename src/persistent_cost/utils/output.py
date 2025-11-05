import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import numpy as np

def array2dict(diagram):
    if not isinstance(diagram, dict):
        return {i: dgm for i, dgm in enumerate(diagram)}
    return diagram

def diagram_stats(diagram):
    """
    Imprime estadísticas básicas del diagrama de persistencia.

    - diagram: dict
        Claves: dimensiones (int)
        Valores: listas de tuplas (nacimiento, muerte), donde muerte puede ser inf.
    """
    stats = []
    diagram = array2dict(diagram)
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
    diagram = array2dict(diagram)
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


def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    marker="o",
    facecolors=None,
    edgecolors=None,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    for i, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            diagrams[i] = np.empty((0, 2), dtype=np.float32)

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, marker=marker, facecolors=facecolors, edgecolors=edgecolors)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()