"""
Funciones para generar los espacios X, Y y la función f para cada experimento.
"""

import numpy as np
from scipy.spatial.distance import pdist


def inclusion_punto(n=50, dim=2, seed=42):
    """
    Experimento 1: Inclusión del punto
    X: un punto
    Y: nube random
    f: inclusión en el primer punto
    """
    rng = np.random.default_rng(seed)
    
    Y = rng.random((n, dim))
    X = Y[0:1].copy()  # X es el primer punto de Y
    f = np.array([0])  # el único punto de X mapea al primer punto de Y
    
    return X, Y, f


def producto(n=50, dim=2, seed=42):
    """
    Experimento 2: Producto
    X: nube random
    Y: igual a X
    f: identidad
    """
    rng = np.random.default_rng(seed)
    
    X = rng.random((n, dim))
    Y = X.copy()
    f = np.arange(n)  # identidad
    
    return X, Y, f


def suspension(n=50, dim=2, seed=42):
    """
    Experimento 3: Suspensión
    X: nube random
    Y: un punto
    f: todo mapeado al único punto de la salida
    """
    rng = np.random.default_rng(seed)
    
    X = rng.random((n, dim))
    Y = X[:1].copy()  # Y es un único punto
    f = [0] * n  # todo mapeado al único punto de Y
    
    return X, Y, f


def toro_proyecta(n=50, seed=42):
    """
    Experimento 4: Toro proyecta
    X: muestreo aleatorio de una superficie tórica (interior vacío), radios r=1 y R=2 en dimensión 3
    Y: proyecciones a las primeras dos coordenadas
    f: identidad (cada punto a su proyección)
    """
    rng = np.random.default_rng(seed)
    
    # Generar puntos en la superficie del toro
    # Parámetros: r=1 (radio menor), R=2 (radio mayor)
    r, R = 1.0, 2.0
    
    # Ángulos uniformes
    theta = rng.uniform(0, 2*np.pi, n)
    phi = rng.uniform(0, 2*np.pi, n)
    
    # Ecuaciones paramétricas del toro
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    X = np.column_stack([x, y, z])
    
    # Y es la proyección a las primeras dos coordenadas
    Y = X[:, :2].copy()
    
    # f es la identidad (cada punto de X mapea a su proyección en Y)
    f = np.arange(n)
    
    return X, Y, f


def circulo_en_toro(n=50, k=None, seed=42):
    """
    Experimento 5: Círculo en el toro
    X: un círculo de radio 1 que entra como sección transversal del toro
    Y: círculo X unión muestreo del toro
    f: inclusión del círculo X en su copia idéntica dentro de Y
    
    k: cardinalidad del círculo (si None, usa n//5)
    """
    rng = np.random.default_rng(seed)
    
    if k is None:
        k = max(10, n // 5)
    
    # Generar círculo X de radio 1 (sección transversal del toro)
    # Lo ubicamos en el plano x=R (donde R=2)
    r, R = 1.0, 2.0
    theta_circle = np.linspace(0, 2*np.pi, k, endpoint=False)
    
    y_circle = np.zeros(k)
    z_circle = r * np.sin(theta_circle)
    # Ajustamos para que sea un círculo en el plano correcto
    x_circle = r * np.cos(theta_circle)
    
    X = np.column_stack([x_circle, y_circle, z_circle])
    
    # Generar muestreo del toro (n-k puntos adicionales)
    n_toro = n - k
    theta = rng.uniform(0, 2*np.pi, n_toro)
    phi = rng.uniform(0, 2*np.pi, n_toro)
    
    x_toro = (R + r * np.cos(theta)) * np.cos(phi)
    y_toro = (R + r * np.cos(theta)) * np.sin(phi)
    z_toro = r * np.sin(theta)
    
    toro_points = np.column_stack([x_toro, y_toro, z_toro])
    
    # Y es la unión del círculo X y el muestreo del toro
    Y = np.vstack([X, toro_points])
    
    # f: los primeros k puntos de X mapean a los primeros k puntos de Y (identidad)
    f = np.arange(k)
    
    return X, Y, f


def muestreo_random(n=50, dim=2, seed=42):
    """
    Experimento 6: Muestreo random
    X: la mitad de los puntos de Y
    Y: una nube random
    f: inclusión de X en Y
    """
    rng = np.random.default_rng(seed)
    
    # Generar nube Y
    Y = rng.random((n, dim))
    
    # X es la mitad de los puntos de Y
    n_x = n // 2
    indices = rng.choice(n, size=n_x, replace=False)
    indices = np.sort(indices)  # mantener orden
    
    X = Y[indices].copy()
    f = indices  # mapeo a los índices seleccionados
    
    return X, Y, f

