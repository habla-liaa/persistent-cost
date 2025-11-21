EXPERIMENTOS

Queremos comparar los resultados de usar los métodos pipeline provistos por cone, cone2 y cylinder en persistent_cost.

Todos los ejemplos propuestos son de la forma: X, Y espacios con una distancia definida, y una funcion “f” saliendo de X y llegando a Y. 

Para cada uno de los experimentos de abajo armar una función que genere X, Y y f con una semilla fija para reproducibilidad, un valor n para el tamaño de las nubes (donde aplique) y un valor dim para la dimensión, por defecto 2 salvo dicho otra cosa. Correr los experimentos para n=50 y 100.

* Inclusion del punto 
Espacio X: un punto
Espacio Y: nube random
Función f: inclusion en el primer punto
* Producto
Espacio X: nube random
Espacio Y: igual a X
Función f: identidad
* Suspensión
Espacio X: nube random
Espacio Y: un punto
Función f: todo mapeados al unico punto de la salida
* Toro proyecta 
Espacio X: muestreo aleatorio de una superficie tórica (interior vacio), radios r=1 y R=2 en dimension 3.
Espacio Y: proyecciones a las primeras dos coordenadas, aquellas en las que gira con radio R=2.
Función f: “identidad”, cada punto a su proyección.
* Circulo en el toro 
Espacio X: un círculo de radio 1 que entra como sección transversal del toro anterior, muestreo de cardinalidad “k” (aleatorio).
Espacio Y: círculo X unión muestreo del toro como en el experimento 4
Función f: inclusion del círculo X en su copio idéntica dentro de Y, asignación identidad en los primeros “k”. 
* Muestreo random
Espacio X: la mitad de los puntos de Y.
Espacio Y: una nube random. 
Función f: inclusion de X en Y.


FORMATO OUTPUT

* constante Lipschitz antes de la normalizacion
“n” de los espacios.
* Gráficar homología persistente para 
X, Y, Cono, Ker, Coker y Cilindro  (6 gráficos).
* Anexar lista de barras de los 6 gráficos.
* Informar la clasificación de las barras del cono como (co)nucleo o desapareadas.
* Expandir las desapareadas (o todas) con su dimensión, simplex crítico, nacimiento y muerte. 
