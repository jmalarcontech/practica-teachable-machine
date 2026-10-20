# Clasificador de imágenes con Teachable Machine

Práctica de Inteligencia Artificial (2º DAM, IES Zaidín-Vergeles). Modelo de clasificación de imágenes entrenado con [Teachable Machine](https://teachablemachine.withgoogle.com/) que distingue entre un smartphone y un cargador. Se ha exportado a TensorFlow.js y se ejecuta en el navegador desde una página web propia.

Página publicada: https://USUARIO.github.io/practica-teachable-machine/clasificador/

## Clases

- Smartphone
- Cargador
- Fondo (la escena sin ningún objeto)

## Funcionalidades

- Clasificación en directo con la cámara frontal o trasera, con botones de inicio y parada.
- Mensaje personalizado para cada clase.
- Umbral mínimo de coincidencia ajustable.
- Histórico de resultados guardado en el navegador y exportable a CSV.
- Clasificación de imágenes subidas o arrastradas a la página.
- Diseño propio, adaptado a móvil y con modo oscuro.

## Estructura

```
practica-teachable-machine/
├── README.md
├── memoria.ipynb        Memoria de la práctica (Google Colab)
├── clasificador/        Página con las mejoras
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── modelo/
│       ├── model.json
│       ├── metadata.json
│       └── weights.bin
└── version-base/        Código original generado por Teachable Machine
    └── index.html
```

## Ejecución en local

El modelo no se carga si se abre `index.html` con doble clic. Hay que usar un servidor local desde la carpeta del proyecto:

```bash
python3 -m http.server 8000
```

y abrir http://localhost:8000/clasificador/

## Tecnologías

HTML, CSS y JavaScript, con [TensorFlow.js](https://www.tensorflow.org/js) 1.3.1 y la librería [@teachablemachine/image](https://www.npmjs.com/package/@teachablemachine/image) 0.8.5.

## Autor

Antonio Jesús Molina Alarcón
