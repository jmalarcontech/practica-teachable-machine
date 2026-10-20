# Clasificador de imágenes con Teachable Machine

Práctica de Inteligencia Artificial (2º DAM, IES Zaidín-Vergeles). Un modelo entrenado con [Teachable Machine](https://teachablemachine.withgoogle.com/) distingue entre un smartphone y un cargador, y una aplicación web lo ejecuta en el navegador con la cámara o con fotos.

**Aplicación:** https://jmalarcontech.github.io/practica-teachable-machine/clasificador/

## Qué hace

Las siete mejoras del enunciado:

- Diseño propio, con tema claro y oscuro.
- Diseño adaptable a móvil, tableta y escritorio.
- Botones para iniciar y detener la cámara.
- Mensaje personalizado para cada clase, editable desde los ajustes.
- Umbral mínimo de coincidencia, con la misma escala que las barras de probabilidad.
- Histórico de resultados guardado en el navegador, con filtros y exportación a CSV y JSON.
- Clasificación de imágenes subidas, arrastradas o pegadas, también varias a la vez.

Y además:

- **Mapa de atención (Grad-CAM):** resalta la zona de la imagen en la que se ha fijado el modelo.
- **TensorFlow.js 4.22** en lugar de la librería de Teachable Machine (TensorFlow.js 1.3.1), con el mismo preprocesado. La página `comprobacion/` demuestra que los resultados coinciden.
- **Elección automática del motor de cálculo:** WebGL, WebAssembly o CPU, con WebGPU opcional.
- Cronograma de los últimos 10 segundos, ficha del modelo y datos de rendimiento.
- Carga directa del `.zip` que descarga Teachable Machine.
- Atajos de teclado y compatibilidad con lectores de pantalla.

Todo el cálculo ocurre en el navegador: ninguna imagen sale del dispositivo.

## Estructura

```
practica-teachable-machine/
├── index.html           Redirige al clasificador
├── memoria.ipynb        Memoria de la práctica (Google Colab)
├── clasificador/        Aplicación
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── modelo/
│       ├── model.json
│       ├── metadata.json
│       └── weights.bin
├── comprobacion/        Compara los resultados con la librería oficial
└── version-base/        Código original generado por Teachable Machine
```

## Ejecución en local

El navegador no deja cargar el modelo si `index.html` se abre con doble clic. Hay que usar un servidor local desde la carpeta del proyecto:

```bash
python3 -m http.server 8000
```

y abrir http://localhost:8000/clasificador/

## Usar otro modelo

Exporta el modelo en Teachable Machine con la opción TensorFlow.js y copia `model.json`, `metadata.json` y `weights.bin` en `clasificador/modelo/`. Para probarlo sin copiarlo, abre la ficha del modelo (botón *Modelo*) y carga el `.zip` descargado.

## Tecnologías

HTML, CSS y JavaScript sin dependencias aparte de [TensorFlow.js](https://www.tensorflow.org/js) 4.22. La tipografía es [Archivo](https://fonts.google.com/specimen/Archivo).

## Autor

Antonio Jesús Molina Alarcón
