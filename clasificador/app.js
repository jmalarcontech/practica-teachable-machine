// Clasificador de imágenes: smartphone o cargador
// Modelo entrenado en Teachable Machine y ejecutado en el navegador con TensorFlow.js 4.

/* =========================================================================
   1. Configuración
   ========================================================================= */

const VERSION_TF = '4.22.0';
const CDN = 'https://cdn.jsdelivr.net/npm/';
// La carpeta del modelo se busca junto a este archivo, sea cual sea la página que lo cargue
const RUTA_MODELO = new URL('modelo/', document.currentScript ? document.currentScript.src : location.href);
const CLAVE = 'clasificador.';
const MAX_HISTORICO = 100;
const VENTANA_GRAFICA = 10000; // milisegundos visibles en el cronograma
const TAU_SUAVIZADO = 180; // constante de tiempo de la media móvil, en milisegundos
const TAU_MAPA = 90; // el mapa se suaviza menos para que siga al objeto sin retraso
const LADO_MINIATURA = 192;
const MENSAJE_DUDA = 'El modelo no está seguro de lo que ve.';

const PREFERENCIAS_POR_DEFECTO = {
  umbral: 80,
  suavizado: true,
  atencion: true,
  registroAuto: true,
  estabilidad: 800,
  espejo: true,
  camara: '',
  motor: 'auto',
  tema: 'auto',
  mensajes: {},
  mensajeDuda: '',
};

const MOTORES = {
  webgl: { nombre: 'WebGL' },
  webgpu: {
    nombre: 'WebGPU',
    script: `${CDN}@tensorflow/tfjs-backend-webgpu@${VERSION_TF}/dist/tf-backend-webgpu.min.js`,
  },
  wasm: {
    nombre: 'WebAssembly',
    script: `${CDN}@tensorflow/tfjs-backend-wasm@${VERSION_TF}/dist/tf-backend-wasm.min.js`,
    rutas: `${CDN}@tensorflow/tfjs-backend-wasm@${VERSION_TF}/dist/`,
  },
  cpu: { nombre: 'CPU' },
};

// Nombres de clase que significan "no hay objeto"; se dibujan en gris
const NOMBRES_FONDO = ['fondo', 'background', 'nada', 'ninguno', 'vacio', 'sin objeto', 'otro', 'otros'];

/* =========================================================================
   2. Utilidades
   ========================================================================= */

const $ = (selector, raiz = document) => raiz.querySelector(selector);

const formatoDecimal = new Intl.NumberFormat('es-ES', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const formatoEntero = new Intl.NumberFormat('es-ES');
const formatoHora = new Intl.DateTimeFormat('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const formatoDia = new Intl.DateTimeFormat('es-ES', { day: 'numeric', month: 'short' });
const formatoFechaLarga = new Intl.DateTimeFormat('es-ES', { dateStyle: 'long', timeStyle: 'short' });

const porcentaje = (p) => `${formatoDecimal.format(p * 100)} %`;
const porcentajeEntero = (p) => `${Math.round(p * 100)} %`;
const normalizar = (texto) => texto.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
const esFondo = (etiqueta) => NOMBRES_FONDO.includes(normalizar(etiqueta));

function formatoTamano(bytes) {
  if (bytes < 1024 * 1024) return `${formatoEntero.format(Math.round(bytes / 1024))} KB`;
  return `${formatoDecimal.format(bytes / (1024 * 1024))} MB`;
}

function fechaCorta(iso) {
  const fecha = new Date(iso);
  const hoy = new Date().toDateString() === fecha.toDateString();
  return hoy ? formatoHora.format(fecha) : `${formatoDia.format(fecha)}, ${formatoHora.format(fecha)}`;
}

function cargarScript(src) {
  return new Promise((resolver, rechazar) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.onload = () => resolver();
    script.onerror = () => rechazar(new Error(`No se ha podido cargar ${src}`));
    document.head.append(script);
  });
}

// localStorage puede fallar (modo privado, almacenamiento lleno); nunca debe romper la página
function leerDato(clave) {
  try {
    return JSON.parse(localStorage.getItem(CLAVE + clave));
  } catch (error) {
    return null;
  }
}

function guardarDato(clave, valor) {
  try {
    localStorage.setItem(CLAVE + clave, JSON.stringify(valor));
    return true;
  } catch (error) {
    return false;
  }
}

function descargar(nombre, contenido, tipo) {
  const enlace = document.createElement('a');
  enlace.href = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  enlace.download = nombre;
  document.body.append(enlace);
  enlace.click();
  enlace.remove();
  setTimeout(() => URL.revokeObjectURL(enlace.href), 1000);
}

function marcaDeTiempo() {
  const f = new Date();
  const dos = (n) => String(n).padStart(2, '0');
  return `${f.getFullYear()}${dos(f.getMonth() + 1)}${dos(f.getDate())}-${dos(f.getHours())}${dos(f.getMinutes())}`;
}

function hexARgb(hex) {
  const limpio = hex.trim().replace('#', '');
  const valor = parseInt(limpio.length === 3 ? limpio.replace(/./g, '$&$&') : limpio, 16);
  return [(valor >> 16) & 255, (valor >> 8) & 255, valor & 255];
}

/* =========================================================================
   3. Preprocesado: réplica exacta de Teachable Machine
   ========================================================================= */

// Igual que cropTo() de @teachablemachine/image: escala la imagen para que su lado
// menor mida "lado" píxeles, se queda con el cuadrado central y, si se pide, lo voltea.
function recortarCentro(origen, lado, espejo, lienzo) {
  const ancho = origen.videoWidth || origen.naturalWidth || origen.width;
  const alto = origen.videoHeight || origen.naturalHeight || origen.height;
  const escala = lado / Math.min(ancho, alto);
  const anchoEscalado = Math.ceil(ancho * escala);
  const altoEscalado = Math.ceil(alto * escala);

  lienzo.width = lado;
  lienzo.height = lado;
  const ctx = lienzo.getContext('2d');
  ctx.save();
  if (espejo) {
    ctx.translate(lado, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(
    origen,
    -Math.floor((anchoEscalado - lado) / 2),
    -Math.floor((altoEscalado - lado) / 2),
    anchoEscalado,
    altoEscalado
  );
  ctx.restore();
  return lienzo;
}

/* =========================================================================
   4. Modelo
   ========================================================================= */

class ErrorModelo extends Error {}

class Clasificador {
  constructor(modelo, metadatos, artefactos) {
    this.modelo = modelo;
    this.metadatos = metadatos;
    this.artefactos = artefactos;
    this.etiquetas = metadatos.labels.map(String);
    this.tamano = metadatos.imageSize || 224;
    this.grises = Boolean(metadatos.grayscale);
    this.lienzo = document.createElement('canvas');
    this.cola = Promise.resolve();
    this.dividir();
  }

  static async crear(artefactos, metadatos) {
    if (!metadatos || !Array.isArray(metadatos.labels) || metadatos.labels.length < 2) {
      throw new ErrorModelo('metadata.json no contiene la lista de clases.');
    }
    const modelo = await tf.loadLayersModel(tf.io.fromMemory(artefactos));
    const salidas = modelo.outputs[0].shape[modelo.outputs[0].shape.length - 1];
    if (salidas !== metadatos.labels.length) {
      modelo.dispose();
      throw new ErrorModelo(`El modelo tiene ${salidas} salidas, pero metadata.json nombra ${metadatos.labels.length} clases.`);
    }
    return new Clasificador(modelo, metadatos, artefactos);
  }

  // El modelo de Teachable Machine es una secuencia de bloques:
  // MobileNet (salida 7 × 7 × 1280) → media global → capas densas → softmax.
  // Se separa en "extractor" (hasta la última salida con forma de imagen) y "cabeza",
  // porque el mapa de atención necesita los rasgos intermedios.
  dividir() {
    const bloques = [];
    const recorrer = (modelo) => {
      for (const capa of modelo.layers) {
        if (capa.getClassName() === 'Sequential') recorrer(capa);
        else bloques.push(capa);
      }
    };
    recorrer(this.modelo);

    const esMapa = (forma) =>
      Array.isArray(forma) && forma.length === 4 && forma.every((d) => d === null || typeof d === 'number');
    let corte = -1;
    bloques.forEach((bloque, i) => {
      if (esMapa(bloque.outputShape)) corte = i;
    });

    const aplicar = (lista, x) => lista.reduce((tensor, capa) => capa.apply(tensor), x);

    if (corte < 0 || corte === bloques.length - 1) {
      this.extraer = (x) => x;
      this.clasificar = (x) => this.modelo.predict(x);
      this.atencionDisponible = false;
      return;
    }

    const extractor = bloques.slice(0, corte + 1);
    const cabeza = bloques.slice(corte + 1);
    this.extraer = (x) => aplicar(extractor, x);
    this.clasificar = (rasgos) => aplicar(cabeza, rasgos);

    // Grad-CAM funciona mejor con la puntuación previa al softmax: si el modelo está
    // muy seguro, el softmax se satura y sus gradientes casi desaparecen.
    const ultima = cabeza[cabeza.length - 1];
    const config = ultima.getConfig();
    if (ultima.getClassName() === 'Dense' && config.activation === 'softmax') {
      const [nucleo, sesgo] = ultima.getWeights();
      const previas = cabeza.slice(0, -1);
      this.puntuar = (rasgos) => {
        const z = tf.matMul(aplicar(previas, rasgos), nucleo);
        return config.useBias && sesgo ? tf.add(z, sesgo) : z;
      };
    } else {
      this.puntuar = this.clasificar;
    }

    const forma = bloques[corte].outputShape;
    this.ladoMapa = { alto: forma[1], ancho: forma[2] };
    this.rasgos = forma[3];
    this.atencionDisponible = true;
  }

  // Las inferencias se encadenan para que nunca se pisen el lienzo compartido
  analizar(origen, opciones = {}) {
    const tarea = this.cola.then(() => this.analizarAhora(origen, opciones));
    this.cola = tarea.catch(() => {});
    return tarea;
  }

  async analizarAhora(origen, { espejo = false, atencion = false } = {}) {
    recortarCentro(origen, this.tamano, espejo, this.lienzo);
    const inicio = performance.now();
    const conMapa = atencion && this.atencionDisponible;

    const { prob, mapa } = tf.tidy(() => {
      let x = tf.cast(tf.browser.fromPixels(this.lienzo), 'float32');
      if (this.grises) x = tf.sum(tf.mul(x, tf.tensor1d([0.2989, 0.587, 0.114])), -1, true);
      // Misma normalización que Teachable Machine: de 0..255 a -1..1 dividiendo entre 127
      x = tf.expandDims(tf.sub(tf.div(x, 127), 1), 0);

      const rasgos = this.extraer(x);
      const prob = tf.squeeze(this.clasificar(rasgos), [0]);
      if (!conMapa) return { prob, mapa: null };

      // Grad-CAM: cuánto cambia la puntuación de la clase ganadora con cada rasgo,
      // promediado por canal, y combinado con los propios rasgos en cada posición.
      const gradiente = tf.grad((r) => tf.sum(tf.max(this.puntuar(r), 1)))(rasgos);
      const pesos = tf.mean(gradiente, [1, 2], true);
      const bruto = tf.relu(tf.sum(tf.mul(rasgos, pesos), 3));
      const mapa = tf.squeeze(tf.div(bruto, tf.add(tf.max(bruto), 1e-7)), [0]);
      return { prob, mapa };
    });

    const [probabilidades, valoresMapa] = await Promise.all([prob.data(), mapa ? mapa.data() : null]);
    tf.dispose([prob, mapa]);
    return { probabilidades, mapa: valoresMapa, ms: performance.now() - inicio };
  }

  // La primera inferencia compila los programas de la tarjeta gráfica; se hace al cargar
  async calentar() {
    const lienzo = document.createElement('canvas');
    lienzo.width = lienzo.height = this.tamano;
    const inicio = performance.now();
    await this.analizar(lienzo, { atencion: this.atencionDisponible });
    return performance.now() - inicio;
  }

  get bytesPesos() {
    const datos = this.artefactos.weightData;
    return Array.isArray(datos) ? datos.reduce((total, b) => total + b.byteLength, 0) : datos.byteLength;
  }

  get parametros() {
    return this.modelo.countParams();
  }

  liberar() {
    this.modelo.dispose();
  }
}

async function leerModeloDelProyecto(base = RUTA_MODELO, alProgresar) {
  const urlModelo = new URL('model.json', base).href;
  const urlMetadatos = new URL('metadata.json', base).href;

  const respuesta = await fetch(urlMetadatos, { cache: 'no-cache' });
  if (!respuesta.ok) throw new ErrorModelo(`No se encuentra metadata.json (código ${respuesta.status}).`);
  const metadatos = await respuesta.json();

  const artefactos = await tf.io
    .http(urlModelo, { requestInit: { cache: 'no-cache' }, onProgress: alProgresar })
    .load();
  return { artefactos, metadatos };
}

// Acepta el .zip que descarga Teachable Machine o sus tres archivos sueltos
async function leerModeloDeArchivos(lista) {
  let archivos = [...lista];
  const zip = archivos.find((a) => /\.zip$/i.test(a.name));
  if (zip) archivos = await leerZip(zip);

  let modelo = null;
  let metadatos = null;
  for (const archivo of archivos.filter((a) => /\.json$/i.test(a.name))) {
    try {
      const datos = JSON.parse(await archivo.text());
      if (datos.modelTopology) modelo = datos;
      else if (Array.isArray(datos.labels)) metadatos = datos;
    } catch (error) {
      // Archivo JSON ajeno al modelo: se ignora
    }
  }
  const pesos = archivos.filter((a) => /\.bin$/i.test(a.name));

  const faltan = [!modelo && 'model.json', !metadatos && 'metadata.json', !pesos.length && 'weights.bin'].filter(Boolean);
  if (faltan.length) {
    throw new ErrorModelo(`Falta ${faltan.join(' y ')}. Selecciona a la vez los tres archivos del modelo o el .zip completo.`);
  }

  const rutas = modelo.weightsManifest.flatMap((grupo) => grupo.paths);
  const buffers = await Promise.all(
    rutas.map((ruta, i) => {
      const nombre = ruta.split('/').pop();
      // Si el navegador renombró el archivo al descargarlo (weights (1).bin), se usa por orden
      const archivo = pesos.find((p) => p.name === nombre) || pesos[i];
      if (!archivo) throw new ErrorModelo(`Falta el archivo de pesos ${nombre}.`);
      return archivo.arrayBuffer();
    })
  );
  const total = buffers.reduce((suma, b) => suma + b.byteLength, 0);
  const unidos = new Uint8Array(total);
  let desplazamiento = 0;
  for (const b of buffers) {
    unidos.set(new Uint8Array(b), desplazamiento);
    desplazamiento += b.byteLength;
  }

  const artefactos = {
    modelTopology: modelo.modelTopology,
    weightSpecs: modelo.weightsManifest.flatMap((grupo) => grupo.weights),
    weightData: unidos.buffer,
    format: modelo.format,
    generatedBy: modelo.generatedBy,
    convertedBy: modelo.convertedBy,
  };
  return { artefactos, metadatos };
}

// Lector mínimo de ZIP: recorre el directorio central y descomprime con DecompressionStream
async function leerZip(archivo) {
  if (typeof DecompressionStream === 'undefined') {
    throw new ErrorModelo('Este navegador no puede abrir archivos .zip. Descomprímelo y selecciona los tres archivos.');
  }
  const buffer = await archivo.arrayBuffer();
  const datos = new DataView(buffer);

  let fin = -1;
  for (let i = datos.byteLength - 22; i >= Math.max(0, datos.byteLength - 65557); i--) {
    if (datos.getUint32(i, true) === 0x06054b50) {
      fin = i;
      break;
    }
  }
  if (fin < 0) throw new ErrorModelo('El archivo .zip está dañado o no es un .zip.');

  const entradas = datos.getUint16(fin + 10, true);
  let posicion = datos.getUint32(fin + 16, true);
  const decodificador = new TextDecoder();
  const archivos = [];

  for (let n = 0; n < entradas; n++) {
    if (datos.getUint32(posicion, true) !== 0x02014b50) break;
    const metodo = datos.getUint16(posicion + 10, true);
    const comprimido = datos.getUint32(posicion + 20, true);
    const largoNombre = datos.getUint16(posicion + 28, true);
    const largoExtra = datos.getUint16(posicion + 30, true);
    const largoComentario = datos.getUint16(posicion + 32, true);
    const local = datos.getUint32(posicion + 42, true);
    const ruta = decodificador.decode(new Uint8Array(buffer, posicion + 46, largoNombre));
    posicion += 46 + largoNombre + largoExtra + largoComentario;

    const nombre = ruta.split('/').pop();
    if (!nombre || ruta.includes('__MACOSX/') || nombre.startsWith('._')) continue;

    const inicio = local + 30 + datos.getUint16(local + 26, true) + datos.getUint16(local + 28, true);
    const bruto = new Uint8Array(buffer, inicio, comprimido);
    let contenido;
    if (metodo === 0) contenido = bruto;
    else if (metodo === 8) {
      const flujo = new Blob([bruto]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
      contenido = await new Response(flujo).arrayBuffer();
    } else continue;
    archivos.push(new File([contenido], nombre));
  }
  return archivos;
}

/* =========================================================================
   5. Motores de cálculo
   ========================================================================= */

let rutasWasmFijadas = false;
let hayWebGPU = null;

// Algunos navegadores tienen la API de WebGPU pero ninguna tarjeta compatible: se pide un adaptador
async function comprobarWebGPU() {
  if (hayWebGPU !== null) return hayWebGPU;
  try {
    hayWebGPU = Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch (error) {
    hayWebGPU = false;
  }
  return hayWebGPU;
}

async function activarMotor(nombre) {
  const motor = MOTORES[nombre];
  if (!motor) return false;
  if (nombre === 'webgpu' && !(await comprobarWebGPU())) return false;
  if (motor.script && !tf.findBackendFactory(nombre)) await cargarScript(motor.script);
  if (nombre === 'wasm' && !rutasWasmFijadas) {
    tf.wasm.setWasmPaths(motor.rutas);
    rutasWasmFijadas = true;
  }
  if (!(await tf.setBackend(nombre))) return false;
  await tf.ready();
  return tf.getBackend() === nombre;
}

// TensorFlow.js rechaza WebGL en equipos con la tarjeta gráfica bloqueada o por software.
// En ese caso WebAssembly es mucho más rápido que la CPU, así que se prueba antes.
async function construirClasificador(artefactos, metadatos, preferido = 'auto') {
  const orden = [...new Set([preferido === 'auto' ? null : preferido, 'webgl', 'wasm', 'cpu'].filter(Boolean))];
  let ultimoError = null;
  for (const nombre of orden) {
    let activo = false;
    try {
      activo = await activarMotor(nombre);
    } catch (error) {
      console.warn(`Motor ${nombre} no disponible`, error);
    }
    if (!activo) continue;
    try {
      const clasificador = await Clasificador.crear(artefactos, metadatos);
      const msCalentar = await clasificador.calentar();
      return { clasificador, motor: nombre, msCalentar };
    } catch (error) {
      if (error instanceof ErrorModelo) throw error;
      console.warn(`El modelo falla con ${nombre}`, error);
      ultimoError = error;
    }
  }
  throw ultimoError || new Error('No hay ningún motor de cálculo disponible.');
}

/* =========================================================================
   6. Estado de la aplicación
   ========================================================================= */

const estado = {
  prefs: { ...PREFERENCIAS_POR_DEFECTO },
  clasificador: null,
  motor: null,
  fuente: 'proyecto',
  nombreFuente: '',
  tiempos: { carga: 0, calentar: 0 },
  modo: 'camara',
  camara: { stream: null, activa: false, id: 0, porFotograma: false, orientacion: 'user' },
  vivo: { probs: null, mapa: null, t: 0 },
  serie: [],
  hover: null,
  candidato: { clave: null, desde: 0 },
  ultimaRegistrada: null,
  anuncio: { clave: null, desde: 0, hecho: null },
  lote: [],
  loteActual: -1,
  historico: [],
  filtro: 'todos',
  detalle: null,
  rendimiento: { ms: 0, fps: 0, ultimo: 0, pintado: 0 },
  colores: [],
  rgb: [],
  filas: [],
};

let ui = {};

function referenciarInterfaz() {
  const ids = [
    'estadoModelo', 'btnModelo', 'btnAjustes', 'visor', 'video', 'imagenVista', 'recorte', 'mapa',
    'visorEtiqueta', 'leyenda', 'visorAviso', 'progreso', 'selectorModo', 'interruptorAtencion',
    'btnIniciar', 'btnDetener', 'btnRegistrar', 'camaraRapida', 'selCamaraRapida', 'btnElegir',
    'pistaPegar', 'inputImagenes', 'lote', 'loteResumen', 'loteLista', 'lectura', 'veredicto',
    'mensaje', 'detalle', 'confianzaCaja', 'confianza', 'escala', 'umbral', 'valorUmbral',
    'cronograma', 'grafica', 'graficaTip', 'telMotor', 'telMs', 'telFps', 'telVelocidad', 'contadorHistorico',
    'btnCsv', 'btnJson', 'btnVaciar', 'resumenHistorico', 'filtros', 'distribucion',
    'confianzaMedia', 'tablaCaja', 'cuerpoHistorico', 'historicoVacio', 'dlgAjustes',
    'listaMensajes', 'btnRestablecerMensajes', 'selCamara', 'ayudaDispositivo', 'selMotor',
    'ayudaMotor', 'selectorTema', 'btnRestablecerAjustes', 'dlgModelo', 'fichaModelo',
    'btnCargarModelo', 'btnModeloProyecto', 'inputModelo', 'dlgDetalle', 'tituloDetalle',
    'detalleImagen', 'detalleMapa', 'detalleMensaje', 'detalleFicha', 'detalleEscala',
    'detalleInterruptor', 'detalleAtencion', 'btnEliminarRegistro', 'dlgConfirmar',
    'tituloConfirmar', 'textoConfirmar', 'btnConfirmar', 'soltar', 'avisos', 'anuncio',
  ];
  ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
  ui.avisoTitulo = $('.visor-aviso-titulo', ui.visorAviso);
  ui.avisoTexto = $('.visor-aviso-texto', ui.visorAviso);
  ui.avisoBoton = $('.boton', ui.visorAviso);
  ui.filaUmbral = $('.escala-umbral', ui.escala);
  ui.accionesCamara = $('.acciones[data-para="camara"]');
  ui.accionesImagen = $('.acciones[data-para="imagen"]');
}

/* =========================================================================
   7. Carga del modelo
   ========================================================================= */

async function cargarModeloDelProyecto() {
  ponerEstadoModelo('cargando', 'Cargando modelo');
  mostrarAvisoVisor({ titulo: 'Cargando el modelo', progreso: 0 });
  const inicio = performance.now();
  let datos;
  try {
    datos = await leerModeloDelProyecto(RUTA_MODELO, (fraccion) => {
      ui.progreso.style.setProperty('--progreso', fraccion);
      ui.avisoTexto.textContent = `${Math.round(fraccion * 100)} %`;
      ui.avisoTexto.hidden = false;
    });
  } catch (error) {
    console.info('Modelo del proyecto no disponible:', error.message);
    mostrarSinModelo();
    return;
  }
  await usarModelo(datos, { fuente: 'proyecto', inicio });
}

function mostrarSinModelo() {
  const local = location.protocol === 'file:';
  ponerEstadoModelo('error', 'Sin modelo');
  mostrarAvisoVisor({
    titulo: local ? 'Abre la página con un servidor local' : 'No se encuentra el modelo',
    texto: local
      ? 'El navegador no deja leer el modelo desde un archivo abierto con doble clic. Usa Live Server o python3 -m http.server, o carga los archivos del modelo desde tu equipo.'
      : 'Copia model.json, metadata.json y weights.bin en la carpeta modelo, o carga los archivos desde tu equipo.',
    boton: { texto: 'Cargar archivos del modelo', accion: () => ui.inputModelo.click() },
  });
  actualizarDisponibilidad();
  pintarFicha();
}

async function cargarModeloDeArchivos(lista) {
  // Se copia la lista enseguida: el selector de archivos se vacía justo después
  const archivos = [...lista];
  const nombres = archivos.map((a) => a.name).join(', ');
  const inicio = performance.now();
  let datos;
  try {
    datos = await leerModeloDeArchivos(archivos);
  } catch (error) {
    aviso(error instanceof ErrorModelo ? error.message : 'No se han podido leer los archivos del modelo.');
    return;
  }
  if (ui.dlgModelo.open) ui.dlgModelo.close();
  const ok = await usarModelo(datos, { fuente: 'archivos', nombre: nombres, inicio });
  if (ok) aviso('Modelo cargado desde tus archivos');
}

// Las cargas de modelo se encadenan: nunca se preparan dos a la vez
let colaModelos = Promise.resolve();
function usarModelo(datos, opciones) {
  const tarea = colaModelos.then(() => prepararModelo(datos, opciones));
  colaModelos = tarea.catch(() => {});
  return tarea;
}

async function prepararModelo({ artefactos, metadatos }, { fuente, nombre = '', inicio = performance.now() }) {
  detenerCamara();
  ponerEstadoModelo('cargando', 'Preparando modelo');
  mostrarAvisoVisor({ titulo: 'Preparando el modelo', texto: 'Compilando las operaciones para este equipo.' });

  let resultado;
  try {
    resultado = await construirClasificador(artefactos, metadatos, estado.prefs.motor);
  } catch (error) {
    console.error(error);
    aviso(error instanceof ErrorModelo ? error.message : 'No se ha podido preparar el modelo en este navegador.');
    if (estado.clasificador) {
      ponerEstadoModelo('listo', 'Modelo listo');
      mostrarVisorInactivo();
    } else {
      mostrarSinModelo();
    }
    return false;
  }

  const anterior = estado.clasificador;
  if (anterior && anterior !== resultado.clasificador) anterior.liberar();

  estado.clasificador = resultado.clasificador;
  estado.motor = resultado.motor;
  estado.fuente = fuente;
  estado.nombreFuente = nombre;
  estado.tiempos = { carga: performance.now() - inicio - resultado.msCalentar, calentar: resultado.msCalentar };
  estado.lote.forEach((elemento) => URL.revokeObjectURL(elemento.url));
  estado.lote = [];
  estado.loteActual = -1;
  estado.serie = [];

  if (estado.prefs.motor !== 'auto' && estado.prefs.motor !== resultado.motor) {
    aviso(`${MOTORES[estado.prefs.motor].nombre} no está disponible en este navegador. Se usa ${MOTORES[resultado.motor].nombre}.`);
  }

  prepararClases();
  ponerEstadoModelo('listo', 'Modelo listo');
  ui.telMotor.textContent = MOTORES[estado.motor].nombre;
  ui.telMs.textContent = 'Sin datos';
  ui.telFps.textContent = 'Sin datos';
  reiniciarLectura();
  pintarLote();
  mostrarVisorInactivo();
  actualizarDisponibilidad();
  pintarFicha();
  pintarHistorico();
  rellenarExplicacion();
  dibujarGrafica();
  return true;
}

// Colores, filas de probabilidad y mensajes de cada clase
function prepararClases() {
  const { etiquetas } = estado.clasificador;
  let ranura = 0;
  estado.colores = etiquetas.map((etiqueta) => {
    if (esFondo(etiqueta) || ranura >= 8) return 'var(--serie-fondo)';
    ranura += 1;
    return `var(--serie-${ranura})`;
  });
  resolverColores();

  ui.escala.querySelectorAll('.escala-fila:not(.escala-umbral)').forEach((fila) => fila.remove());
  estado.filas = etiquetas.map((etiqueta, i) => {
    const fila = crearFilaEscala(etiqueta, estado.colores[i]);
    ui.escala.insertBefore(fila.raiz, ui.filaUmbral);
    return fila;
  });

  pintarMensajesAjustes();
  ajustarVeredicto();
}

function crearFilaEscala(etiqueta, color) {
  const raiz = document.createElement('div');
  raiz.className = 'escala-fila';
  raiz.style.setProperty('--color', color);
  raiz.innerHTML =
    '<span class="escala-nombre"></span><div class="escala-pista"><div class="escala-relleno"></div></div><span class="escala-valor">0 %</span>';
  raiz.querySelector('.escala-nombre').textContent = etiqueta;
  return {
    raiz,
    relleno: raiz.querySelector('.escala-relleno'),
    valor: raiz.querySelector('.escala-valor'),
  };
}

// Convierte las variables CSS de color en valores para los lienzos (solo al cambiar de tema)
function resolverColores() {
  const estilos = getComputedStyle(document.documentElement);
  estado.rgb = estado.colores.map((color) => {
    const variable = color.slice(4, -1);
    return hexARgb(estilos.getPropertyValue(variable) || '#888888');
  });
  estado.tintas = {
    linea: estilos.getPropertyValue('--linea').trim(),
    texto: estilos.getPropertyValue('--niebla').trim(),
    tinta: estilos.getPropertyValue('--tinta').trim(),
    fondo: estilos.getPropertyValue('--pagina').trim(),
    fuente: estilos.getPropertyValue('--fuente').trim(),
  };
}

function colorDeEtiqueta(etiqueta) {
  const i = estado.clasificador ? estado.clasificador.etiquetas.indexOf(etiqueta) : -1;
  return i >= 0 ? estado.colores[i] : 'var(--serie-fondo)';
}

function mensajePorDefecto(etiqueta) {
  const nombre = normalizar(etiqueta);
  if (esFondo(etiqueta)) return 'El modelo no reconoce ningún objeto.';
  if (['smartphone', 'movil', 'telefono', 'telefono movil'].includes(nombre)) return 'El modelo reconoce un smartphone.';
  if (nombre === 'cargador') return 'El modelo reconoce un cargador.';
  return `El modelo reconoce: ${etiqueta}.`;
}

function mensajePara(etiqueta) {
  const propio = estado.prefs.mensajes[etiqueta];
  return propio && propio.trim() ? propio.trim() : mensajePorDefecto(etiqueta);
}

function rellenarExplicacion() {
  const c = estado.clasificador;
  if (!c) return;
  document.querySelectorAll('[data-dato="entrada"]').forEach((el) => (el.textContent = `${c.tamano} × ${c.tamano}`));
  if (c.rasgos) document.querySelectorAll('[data-dato="rasgos"]').forEach((el) => (el.textContent = formatoEntero.format(c.rasgos)));
}

function objetosDelModelo() {
  const objetos = estado.clasificador.etiquetas.filter((e) => !esFondo(e)).map((e) => e.toLowerCase());
  if (objetos.length <= 1) return objetos[0] || 'un objeto';
  return `${objetos.slice(0, -1).join(', ')} o ${objetos[objetos.length - 1]}`;
}

/* =========================================================================
   8. Estados del visor y de la barra
   ========================================================================= */

function ponerEstadoModelo(estadoModelo, texto) {
  ui.estadoModelo.dataset.estado = estadoModelo;
  ui.estadoModelo.textContent = texto;
  ui.estadoModelo.setAttribute('aria-label', texto);
}

function mostrarAvisoVisor({ titulo, texto = '', boton = null, progreso = null, estadoVisor = 'aviso' }) {
  ui.visor.dataset.estado = estadoVisor;
  ui.avisoTitulo.textContent = titulo;
  ui.avisoTexto.textContent = texto;
  ui.avisoTexto.hidden = !texto;
  ui.progreso.hidden = progreso === null;
  if (progreso !== null) ui.progreso.style.setProperty('--progreso', progreso);
  ui.avisoBoton.hidden = !boton;
  if (boton) {
    ui.avisoBoton.textContent = boton.texto;
    ui.avisoBoton.onclick = boton.accion;
  }
  ui.visorEtiqueta.hidden = true;
  ui.leyenda.hidden = true;
  ui.mapa.dataset.visible = 'no';
}

function mostrarVisorInactivo() {
  if (!estado.clasificador) return;
  if (estado.modo === 'camara') {
    mostrarAvisoVisor({
      titulo: 'La cámara está apagada',
      texto: `Pulsa Iniciar cámara y apunta a uno de estos objetos: ${objetosDelModelo()}.`,
      estadoVisor: 'vacio-camara',
    });
  } else {
    mostrarAvisoVisor({
      titulo: 'Ninguna imagen seleccionada',
      texto: matchMedia('(pointer: coarse)').matches
        ? 'Pulsa Elegir imágenes. Puedes seleccionar varias a la vez.'
        : 'Arrastra imágenes aquí, pégalas o pulsa Elegir imágenes.',
      estadoVisor: 'vacio-imagen',
    });
  }
}

function ponerEtiquetaVisor(texto, vivo = false) {
  ui.visorEtiqueta.replaceChildren(Object.assign(document.createElement('span'), { textContent: texto }));
  ui.visorEtiqueta.dataset.vivo = vivo ? 'si' : 'no';
  ui.visorEtiqueta.hidden = false;
}

function actualizarDisponibilidad() {
  const hayModelo = Boolean(estado.clasificador);
  ui.btnIniciar.disabled = !hayModelo || estado.camara.activa;
  ui.btnDetener.disabled = !estado.camara.activa;
  ui.btnRegistrar.disabled = !estado.camara.activa;
  ui.btnElegir.disabled = !hayModelo;
  ui.inputImagenes.disabled = !hayModelo;
  const atencion = hayModelo && estado.clasificador.atencionDisponible;
  ui.interruptorAtencion.disabled = hayModelo && !atencion;
  ui.btnModeloProyecto.hidden = estado.fuente !== 'archivos';
}

/* =========================================================================
   9. Cámara
   ========================================================================= */

async function iniciarCamara() {
  if (!estado.clasificador || estado.camara.activa) return;
  if (estado.modo !== 'camara') cambiarModo('camara');

  if (!window.isSecureContext) return errorCamara('inseguro');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return errorCamara('sin-soporte');

  ui.btnIniciar.disabled = true;
  mostrarAvisoVisor({ titulo: 'Abriendo la cámara', texto: 'Si el navegador lo pide, permite el acceso a la cámara.' });

  const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  const trasera = matchMedia('(pointer: coarse)').matches;
  const video = estado.prefs.camara
    ? { ...base, deviceId: { exact: estado.prefs.camara } }
    : { ...base, facingMode: trasera ? 'environment' : 'user' };

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (error) {
    if (estado.prefs.camara && ['OverconstrainedError', 'NotFoundError'].includes(error.name)) {
      // La cámara guardada ya no existe: se vuelve a la predeterminada
      estado.prefs.camara = '';
      guardarPreferencias();
      ui.btnIniciar.disabled = false;
      return iniciarCamara();
    }
    return errorCamara(error.name);
  }

  if (estado.modo !== 'camara') {
    stream.getTracks().forEach((pista) => pista.stop());
    actualizarDisponibilidad();
    return;
  }

  estado.camara.stream = stream;
  ui.video.srcObject = stream;
  try {
    await ui.video.play();
  } catch (error) {
    // Con muted y playsinline la reproducción automática está permitida
  }

  const pista = stream.getVideoTracks()[0];
  const ajustes = pista.getSettings ? pista.getSettings() : {};
  estado.camara.orientacion = ajustes.facingMode || 'user';
  estado.camara.activa = true;
  pista.addEventListener('ended', () => detenerCamara());

  aplicarEspejo();
  ui.visor.style.setProperty('--recorte', '100%');
  ui.visor.dataset.recortada = 'no';
  ui.visor.dataset.estado = 'camara';
  ponerEtiquetaVisor('En directo', true);
  estado.vivo = { probs: null, mapa: null, t: 0 };
  estado.candidato = { clave: null, desde: 0 };
  estado.ultimaRegistrada = null;
  estado.rendimiento = { ms: 0, fps: 0, ultimo: 0, pintado: 0 };
  actualizarDisponibilidad();
  listarCamaras(ajustes.deviceId);
  programarFotograma();
}

function detenerCamara() {
  const estabaActiva = estado.camara.activa;
  estado.camara.activa = false;
  if (estado.camara.id) {
    if (estado.camara.porFotograma) ui.video.cancelVideoFrameCallback(estado.camara.id);
    else cancelAnimationFrame(estado.camara.id);
    estado.camara.id = 0;
  }
  if (estado.camara.stream) {
    estado.camara.stream.getTracks().forEach((pista) => pista.stop());
    estado.camara.stream = null;
  }
  ui.video.srcObject = null;
  actualizarDisponibilidad();
  if (estabaActiva && estado.modo === 'camara') {
    mostrarVisorInactivo();
    reiniciarLectura('La cámara está detenida. El cronograma conserva los últimos segundos.');
    dibujarGrafica();
  }
}

function errorCamara(tipo) {
  const textos = {
    NotAllowedError: [
      'Sin permiso para usar la cámara',
      'Permite el acceso a la cámara desde la barra de direcciones del navegador y pulsa Reintentar.',
    ],
    SecurityError: ['Sin permiso para usar la cámara', 'El navegador ha bloqueado el acceso a la cámara en esta página.'],
    NotFoundError: ['No hay ninguna cámara', 'Conecta una cámara o analiza una imagen en la pestaña Imagen.'],
    NotReadableError: ['La cámara está ocupada', 'Otra aplicación la está usando. Ciérrala y pulsa Reintentar.'],
    inseguro: ['La cámara necesita una conexión segura', 'Abre la página con HTTPS o desde localhost.'],
    'sin-soporte': ['Este navegador no puede usar la cámara', 'Prueba con una versión actual de Chrome, Edge, Firefox o Safari.'],
  };
  const [titulo, texto] = textos[tipo] || ['No se ha podido abrir la cámara', 'Comprueba que está conectada y pulsa Reintentar.'];
  actualizarDisponibilidad();
  mostrarAvisoVisor({ titulo, texto, boton: { texto: 'Reintentar', accion: iniciarCamara } });
}

function espejoActivo() {
  return estado.prefs.espejo && estado.camara.orientacion !== 'environment';
}

function aplicarEspejo() {
  ui.visor.dataset.espejo = espejoActivo() ? 'si' : 'no';
}

// requestVideoFrameCallback avisa cuando llega un fotograma nuevo; así no se analiza dos veces el mismo
function programarFotograma() {
  if (!estado.camara.activa) return;
  estado.camara.porFotograma = Boolean(ui.video.requestVideoFrameCallback);
  if (estado.camara.porFotograma) estado.camara.id = ui.video.requestVideoFrameCallback(procesarFotograma);
  else estado.camara.id = requestAnimationFrame(procesarFotograma);
}

async function procesarFotograma() {
  if (!estado.camara.activa) return;
  if (!ui.video.videoWidth) {
    programarFotograma();
    return;
  }
  let resultado;
  try {
    resultado = await estado.clasificador.analizar(ui.video, {
      espejo: espejoActivo(),
      atencion: estado.prefs.atencion,
    });
  } catch (error) {
    console.error(error);
    detenerCamara();
    aviso('Se ha producido un error al analizar la imagen de la cámara.');
    return;
  }
  if (!estado.camara.activa) return;
  procesarResultadoCamara(resultado);
  programarFotograma();
}

function procesarResultadoCamara({ probabilidades, mapa, ms }) {
  const ahora = performance.now();
  const anterior = estado.vivo;
  let probs = probabilidades;
  let mapaSuave = mapa;

  // Media móvil exponencial: independiente de los fotogramas por segundo
  if (estado.prefs.suavizado && anterior.probs && anterior.probs.length === probs.length) {
    const dt = ahora - anterior.t;
    const a = 1 - Math.exp(-dt / TAU_SUAVIZADO);
    probs = probs.map((p, i) => anterior.probs[i] + a * (p - anterior.probs[i]));
    if (mapa && anterior.mapa && anterior.mapa.length === mapa.length) {
      const am = 1 - Math.exp(-dt / TAU_MAPA);
      mapaSuave = mapa.map((m, i) => anterior.mapa[i] + am * (m - anterior.mapa[i]));
    }
  }
  estado.vivo = { probs, mapa: mapaSuave, t: ahora };

  const decision = decidir(probs);
  medirRendimiento(ms, ahora);
  pintarLectura(decision, probs);
  pintarMapa(mapaSuave, decision);
  anadirASerie(ahora, probs);
  comprobarEstabilidad(decision);
  anunciar(decision);
}

async function listarCamaras(actual) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  let dispositivos = [];
  try {
    dispositivos = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  } catch (error) {
    return;
  }
  const opciones = (seleccionado) =>
    dispositivos.map((d, i) => {
      const opcion = new Option(d.label || `Cámara ${i + 1}`, d.deviceId);
      opcion.selected = d.deviceId === seleccionado;
      return opcion;
    });
  ui.selCamara.replaceChildren(new Option('Predeterminada', ''), ...opciones(estado.prefs.camara));
  ui.selCamaraRapida.replaceChildren(...opciones(actual));
  ui.camaraRapida.hidden = dispositivos.length < 2;
  ui.ayudaDispositivo.textContent =
    dispositivos.length > 1 ? `Hay ${dispositivos.length} cámaras disponibles.` : 'Solo hay una cámara disponible.';
}

async function cambiarCamara(deviceId) {
  estado.prefs.camara = deviceId;
  guardarPreferencias();
  if (estado.camara.activa) {
    detenerCamara();
    await iniciarCamara();
  }
}

/* =========================================================================
   10. Imágenes
   ========================================================================= */

function cargarImagen(url) {
  return new Promise((resolver, rechazar) => {
    const imagen = new Image();
    imagen.onload = () => resolver(imagen);
    imagen.onerror = () => rechazar(new Error('Imagen no válida'));
    imagen.src = url;
  });
}

async function analizarArchivos(lista) {
  const imagenes = [...lista].filter((a) => a.type.startsWith('image/'));
  if (!imagenes.length) {
    aviso('Ninguno de los archivos es una imagen.');
    return;
  }
  if (!estado.clasificador) {
    aviso('Primero hay que cargar el modelo.');
    return;
  }
  cambiarModo('imagen');

  estado.lote.forEach((elemento) => URL.revokeObjectURL(elemento.url));
  estado.lote = [];
  estado.loteActual = -1;
  pintarLote();

  let fallidas = 0;
  for (const [n, archivo] of imagenes.entries()) {
    estado.progresoLote = imagenes.length > 1 ? `Analizando ${n + 1} de ${imagenes.length}` : '';
    if (estado.progresoLote) ui.loteResumen.textContent = estado.progresoLote;
    const url = URL.createObjectURL(archivo);
    try {
      const imagen = await cargarImagen(url);
      // En las fotos siempre se calcula el mapa: así aparece al instante si se activa después
      const resultado = await estado.clasificador.analizar(imagen, { atencion: true });
      const elemento = {
        nombre: archivo.name || 'Imagen pegada',
        url,
        ancho: imagen.naturalWidth,
        alto: imagen.naturalHeight,
        probs: resultado.probabilidades,
        mapa: resultado.mapa,
        ms: resultado.ms,
        miniatura: crearMiniatura(estado.clasificador.lienzo),
      };
      estado.lote.push(elemento);
      registrar('imagen', { probs: elemento.probs, mapa: elemento.mapa, nombre: elemento.nombre, miniatura: elemento.miniatura });
      if (estado.lote.length === 1) mostrarElementoLote(0);
      pintarLote();
    } catch (error) {
      console.error(error);
      URL.revokeObjectURL(url);
      fallidas += 1;
    }
  }

  estado.progresoLote = '';
  if (fallidas) aviso(fallidas === 1 ? 'Una imagen no se ha podido abrir.' : `${fallidas} imágenes no se han podido abrir.`);
  else if (imagenes.length > 1) aviso(`${imagenes.length} imágenes analizadas`);
  if (!estado.lote.length) mostrarVisorInactivo();
  pintarLote();
}

function mostrarElementoLote(i) {
  const elemento = estado.lote[i];
  if (!elemento) return;
  estado.loteActual = i;
  const proporcion = Math.min(elemento.ancho, elemento.alto) / Math.max(elemento.ancho, elemento.alto);
  ui.imagenVista.src = elemento.url;
  ui.imagenVista.alt = elemento.nombre;
  ui.imagenVista.hidden = false;
  ui.visor.style.setProperty('--recorte', `${proporcion * 100}%`);
  ui.visor.dataset.recortada = proporcion < 0.995 ? 'si' : 'no';
  ui.visor.dataset.estado = 'imagen';
  ponerEtiquetaVisor(elemento.nombre);
  ui.telMs.textContent = `${Math.max(1, Math.round(elemento.ms))} ms`;
  const decision = decidir(elemento.probs);
  pintarLectura(decision, elemento.probs);
  pintarMapa(elemento.mapa, decision);
  ui.loteLista.querySelectorAll('button').forEach((boton, j) => boton.setAttribute('aria-current', String(j === i)));
}

function pintarLote() {
  const varias = estado.lote.length > 1;
  ui.lote.hidden = estado.modo !== 'imagen' || !varias;
  if (!varias) return;
  ui.loteResumen.textContent = estado.progresoLote || `${estado.lote.length} imágenes. Elige una para ver su resultado.`;
  ui.loteLista.replaceChildren(
    ...estado.lote.map((elemento, i) => {
      const decision = decidir(elemento.probs);
      const etiqueta = estado.clasificador.etiquetas[decision.indice];
      const item = document.createElement('li');
      const boton = document.createElement('button');
      boton.type = 'button';
      boton.style.setProperty('--color', decision.aceptado ? estado.colores[decision.indice] : 'transparent');
      boton.setAttribute('aria-current', String(i === estado.loteActual));
      boton.setAttribute(
        'aria-label',
        `${elemento.nombre}: ${decision.aceptado ? etiqueta : 'no concluyente'}, ${porcentajeEntero(decision.confianza)}`
      );
      boton.title = elemento.nombre;
      const miniatura = document.createElement('img');
      miniatura.src = elemento.miniatura;
      miniatura.alt = '';
      boton.append(miniatura);
      boton.addEventListener('click', () => mostrarElementoLote(i));
      item.append(boton);
      return item;
    })
  );
}

function crearMiniatura(origen, lado = LADO_MINIATURA) {
  const lienzo = document.createElement('canvas');
  lienzo.width = lienzo.height = lado;
  lienzo.getContext('2d').drawImage(origen, 0, 0, lado, lado);
  return lienzo.toDataURL('image/jpeg', 0.8);
}

/* =========================================================================
   11. Resultado: veredicto, barras y mapa de atención
   ========================================================================= */

function decidir(probs) {
  let indice = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[indice]) indice = i;
  const confianza = probs[indice];
  // Se compara con el valor redondeado que se muestra, para que pantalla y decisión coincidan
  const aceptado = Math.round(confianza * 1000) / 10 >= estado.prefs.umbral;
  return { indice, confianza, aceptado };
}

function pintarLectura(decision, probs) {
  const etiqueta = estado.clasificador.etiquetas[decision.indice];
  if (decision.aceptado) {
    ponerVeredicto(etiqueta, 'clase');
    ponerTexto(ui.mensaje, mensajePara(etiqueta));
    ui.detalle.hidden = true;
  } else {
    ponerVeredicto('No concluyente', 'estado');
    ponerTexto(ui.mensaje, estado.prefs.mensajeDuda.trim() || MENSAJE_DUDA);
    ponerTexto(
      ui.detalle,
      `Lo más probable es ${etiqueta} (${porcentaje(decision.confianza)}), por debajo del umbral del ${estado.prefs.umbral} %.`
    );
    ui.detalle.hidden = false;
  }
  ui.confianzaCaja.hidden = false;
  ponerTexto(ui.confianza, porcentaje(decision.confianza));

  estado.filas.forEach((fila, i) => {
    fila.relleno.style.setProperty('--p', (probs[i] * 100).toFixed(1));
    ponerTexto(fila.valor, porcentajeEntero(probs[i]));
    const ganadora = decision.aceptado && i === decision.indice ? 'si' : 'no';
    if (fila.raiz.dataset.ganadora !== ganadora) fila.raiz.dataset.ganadora = ganadora;
  });
}

function ponerTexto(elemento, texto) {
  if (elemento.textContent !== texto) elemento.textContent = texto;
}

function ponerVeredicto(texto, tipo) {
  if (ui.veredicto.textContent === texto && ui.veredicto.dataset.tipo === tipo) return;
  ui.veredicto.textContent = texto;
  ui.veredicto.dataset.tipo = tipo;
  ui.veredicto.dataset.anima = 'no';
  void ui.veredicto.offsetWidth; // reinicia la animación
  ui.veredicto.dataset.anima = 'si';
}

function reiniciarLectura(texto) {
  ponerVeredicto('En espera', 'estado');
  ui.mensaje.textContent =
    texto || (estado.modo === 'camara' ? 'Inicia la cámara para ver el resultado.' : 'Elige una imagen para ver el resultado.');
  ui.detalle.hidden = true;
  ui.confianzaCaja.hidden = true;
  estado.filas.forEach((fila) => {
    fila.relleno.style.setProperty('--p', 0);
    fila.valor.textContent = '0 %';
    fila.raiz.dataset.ganadora = 'no';
  });
  ui.mapa.dataset.visible = 'no';
  ui.leyenda.hidden = true;
  estado.vivo = { probs: null, mapa: null, t: 0 };
}

// El veredicto usa el mismo cuerpo para todas las clases: el que hace caber la más larga
function ajustarVeredicto() {
  const caja = ui.veredicto.parentElement;
  const ancho = caja.clientWidth;
  if (!ancho) return;
  if (!estado.medidor) {
    estado.medidor = document.createElement('span');
    estado.medidor.className = 'veredicto';
    estado.medidor.setAttribute('aria-hidden', 'true');
    Object.assign(estado.medidor.style, { position: 'absolute', visibility: 'hidden', left: '-9999px', top: '0', fontSize: '100px' });
    document.body.append(estado.medidor);
  }
  const textos = [
    ...(estado.clasificador ? estado.clasificador.etiquetas.map((e) => [e, 'clase']) : []),
    ['No concluyente', 'estado'],
    ['En espera', 'estado'],
  ];
  let mayor = 1;
  for (const [texto, tipo] of textos) {
    estado.medidor.dataset.tipo = tipo;
    estado.medidor.textContent = texto;
    mayor = Math.max(mayor, estado.medidor.getBoundingClientRect().width);
  }
  const tamano = Math.max(34, Math.min(88, Math.floor((100 * ancho) / mayor)));
  caja.style.setProperty('--tam-veredicto', `${tamano}px`);
}

function pintarMapa(mapa, decision, lienzo = ui.mapa) {
  const c = estado.clasificador;
  const visible = Boolean(mapa && c && c.atencionDisponible && estado.prefs.atencion);
  if (lienzo === ui.mapa) ui.leyenda.hidden = !visible;
  if (!visible) {
    lienzo.dataset.visible = 'no';
    return;
  }
  const color = decision.aceptado ? estado.rgb[decision.indice] : [198, 204, 211];
  dibujarMapa(lienzo, mapa, c.ladoMapa, color);
  lienzo.dataset.visible = 'si';
  if (lienzo === ui.mapa) ui.leyenda.style.setProperty('--color-atencion', `rgb(${color.join(',')})`);
}

// Interpola el mapa de 7 × 7 con Catmull-Rom (primero en horizontal y después en vertical)
// y lo colorea con el color de la clase: la transparencia indica cuánto influye cada zona.
const cubica = (p0, p1, p2, p3, t) =>
  p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));

function muestreo(origen, destino) {
  const indices = new Int16Array(destino * 4);
  const pesos = new Float32Array(destino);
  for (let i = 0; i < destino; i++) {
    const f = ((i + 0.5) / destino) * origen - 0.5;
    const base = Math.floor(f);
    pesos[i] = f - base;
    for (let k = 0; k < 4; k++) indices[i * 4 + k] = Math.min(origen - 1, Math.max(0, base - 1 + k));
  }
  return { indices, pesos };
}

const cacheMapa = new Map();

function dibujarMapa(lienzo, mapa, { alto, ancho }, [r, g, b]) {
  const lado = lienzo.width;
  const clave = `${alto}x${ancho}x${lado}`;
  if (!cacheMapa.has(clave)) {
    cacheMapa.set(clave, { x: muestreo(ancho, lado), y: muestreo(alto, lado), filas: new Float32Array(alto * lado) });
  }
  const { x: mx, y: my, filas } = cacheMapa.get(clave);

  for (let fila = 0; fila < alto; fila++) {
    const desde = fila * ancho;
    for (let x = 0; x < lado; x++) {
      const k = x * 4;
      filas[fila * lado + x] = cubica(
        mapa[desde + mx.indices[k]], mapa[desde + mx.indices[k + 1]],
        mapa[desde + mx.indices[k + 2]], mapa[desde + mx.indices[k + 3]], mx.pesos[x]
      );
    }
  }

  const ctx = lienzo.getContext('2d');
  const imagen = ctx.createImageData(lado, lado);
  const datos = imagen.data;
  for (let y = 0; y < lado; y++) {
    const k = y * 4;
    const f0 = my.indices[k] * lado;
    const f1 = my.indices[k + 1] * lado;
    const f2 = my.indices[k + 2] * lado;
    const f3 = my.indices[k + 3] * lado;
    const t = my.pesos[y];
    for (let x = 0; x < lado; x++) {
      let v = cubica(filas[f0 + x], filas[f1 + x], filas[f2 + x], filas[f3 + x], t);
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      // Transparente por debajo del 15 %, opaco en las zonas que más influyen
      const a = Math.min(1, Math.max(0, (v - 0.15) / 0.85));
      const alfa = a * a * (3 - 2 * a);
      const brillo = v > 0.75 ? ((v - 0.75) / 0.25) * 0.3 : 0;
      const i = (y * lado + x) * 4;
      datos[i] = r + (255 - r) * brillo;
      datos[i + 1] = g + (255 - g) * brillo;
      datos[i + 2] = b + (255 - b) * brillo;
      datos[i + 3] = alfa * 215;
    }
  }
  ctx.putImageData(imagen, 0, 0);
}

function alternarAtencion(valor = !estado.prefs.atencion) {
  estado.prefs.atencion = valor;
  guardarPreferencias();
  sincronizarControles();
  const c = estado.clasificador;
  if (!c) return;
  if (estado.modo === 'imagen' && estado.lote[estado.loteActual]) {
    const elemento = estado.lote[estado.loteActual];
    pintarMapa(elemento.mapa, decidir(elemento.probs));
  } else if (estado.vivo.probs) {
    pintarMapa(valor ? estado.vivo.mapa : null, decidir(estado.vivo.probs));
  }
}

/* =========================================================================
   12. Registro automático, avisos de voz y rendimiento
   ========================================================================= */

// Con la cámara se registra un resultado solo cuando cambia y se mantiene un tiempo
function comprobarEstabilidad(decision) {
  if (!estado.prefs.registroAuto) return;
  const clave = decision.aceptado ? decision.indice : -1;
  const ahora = performance.now();
  if (clave !== estado.candidato.clave) {
    estado.candidato = { clave, desde: ahora };
    return;
  }
  if (ahora - estado.candidato.desde >= estado.prefs.estabilidad && clave !== estado.ultimaRegistrada) {
    estado.ultimaRegistrada = clave;
    registrarFotograma(false);
  }
}

function registrarFotograma(manual = true) {
  if (!estado.camara.activa || !estado.vivo.probs) return;
  registrar('camara', {
    probs: estado.vivo.probs,
    mapa: estado.vivo.mapa,
    miniatura: crearMiniatura(estado.clasificador.lienzo),
  });
  if (manual) aviso('Fotograma registrado');
}

// Para lectores de pantalla: anuncia el resultado cuando se mantiene más de un segundo
function anunciar(decision) {
  const clave = decision.aceptado ? decision.indice : -1;
  const a = estado.anuncio;
  const ahora = performance.now();
  if (clave !== a.clave) {
    a.clave = clave;
    a.desde = ahora;
    return;
  }
  if (ahora - a.desde > 1200 && a.hecho !== clave) {
    a.hecho = clave;
    ui.anuncio.textContent = decision.aceptado
      ? `${estado.clasificador.etiquetas[decision.indice]}, ${Math.round(decision.confianza * 100)} por ciento`
      : 'Resultado no concluyente';
  }
}

function medirRendimiento(ms, ahora) {
  const r = estado.rendimiento;
  r.ms = r.ms ? r.ms * 0.85 + ms * 0.15 : ms;
  if (r.ultimo) {
    const fps = 1000 / Math.max(1, ahora - r.ultimo);
    r.fps = r.fps ? r.fps * 0.85 + fps * 0.15 : fps;
  }
  r.ultimo = ahora;
  if (ahora - r.pintado > 250) {
    r.pintado = ahora;
    ui.telMs.textContent = `${Math.round(r.ms)} ms`;
    ui.telFps.textContent = r.fps ? `${Math.round(r.fps)} fps` : 'Sin datos';
  }
}

/* =========================================================================
   13. Cronograma de los últimos segundos
   ========================================================================= */

function anadirASerie(t, probs) {
  estado.serie.push({ t, p: Float32Array.from(probs) });
  const limite = t - VENTANA_GRAFICA - 1000;
  while (estado.serie.length && estado.serie[0].t < limite) estado.serie.shift();
  programarGrafica();
}

let graficaPendiente = false;
function programarGrafica() {
  if (graficaPendiente) return;
  graficaPendiente = true;
  requestAnimationFrame(() => {
    graficaPendiente = false;
    dibujarGrafica();
  });
}

function geometriaGrafica() {
  const ancho = ui.grafica.clientWidth;
  const alto = ui.grafica.clientHeight;
  return { ancho, alto, izq: 0, der: ancho - 40, arriba: 6, abajo: alto - 22 };
}

function dibujarGrafica() {
  const lienzo = ui.grafica;
  const g = geometriaGrafica();
  if (!g.ancho || !g.alto) return;
  const ppp = Math.min(2, window.devicePixelRatio || 1);
  if (lienzo.width !== Math.round(g.ancho * ppp) || lienzo.height !== Math.round(g.alto * ppp)) {
    lienzo.width = Math.round(g.ancho * ppp);
    lienzo.height = Math.round(g.alto * ppp);
  }
  const ctx = lienzo.getContext('2d');
  ctx.setTransform(ppp, 0, 0, ppp, 0, 0);
  ctx.clearRect(0, 0, g.ancho, g.alto);

  if (!estado.tintas) resolverColores();
  const { linea: colorLinea, texto: colorTexto, tinta: colorTinta, fondo: colorFondo, fuente } = estado.tintas;
  const y = (p) => g.abajo - p * (g.abajo - g.arriba);

  // Rejilla y ejes
  ctx.font = `12px ${fuente}`;
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 1;
  for (const p of [0, 0.5, 1]) {
    ctx.strokeStyle = colorLinea;
    ctx.beginPath();
    ctx.moveTo(g.izq, Math.round(y(p)) + 0.5);
    ctx.lineTo(g.der, Math.round(y(p)) + 0.5);
    ctx.stroke();
    ctx.fillStyle = colorTexto;
    ctx.textAlign = 'right';
    ctx.fillText(`${p * 100} %`, g.ancho, y(p));
  }
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.fillText('hace 10 s', g.izq, g.alto - 4);
  ctx.textAlign = 'right';
  ctx.fillText('ahora', g.der, g.alto - 4);

  // Umbral: línea discontinua, porque marca un límite y no un dato
  ctx.save();
  ctx.strokeStyle = colorTinta;
  ctx.globalAlpha = 0.45;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(g.izq, Math.round(y(estado.prefs.umbral / 100)) + 0.5);
  ctx.lineTo(g.der, Math.round(y(estado.prefs.umbral / 100)) + 0.5);
  ctx.stroke();
  ctx.restore();

  const serie = estado.serie;
  if (!serie.length || !estado.clasificador) return;
  const fin = estado.camara.activa ? performance.now() : serie[serie.length - 1].t;
  const x = (t) => g.der - ((fin - t) / VENTANA_GRAFICA) * (g.der - g.izq);
  const n = estado.clasificador.etiquetas.length;

  ctx.save();
  ctx.beginPath();
  ctx.rect(g.izq, 0, g.der - g.izq, g.alto);
  ctx.clip();
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (let c = 0; c < n; c++) {
    ctx.strokeStyle = `rgb(${estado.rgb[c].join(',')})`;
    ctx.beginPath();
    serie.forEach((punto, i) => {
      if (i === 0) ctx.moveTo(x(punto.t), y(punto.p[c]));
      else ctx.lineTo(x(punto.t), y(punto.p[c]));
    });
    ctx.stroke();
  }
  ctx.restore();

  // Punto final de cada línea con un anillo del color de fondo
  const ultimo = serie[serie.length - 1];
  for (let c = 0; c < n; c++) {
    ctx.beginPath();
    ctx.arc(x(ultimo.t), y(ultimo.p[c]), 4, 0, Math.PI * 2);
    ctx.fillStyle = `rgb(${estado.rgb[c].join(',')})`;
    ctx.strokeStyle = colorFondo;
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }

  // Línea vertical bajo el puntero
  if (estado.hover) {
    const punto = estado.hover;
    const px = x(punto.t);
    ctx.strokeStyle = colorTinta;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(px) + 0.5, g.arriba);
    ctx.lineTo(Math.round(px) + 0.5, g.abajo);
    ctx.stroke();
    ctx.globalAlpha = 1;
    for (let c = 0; c < n; c++) {
      ctx.beginPath();
      ctx.arc(px, y(punto.p[c]), 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${estado.rgb[c].join(',')})`;
      ctx.strokeStyle = colorFondo;
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
    }
  }
}

function moverSobreGrafica(evento) {
  const serie = estado.serie;
  if (!serie.length || !estado.clasificador) return;
  const g = geometriaGrafica();
  const rect = ui.grafica.getBoundingClientRect();
  const px = evento.clientX - rect.left;
  const fin = estado.camara.activa ? performance.now() : serie[serie.length - 1].t;
  const t = fin - ((g.der - px) / (g.der - g.izq)) * VENTANA_GRAFICA;

  let cercano = serie[0];
  for (const punto of serie) if (Math.abs(punto.t - t) < Math.abs(cercano.t - t)) cercano = punto;
  estado.hover = cercano;

  const hace = (fin - cercano.t) / 1000;
  ui.graficaTip.replaceChildren();
  const tiempo = document.createElement('p');
  tiempo.className = 'tip-tiempo';
  tiempo.textContent = hace < 0.05 ? 'Ahora' : `Hace ${formatoDecimal.format(hace)} s`;
  ui.graficaTip.append(tiempo);
  estado.clasificador.etiquetas.forEach((etiqueta, c) => {
    const fila = document.createElement('div');
    fila.className = 'tip-fila';
    fila.style.setProperty('--color', `rgb(${estado.rgb[c].join(',')})`);
    const clave = document.createElement('i');
    const nombre = document.createElement('span');
    nombre.textContent = etiqueta;
    const valor = document.createElement('b');
    valor.textContent = porcentajeEntero(cercano.p[c]);
    fila.append(clave, nombre, valor);
    ui.graficaTip.append(fila);
  });
  // El recuadro se coloca junto a la línea vertical, dentro de la gráfica, y cambia de lado cerca del borde
  const xPunto = g.der - ((fin - cercano.t) / VENTANA_GRAFICA) * (g.der - g.izq);
  ui.graficaTip.hidden = false;
  const anchoTip = ui.graficaTip.offsetWidth;
  const izquierda = xPunto + 14 + anchoTip > g.der ? xPunto - 14 - anchoTip : xPunto + 14;
  ui.graficaTip.style.left = `${Math.max(0, izquierda)}px`;
  programarGrafica();
}

function salirDeGrafica() {
  estado.hover = null;
  ui.graficaTip.hidden = true;
  programarGrafica();
}

/* =========================================================================
   14. Histórico
   ========================================================================= */

function registrar(origen, { probs, mapa, nombre = null, miniatura }) {
  const c = estado.clasificador;
  const decision = decidir(probs);
  const registro = {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    fecha: new Date().toISOString(),
    origen,
    nombre,
    clase: c.etiquetas[decision.indice],
    aceptado: decision.aceptado,
    confianza: Number(decision.confianza.toFixed(4)),
    umbral: estado.prefs.umbral,
    etiquetas: c.etiquetas.slice(),
    probabilidades: Array.from(probs, (p) => Number(p.toFixed(4))),
    mapa: mapa ? { ...c.ladoMapa, valores: Array.from(mapa, (v) => Math.round(v * 100) / 100) } : null,
    miniatura,
  };
  estado.historico.unshift(registro);
  if (estado.historico.length > MAX_HISTORICO) estado.historico.length = MAX_HISTORICO;
  guardarHistorico();
  pintarHistorico();
}

function guardarHistorico() {
  let lista = estado.historico;
  while (!guardarDato('historico', lista) && lista.length > 5) {
    lista = lista.slice(0, Math.floor(lista.length * 0.8));
  }
  if (lista !== estado.historico) {
    estado.historico = lista;
    aviso('El navegador se ha quedado sin espacio. Se han borrado los registros más antiguos.');
  }
}

function cargarHistorico() {
  const datos = leerDato('historico');
  return Array.isArray(datos) ? datos.filter((r) => r && r.id && Array.isArray(r.probabilidades)) : [];
}

function etiquetasDelHistorico() {
  const propias = estado.clasificador ? estado.clasificador.etiquetas : [];
  const extra = estado.historico.filter((r) => r.aceptado).map((r) => r.clase);
  return [...new Set([...propias, ...extra])];
}

function pintarHistorico() {
  const total = estado.historico.length;
  ui.contadorHistorico.textContent = total === 0 ? 'Sin registros' : total === 1 ? '1 registro' : `${total} registros`;
  ui.btnCsv.disabled = ui.btnJson.disabled = ui.btnVaciar.disabled = total === 0;
  ui.resumenHistorico.hidden = total === 0;
  ui.historicoVacio.hidden = total > 0;
  ui.tablaCaja.hidden = total === 0;
  if (!total) {
    estado.filtro = 'todos';
    ui.cuerpoHistorico.replaceChildren();
    ui.filtros.replaceChildren();
    ui.distribucion.replaceChildren();
    return;
  }

  const etiquetas = etiquetasDelHistorico();
  const cuenta = new Map(etiquetas.map((e) => [e, 0]));
  let dudas = 0;
  let suma = 0;
  for (const r of estado.historico) {
    suma += r.confianza;
    if (r.aceptado) cuenta.set(r.clase, (cuenta.get(r.clase) || 0) + 1);
    else dudas += 1;
  }

  // Si el filtro activo se ha quedado vacío, se vuelve a "Todos"
  const valido = estado.filtro === 'todos' || (estado.filtro === 'duda' ? dudas > 0 : cuenta.get(estado.filtro) > 0);
  if (!valido) estado.filtro = 'todos';

  const filtros = [{ clave: 'todos', texto: 'Todos', n: total }];
  for (const e of etiquetas) if (cuenta.get(e)) filtros.push({ clave: e, texto: e, n: cuenta.get(e), color: colorDeEtiqueta(e) });
  if (dudas) filtros.push({ clave: 'duda', texto: 'No concluyente', n: dudas, hueca: true });
  ui.filtros.replaceChildren(
    ...filtros.map((f) => {
      const boton = document.createElement('button');
      boton.type = 'button';
      boton.className = 'filtro';
      boton.setAttribute('aria-pressed', String(estado.filtro === f.clave));
      if (f.color || f.hueca) {
        const muestra = document.createElement('span');
        muestra.className = 'filtro-muestra';
        if (f.color) muestra.style.setProperty('--color', f.color);
        if (f.hueca) muestra.dataset.hueca = 'si';
        boton.append(muestra);
      }
      boton.append(document.createTextNode(f.texto));
      const n = document.createElement('b');
      n.textContent = f.n;
      boton.append(n);
      boton.addEventListener('click', () => {
        estado.filtro = f.clave;
        pintarHistorico();
      });
      return boton;
    })
  );

  // Distribución de resultados: una barra apilada con un hueco de 2 px entre tramos
  const tramos = [
    ...etiquetas.filter((e) => cuenta.get(e)).map((e) => ({ texto: e, n: cuenta.get(e), color: colorDeEtiqueta(e) })),
    ...(dudas ? [{ texto: 'No concluyente', n: dudas, color: 'transparent', hueco: true }] : []),
  ];
  ui.distribucion.replaceChildren(
    ...tramos.map((t) => {
      const tramo = document.createElement('span');
      tramo.style.flexGrow = t.n;
      tramo.style.setProperty('--color', t.color);
      if (t.hueco) tramo.dataset.hueco = 'si';
      tramo.title = `${t.texto}: ${t.n} (${Math.round((t.n / total) * 100)} %)`;
      return tramo;
    })
  );
  ui.distribucion.setAttribute(
    'aria-label',
    `Distribución: ${tramos.map((t) => `${t.texto} ${t.n}`).join(', ')}`
  );
  ui.confianzaMedia.textContent = `Confianza media ${porcentajeEntero(suma / total)}`;

  // Tabla
  const visibles = estado.historico.filter((r) =>
    estado.filtro === 'todos' ? true : estado.filtro === 'duda' ? !r.aceptado : r.aceptado && r.clase === estado.filtro
  );
  ui.cuerpoHistorico.replaceChildren(...visibles.map(crearFilaHistorico));
}

function crearFilaHistorico(r) {
  const fila = document.createElement('tr');
  fila.dataset.id = r.id;

  const celdaImagen = document.createElement('td');
  const imagen = document.createElement('img');
  imagen.src = r.miniatura || '';
  imagen.alt = '';
  imagen.loading = 'lazy';
  celdaImagen.append(imagen);

  const celdaResultado = document.createElement('td');
  const boton = document.createElement('button');
  boton.type = 'button';
  boton.className = 'celda-resultado';
  const muestra = document.createElement('span');
  muestra.className = 'filtro-muestra';
  if (r.aceptado) muestra.style.setProperty('--color', colorDeEtiqueta(r.clase));
  else {
    muestra.dataset.hueca = 'si';
    boton.dataset.duda = 'si';
  }
  boton.append(muestra, document.createTextNode(r.aceptado ? r.clase : 'No concluyente'));
  boton.setAttribute('aria-label', `Ver detalle: ${r.aceptado ? r.clase : 'no concluyente'}, ${porcentajeEntero(r.confianza)}`);
  celdaResultado.append(boton);

  const celdaConfianza = document.createElement('td');
  celdaConfianza.className = 'num';
  const envoltorio = document.createElement('span');
  envoltorio.className = 'celda-confianza';
  const barra = document.createElement('span');
  barra.className = 'mini-barra';
  const relleno = document.createElement('span');
  relleno.style.setProperty('--p', (r.confianza * 100).toFixed(1));
  barra.append(relleno);
  envoltorio.append(barra, document.createTextNode(porcentaje(r.confianza)));
  celdaConfianza.append(envoltorio);

  const celdaOrigen = document.createElement('td');
  celdaOrigen.className = 'col-origen';
  celdaOrigen.textContent = r.origen === 'camara' ? 'Cámara' : r.nombre || 'Imagen';
  celdaOrigen.title = celdaOrigen.textContent;

  const celdaFecha = document.createElement('td');
  celdaFecha.className = 'num';
  celdaFecha.textContent = fechaCorta(r.fecha);

  fila.append(celdaImagen, celdaResultado, celdaConfianza, celdaOrigen, celdaFecha);
  return fila;
}

function abrirDetalle(id) {
  const r = estado.historico.find((registro) => registro.id === id);
  if (!r) return;
  estado.detalle = id;
  ui.tituloDetalle.textContent = r.aceptado ? r.clase : 'No concluyente';
  ui.detalleImagen.src = r.miniatura || '';
  ui.detalleImagen.alt = r.origen === 'camara' ? 'Fotograma de la cámara' : r.nombre || 'Imagen analizada';
  ui.detalleMensaje.textContent = r.aceptado ? mensajePara(r.clase) : estado.prefs.mensajeDuda.trim() || MENSAJE_DUDA;

  const datos = [
    ['Origen', r.origen === 'camara' ? 'Cámara' : r.nombre || 'Imagen'],
    ['Fecha', formatoFechaLarga.format(new Date(r.fecha))],
    ['Confianza', porcentaje(r.confianza)],
    ['Umbral aplicado', `${r.umbral} %`],
  ];
  ui.detalleFicha.replaceChildren(
    ...datos.flatMap(([termino, valor]) => [
      Object.assign(document.createElement('dt'), { textContent: termino }),
      Object.assign(document.createElement('dd'), { textContent: valor }),
    ])
  );

  ui.detalleEscala.replaceChildren();
  ui.detalleEscala.style.setProperty('--umbral', r.umbral);
  const etiquetas = r.etiquetas || [];
  let mejor = 0;
  r.probabilidades.forEach((p, i) => {
    if (p > r.probabilidades[mejor]) mejor = i;
  });
  etiquetas.forEach((etiqueta, i) => {
    const fila = crearFilaEscala(etiqueta, colorDeEtiqueta(etiqueta));
    fila.relleno.style.setProperty('--p', (r.probabilidades[i] * 100).toFixed(1));
    fila.valor.textContent = porcentajeEntero(r.probabilidades[i]);
    fila.raiz.dataset.ganadora = r.aceptado && i === mejor ? 'si' : 'no';
    ui.detalleEscala.append(fila.raiz);
  });

  const hayMapa = Boolean(r.mapa && r.mapa.valores);
  ui.detalleInterruptor.hidden = !hayMapa;
  ui.detalleAtencion.checked = hayMapa && estado.prefs.atencion;
  pintarMapaDetalle(r);
  ui.dlgDetalle.showModal();
}

function pintarMapaDetalle(r) {
  const visible = Boolean(r.mapa && r.mapa.valores && ui.detalleAtencion.checked);
  ui.detalleMapa.dataset.visible = visible ? 'si' : 'no';
  if (!visible) return;
  const i = estado.clasificador ? estado.clasificador.etiquetas.indexOf(r.clase) : -1;
  const color = r.aceptado && i >= 0 ? estado.rgb[i] : [198, 204, 211];
  dibujarMapa(ui.detalleMapa, r.mapa.valores, r.mapa, color);
}

function exportarCsv() {
  const etiquetas = [...new Set(estado.historico.flatMap((r) => r.etiquetas || []))];
  const numero = (v) => formatoDecimal.format(v);
  const celda = (valor) => {
    const texto = String(valor ?? '');
    return /[;"\n]/.test(texto) ? `"${texto.replace(/"/g, '""')}"` : texto;
  };
  const filas = [
    ['fecha', 'hora', 'origen', 'archivo', 'resultado', 'confianza (%)', 'umbral (%)', ...etiquetas.map((e) => `${e} (%)`)],
    ...estado.historico.map((r) => {
      const fecha = new Date(r.fecha);
      return [
        fecha.toLocaleDateString('es-ES'),
        fecha.toLocaleTimeString('es-ES'),
        r.origen === 'camara' ? 'cámara' : 'imagen',
        r.nombre || '',
        r.aceptado ? r.clase : 'no concluyente',
        numero(r.confianza * 100),
        r.umbral,
        ...etiquetas.map((e) => {
          const i = (r.etiquetas || []).indexOf(e);
          return i >= 0 ? numero(r.probabilidades[i] * 100) : '';
        }),
      ];
    }),
  ];
  // El BOM y el punto y coma hacen que Excel en español abra el archivo bien
  const csv = '﻿' + filas.map((fila) => fila.map(celda).join(';')).join('\r\n');
  descargar(`historico-${marcaDeTiempo()}.csv`, csv, 'text/csv;charset=utf-8');
  aviso('Histórico exportado en CSV');
}

function exportarJson() {
  const c = estado.clasificador;
  const datos = {
    exportado: new Date().toISOString(),
    modelo: c ? { nombre: c.metadatos.modelName || null, clases: c.etiquetas, entrada: c.tamano } : null,
    registros: estado.historico.map(({ miniatura, ...resto }) => resto),
  };
  descargar(`historico-${marcaDeTiempo()}.json`, JSON.stringify(datos, null, 2), 'application/json');
  aviso('Histórico exportado en JSON');
}

async function vaciarHistorico() {
  const ok = await confirmar({
    titulo: '¿Vaciar el histórico?',
    texto: `Se borrarán ${estado.historico.length === 1 ? 'el registro guardado' : `los ${estado.historico.length} registros guardados`} en este navegador.`,
    accion: 'Vaciar histórico',
  });
  if (!ok) return;
  estado.historico = [];
  guardarHistorico();
  pintarHistorico();
  aviso('Histórico vaciado');
}

function eliminarRegistro() {
  estado.historico = estado.historico.filter((r) => r.id !== estado.detalle);
  guardarHistorico();
  pintarHistorico();
  ui.dlgDetalle.close();
  aviso('Registro eliminado');
}

/* =========================================================================
   15. Ajustes, tema y ficha del modelo
   ========================================================================= */

function cargarPreferencias() {
  const guardadas = leerDato('preferencias') || {};
  const prefs = { ...PREFERENCIAS_POR_DEFECTO, ...guardadas };
  prefs.mensajes = { ...(guardadas.mensajes || {}) };
  prefs.umbral = Math.min(99, Math.max(50, Number(prefs.umbral) || PREFERENCIAS_POR_DEFECTO.umbral));
  prefs.estabilidad = Math.min(3000, Math.max(300, Number(prefs.estabilidad) || PREFERENCIAS_POR_DEFECTO.estabilidad));
  if (!MOTORES[prefs.motor] && prefs.motor !== 'auto') prefs.motor = 'auto';
  return prefs;
}

function guardarPreferencias() {
  guardarDato('preferencias', estado.prefs);
}

function sincronizarControles() {
  const p = estado.prefs;
  ui.umbral.value = p.umbral;
  ui.valorUmbral.textContent = `${p.umbral} %`;
  ui.escala.style.setProperty('--umbral', p.umbral);
  ui.interruptorAtencion.checked = p.atencion;
  ui.dlgAjustes.querySelectorAll('[data-pref]').forEach((control) => {
    const valor = p[control.dataset.pref];
    if (control.type === 'checkbox') control.checked = Boolean(valor);
    else control.value = valor;
  });
  const salida = ui.dlgAjustes.querySelector('[data-salida="estabilidad"]');
  salida.textContent = `${formatoDecimal.format(p.estabilidad / 1000)} s`;
  ui.selectorTema.querySelectorAll('input').forEach((radio) => (radio.checked = radio.value === p.tema));
  const opcionGpu = ui.selMotor.querySelector('option[value="webgpu"]');
  opcionGpu.disabled = hayWebGPU !== true;
  opcionGpu.textContent = hayWebGPU === false ? 'WebGPU (no disponible en este navegador)' : 'WebGPU (experimental)';
  ui.ayudaMotor.textContent = estado.motor
    ? `En uso: ${MOTORES[estado.motor].nombre}. El automático prueba la tarjeta gráfica y, si no está disponible, WebAssembly.`
    : 'El automático prueba la tarjeta gráfica y, si no está disponible, WebAssembly.';
}

function cambiarUmbral(valor) {
  estado.prefs.umbral = valor;
  guardarPreferencias();
  ui.valorUmbral.textContent = `${valor} %`;
  ui.escala.style.setProperty('--umbral', valor);
  if (estado.modo === 'imagen' && estado.lote[estado.loteActual]) {
    mostrarElementoLote(estado.loteActual);
    pintarLote();
  } else if (estado.vivo.probs) {
    const decision = decidir(estado.vivo.probs);
    pintarLectura(decision, estado.vivo.probs);
    pintarMapa(estado.vivo.mapa, decision);
  }
  dibujarGrafica();
}

function pintarMensajesAjustes() {
  const etiquetas = estado.clasificador ? estado.clasificador.etiquetas : [];
  const campos = etiquetas.map((etiqueta, i) => {
    const campo = document.createElement('label');
    campo.className = 'campo';
    const titulo = document.createElement('span');
    const muestra = document.createElement('i');
    muestra.style.setProperty('--color', estado.colores[i]);
    titulo.append(muestra, document.createTextNode(etiqueta));
    const entrada = document.createElement('input');
    entrada.type = 'text';
    entrada.maxLength = 120;
    entrada.placeholder = mensajePorDefecto(etiqueta);
    entrada.value = estado.prefs.mensajes[etiqueta] || '';
    entrada.addEventListener('input', () => {
      if (entrada.value.trim()) estado.prefs.mensajes[etiqueta] = entrada.value;
      else delete estado.prefs.mensajes[etiqueta];
      guardarPreferencias();
      refrescarLectura();
    });
    campo.append(titulo, entrada);
    return campo;
  });

  const duda = document.createElement('label');
  duda.className = 'campo';
  const tituloDuda = document.createElement('span');
  tituloDuda.textContent = 'Resultado no concluyente';
  const entradaDuda = document.createElement('input');
  entradaDuda.type = 'text';
  entradaDuda.maxLength = 120;
  entradaDuda.placeholder = MENSAJE_DUDA;
  entradaDuda.value = estado.prefs.mensajeDuda;
  entradaDuda.addEventListener('input', () => {
    estado.prefs.mensajeDuda = entradaDuda.value;
    guardarPreferencias();
    refrescarLectura();
  });
  duda.append(tituloDuda, entradaDuda);
  ui.listaMensajes.replaceChildren(...campos, duda);
}

function refrescarLectura() {
  if (estado.modo === 'imagen' && estado.lote[estado.loteActual]) {
    const e = estado.lote[estado.loteActual];
    pintarLectura(decidir(e.probs), e.probs);
  } else if (estado.vivo.probs) {
    pintarLectura(decidir(estado.vivo.probs), estado.vivo.probs);
  }
}

async function cambiarPreferencia(clave, valor) {
  estado.prefs[clave] = valor;
  guardarPreferencias();
  sincronizarControles();
  if (clave === 'atencion') alternarAtencion(valor);
  if (clave === 'espejo') aplicarEspejo();
  if (clave === 'camara') await cambiarCamara(valor);
  if (clave === 'motor') await cambiarMotor();
  if (clave === 'tema') aplicarTema();
}

async function cambiarMotor() {
  if (!estado.clasificador) return;
  const estabaActiva = estado.camara.activa;
  const { artefactos, metadatos } = estado.clasificador;
  const ok = await usarModelo({ artefactos, metadatos }, { fuente: estado.fuente, nombre: estado.nombreFuente });
  sincronizarControles();
  if (ok && (estado.prefs.motor === 'auto' || estado.prefs.motor === estado.motor)) {
    aviso(`Motor de cálculo: ${MOTORES[estado.motor].nombre}`);
  }
  if (ok && estabaActiva) iniciarCamara();
}

function aplicarTema() {
  const raiz = document.documentElement;
  const tema = estado.prefs.tema;
  if (tema === 'claro') raiz.dataset.theme = 'light';
  else if (tema === 'oscuro') raiz.dataset.theme = 'dark';
  else delete raiz.dataset.theme;
  const pagina = getComputedStyle(raiz).getPropertyValue('--pagina').trim();
  document.querySelectorAll('meta[name="theme-color"]').forEach((meta) => {
    if (tema === 'auto') meta.content = meta.media.includes('dark') ? '#111417' : '#eceef1';
    else meta.content = pagina;
  });
  repintarColores();
}

function repintarColores() {
  if (!estado.clasificador) return;
  resolverColores();
  dibujarGrafica();
  refrescarLectura();
  if (estado.modo === 'imagen' && estado.lote[estado.loteActual]) {
    const e = estado.lote[estado.loteActual];
    pintarMapa(e.mapa, decidir(e.probs));
  } else if (estado.vivo.probs) {
    pintarMapa(estado.vivo.mapa, decidir(estado.vivo.probs));
  }
}

async function restablecerAjustes() {
  const tema = estado.prefs.tema;
  const motor = estado.prefs.motor;
  estado.prefs = { ...PREFERENCIAS_POR_DEFECTO, mensajes: {} };
  guardarPreferencias();
  sincronizarControles();
  pintarMensajesAjustes();
  aplicarEspejo();
  if (tema !== 'auto') aplicarTema();
  if (motor !== 'auto') await cambiarMotor();
  refrescarLectura();
  aviso('Ajustes restablecidos');
}

function pintarFicha() {
  const c = estado.clasificador;
  const filas = [];
  if (!c) {
    filas.push(['Estado', 'No hay ningún modelo cargado.']);
  } else {
    const m = c.metadatos;
    const clases = document.createElement('span');
    clases.className = 'ficha-clases';
    c.etiquetas.forEach((etiqueta, i) => {
      const item = document.createElement('span');
      const muestra = document.createElement('i');
      muestra.style.setProperty('--color', estado.colores[i]);
      item.append(muestra, document.createTextNode(etiqueta));
      clases.append(item);
    });
    filas.push(
      ['Origen', estado.fuente === 'proyecto' ? 'Carpeta modelo del proyecto' : `Archivos cargados: ${estado.nombreFuente}`],
      ['Nombre', m.modelName || 'Sin nombre'],
      ['Clases', clases],
      ['Imagen de entrada', `${c.tamano} × ${c.tamano} píxeles, ${c.grises ? 'en escala de grises' : 'en color'}`],
      [
        'Arquitectura',
        c.atencionDisponible
          ? `Extractor convolucional de ${formatoEntero.format(c.rasgos)} rasgos y clasificador denso`
          : 'Red neuronal secuencial',
      ],
      ['Parámetros', formatoEntero.format(c.parametros)],
      ['Tamaño de los pesos', formatoTamano(c.bytesPesos)],
      ['Entrenado', m.timeStamp ? formatoFechaLarga.format(new Date(m.timeStamp)) : 'Sin fecha'],
      [
        'Exportado con',
        [m.tmVersion && `Teachable Machine ${m.tmVersion}`, m.tfjsVersion && `TensorFlow.js ${m.tfjsVersion}`].filter(Boolean).join(', ') ||
          'Sin datos',
      ],
      ['Ejecutado con', `TensorFlow.js ${tf.version ? tf.version.tfjs : VERSION_TF}, motor ${MOTORES[estado.motor].nombre}`],
      ['Carga', `${formatoEntero.format(Math.round(estado.tiempos.carga))} ms, más ${formatoEntero.format(Math.round(estado.tiempos.calentar))} ms de preparación`],
      ['Mapa de atención', c.atencionDisponible ? 'Disponible (Grad-CAM)' : 'No disponible para este modelo']
    );
  }
  ui.fichaModelo.replaceChildren(
    ...filas.flatMap(([termino, valor]) => {
      const dt = document.createElement('dt');
      dt.textContent = termino;
      const dd = document.createElement('dd');
      if (typeof valor === 'string') dd.textContent = valor;
      else dd.append(valor);
      return [dt, dd];
    })
  );
}

/* =========================================================================
   16. Modo, avisos y diálogos
   ========================================================================= */

function cambiarModo(modo) {
  if (estado.modo === modo) return;
  estado.modo = modo;
  ui.selectorModo.querySelectorAll('input').forEach((radio) => (radio.checked = radio.value === modo));
  ui.accionesCamara.hidden = modo !== 'camara';
  ui.accionesImagen.hidden = modo !== 'imagen';
  ui.cronograma.hidden = modo !== 'camara';
  ui.telVelocidad.hidden = modo !== 'camara';
  ui.telMs.textContent = 'Sin datos';
  ui.telFps.textContent = 'Sin datos';
  if (modo === 'imagen') {
    detenerCamara();
    if (estado.lote[estado.loteActual]) mostrarElementoLote(estado.loteActual);
    else {
      mostrarVisorInactivo();
      reiniciarLectura();
    }
  } else {
    ui.imagenVista.hidden = true;
    mostrarVisorInactivo();
    reiniciarLectura();
    dibujarGrafica();
  }
  pintarLote();
}

function aviso(texto) {
  const elemento = document.createElement('p');
  elemento.className = 'aviso';
  elemento.textContent = texto;
  ui.avisos.append(elemento);
  while (ui.avisos.children.length > 3) ui.avisos.firstElementChild.remove();
  setTimeout(() => {
    elemento.dataset.saliendo = 'si';
    setTimeout(() => elemento.remove(), 220);
  }, 2800);
}

function confirmar({ titulo, texto, accion }) {
  return new Promise((resolver) => {
    ui.tituloConfirmar.textContent = titulo;
    ui.textoConfirmar.textContent = texto;
    ui.btnConfirmar.textContent = accion;
    ui.dlgConfirmar.returnValue = '';
    ui.dlgConfirmar.addEventListener('close', () => resolver(ui.dlgConfirmar.returnValue === 'si'), { once: true });
    ui.dlgConfirmar.showModal();
  });
}

/* =========================================================================
   17. Eventos
   ========================================================================= */

function conectarEventos() {
  ui.btnIniciar.addEventListener('click', iniciarCamara);
  ui.btnDetener.addEventListener('click', detenerCamara);
  ui.btnRegistrar.addEventListener('click', () => registrarFotograma(true));
  ui.btnElegir.addEventListener('click', () => ui.inputImagenes.click());
  ui.visor.addEventListener('click', () => {
    if (ui.visor.dataset.estado === 'vacio-imagen') ui.inputImagenes.click();
  });
  ui.inputImagenes.addEventListener('change', () => {
    analizarArchivos(ui.inputImagenes.files);
    ui.inputImagenes.value = '';
  });
  ui.selectorModo.addEventListener('change', (e) => cambiarModo(e.target.value));
  ui.interruptorAtencion.addEventListener('change', () => alternarAtencion(ui.interruptorAtencion.checked));
  ui.selCamaraRapida.addEventListener('change', () => cambiarCamara(ui.selCamaraRapida.value));

  ui.umbral.addEventListener('input', () => cambiarUmbral(Number(ui.umbral.value)));

  ui.grafica.addEventListener('pointermove', moverSobreGrafica);
  ui.grafica.addEventListener('pointerleave', salirDeGrafica);

  ui.btnCsv.addEventListener('click', exportarCsv);
  ui.btnJson.addEventListener('click', exportarJson);
  ui.btnVaciar.addEventListener('click', vaciarHistorico);
  ui.cuerpoHistorico.addEventListener('click', (e) => {
    const fila = e.target.closest('tr');
    if (fila) abrirDetalle(fila.dataset.id);
  });
  ui.detalleAtencion.addEventListener('change', () => {
    const r = estado.historico.find((registro) => registro.id === estado.detalle);
    if (r) pintarMapaDetalle(r);
  });
  ui.btnEliminarRegistro.addEventListener('click', eliminarRegistro);

  ui.btnAjustes.addEventListener('click', () => {
    sincronizarControles();
    ui.dlgAjustes.showModal();
  });
  ui.btnModelo.addEventListener('click', () => {
    pintarFicha();
    ui.dlgModelo.showModal();
  });
  ui.btnCargarModelo.addEventListener('click', () => ui.inputModelo.click());
  ui.inputModelo.addEventListener('change', () => {
    if (ui.inputModelo.files.length) cargarModeloDeArchivos(ui.inputModelo.files);
    ui.inputModelo.value = '';
  });
  ui.btnModeloProyecto.addEventListener('click', () => {
    ui.dlgModelo.close();
    cargarModeloDelProyecto();
  });

  ui.dlgAjustes.addEventListener('change', (e) => {
    const control = e.target.closest('[data-pref]');
    if (control) {
      const valor = control.type === 'checkbox' ? control.checked : control.type === 'range' ? Number(control.value) : control.value;
      cambiarPreferencia(control.dataset.pref, valor);
    }
    if (e.target.name === 'tema') cambiarPreferencia('tema', e.target.value);
  });
  ui.dlgAjustes.addEventListener('input', (e) => {
    if (e.target.dataset.pref === 'estabilidad') {
      estado.prefs.estabilidad = Number(e.target.value);
      guardarPreferencias();
      sincronizarControles();
    }
  });
  ui.btnRestablecerMensajes.addEventListener('click', () => {
    estado.prefs.mensajes = {};
    estado.prefs.mensajeDuda = '';
    guardarPreferencias();
    pintarMensajesAjustes();
    refrescarLectura();
    aviso('Mensajes restablecidos');
  });
  ui.btnRestablecerAjustes.addEventListener('click', restablecerAjustes);

  // Un clic fuera del diálogo lo cierra
  document.querySelectorAll('dialog').forEach((dialogo) =>
    dialogo.addEventListener('click', (e) => {
      if (e.target === dialogo) dialogo.close('cerrar');
    })
  );

  // Teclado
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    if (document.querySelector('dialog[open]')) return;
    if (e.target.closest('input, textarea, select, button, a, [contenteditable="true"]')) return;
    const tecla = e.key.toLowerCase();
    if (tecla === ' ' && estado.modo === 'camara') {
      e.preventDefault();
      if (estado.camara.activa) detenerCamara();
      else iniciarCamara();
    } else if (tecla === 'r') registrarFotograma(true);
    else if (tecla === 'a' && estado.clasificador && estado.clasificador.atencionDisponible) alternarAtencion();
    else if (tecla === 'c') cambiarModo('camara');
    else if (tecla === 'i') cambiarModo('imagen');
    else if (tecla === 'o' && estado.clasificador) ui.inputImagenes.click();
  });

  // Pegar una imagen desde el portapapeles
  window.addEventListener('paste', (e) => {
    const archivos = [...((e.clipboardData && e.clipboardData.files) || [])].filter((a) => a.type.startsWith('image/'));
    if (!archivos.length) return;
    e.preventDefault();
    analizarArchivos(archivos);
  });

  // Arrastrar y soltar en cualquier parte de la página
  let profundidad = 0;
  const tieneArchivos = (e) => Boolean(e.dataTransfer) && [...e.dataTransfer.types].includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!tieneArchivos(e)) return;
    profundidad += 1;
    ui.soltar.hidden = false;
  });
  window.addEventListener('dragleave', (e) => {
    if (!tieneArchivos(e)) return;
    profundidad -= 1;
    if (profundidad <= 0) {
      profundidad = 0;
      ui.soltar.hidden = true;
    }
  });
  window.addEventListener('dragover', (e) => {
    if (tieneArchivos(e)) e.preventDefault();
  });
  window.addEventListener('drop', (e) => {
    if (!tieneArchivos(e)) return;
    e.preventDefault();
    profundidad = 0;
    ui.soltar.hidden = true;
    repartirArchivos(e.dataTransfer.files);
  });

  // Si se cambia de pestaña con la cámara encendida, se detiene
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && estado.camara.activa) detenerCamara();
  });

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (estado.prefs.tema === 'auto') repintarColores();
  });

  new ResizeObserver(() => {
    ajustarVeredicto();
    dibujarGrafica();
  }).observe(ui.lectura);
  if (document.fonts) document.fonts.ready.then(ajustarVeredicto);
}

function repartirArchivos(lista) {
  const archivos = [...lista];
  const deModelo = archivos.filter((a) => /\.(zip|json|bin)$/i.test(a.name));
  const imagenes = archivos.filter((a) => a.type.startsWith('image/'));
  if (deModelo.length) cargarModeloDeArchivos(deModelo);
  else if (imagenes.length) analizarArchivos(imagenes);
  else aviso('Suelta imágenes o los archivos del modelo de Teachable Machine.');
}

/* =========================================================================
   18. Inicio
   ========================================================================= */

async function iniciarAplicacion() {
  referenciarInterfaz();
  estado.prefs = cargarPreferencias();
  estado.historico = cargarHistorico();
  aplicarTema();
  sincronizarControles();
  conectarEventos();
  pintarHistorico();
  ajustarVeredicto();
  dibujarGrafica();

  const mac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  ui.pistaPegar.textContent = matchMedia('(pointer: coarse)').matches
    ? 'Puedes elegir varias a la vez.'
    : `También puedes arrastrarlas o pegarlas con ${mac ? 'Cmd+V' : 'Ctrl+V'}.`;

  if (typeof tf === 'undefined') {
    ponerEstadoModelo('error', 'Sin conexión');
    mostrarAvisoVisor({
      titulo: 'No se ha podido cargar TensorFlow.js',
      texto: 'Comprueba la conexión a internet y recarga la página.',
      boton: { texto: 'Recargar', accion: () => location.reload() },
    });
    return;
  }
  tf.enableProdMode();
  comprobarWebGPU().then(sincronizarControles);
  await cargarModeloDelProyecto();
}

if (document.documentElement.dataset.app === 'clasificador') iniciarAplicacion();
