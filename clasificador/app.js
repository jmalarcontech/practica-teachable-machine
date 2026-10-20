// Clasificador smartphone / cargador
// Modelo entrenado en Teachable Machine y exportado a TensorFlow.js

const CONFIG = {
  rutaModelo: './modelo/',
  umbralPorDefecto: 80,
  tamanoCamara: 320,
  msEstabilidad: 800, // tiempo que tiene que mantenerse un resultado para guardarlo
  maxHistorico: 50,

  // Mensaje de cada clase. Al buscarlo se ignoran mayúsculas y tildes
  mensajes: {
    smartphone: 'Smartphone detectado',
    cargador: 'Cargador detectado',
    fondo: 'Ningún objeto detectado',
  },
  mensajeDuda: 'Resultado no concluyente',
};

let modelo = null;
let etiquetas = [];
let webcam = null;
let enMarcha = false;
let idAnimacion = null;
let umbral = CONFIG.umbralPorDefecto;
let historico = [];
let candidato = { clase: null, desde: 0 };
let ultimaGuardada = null;
let prediccionesImagen = null;

const $ = (id) => document.getElementById(id);

const elementos = {
  estado: $('estadoModelo'),
  aviso: $('aviso'),
  umbral: $('umbral'),
  valorUmbral: $('valorUmbral'),
  selectCamara: $('selectCamara'),
  contenedorCamara: $('contenedorCamara'),
  btnIniciar: $('btnIniciar'),
  btnParar: $('btnParar'),
  zonaSubida: $('zonaSubida'),
  inputImagen: $('inputImagen'),
  vistaPrevia: $('vistaPrevia'),
  textoSubida: $('textoSubida'),
  tablaHistorico: $('tablaHistorico'),
  cuerpoHistorico: $('cuerpoHistorico'),
  historicoVacio: $('historicoVacio'),
  contadorHistorico: $('contadorHistorico'),
  btnCsv: $('btnCsv'),
  btnBorrar: $('btnBorrar'),
};

const panelCamara = crearPanel('resultadoCamara');
const panelImagen = crearPanel('resultadoImagen');

function crearPanel(id) {
  const raiz = $(id);
  return {
    raiz,
    titulo: raiz.querySelector('.resultado-titulo'),
    detalle: raiz.querySelector('.resultado-detalle'),
    barras: raiz.querySelector('.barras'),
    filas: new Map(),
  };
}

/* ---------- Modelo ---------- */

async function cargarModelo() {
  try {
    modelo = await tmImage.load(CONFIG.rutaModelo + 'model.json', CONFIG.rutaModelo + 'metadata.json');
  } catch (error) {
    console.error(error);
    ponerEstado('error', 'Error al cargar el modelo');

    if (location.protocol === 'file:') {
      mostrarAviso('El modelo no se puede cargar abriendo el archivo directamente. Abre la página con un servidor local, por ejemplo Live Server o python3 -m http.server.');
    } else {
      mostrarAviso('No se encuentra el modelo. Comprueba que model.json, metadata.json y weights.bin están en la carpeta modelo.');
    }
    return;
  }

  etiquetas = modelo.getClassLabels();
  crearBarras(panelCamara);
  crearBarras(panelImagen);

  ponerEstado('ok', `Modelo cargado (${etiquetas.length} clases)`);
  elementos.btnIniciar.disabled = false;
  elementos.inputImagen.disabled = false;
}

/* ---------- Cámara ---------- */

async function iniciarCamara() {
  elementos.btnIniciar.disabled = true;
  ocultarAviso();

  const camara = elementos.selectCamara.value; // 'user' = frontal, 'environment' = trasera

  try {
    // La librería no permite reanudar una webcam detenida, así que se crea una nueva cada vez
    webcam = new tmImage.Webcam(CONFIG.tamanoCamara, CONFIG.tamanoCamara, camara === 'user');
    await webcam.setup({ facingMode: camara });
    webcam.webcam.setAttribute('playsinline', ''); // necesario en iPhone
    webcam.webcam.muted = true;
    await webcam.play();
  } catch (error) {
    console.error(error);
    webcam = null;
    elementos.btnIniciar.disabled = false;
    mostrarAviso('No se ha podido acceder a la cámara. Comprueba que el navegador tiene permiso para usarla.');
    return;
  }

  elementos.contenedorCamara.replaceChildren(webcam.canvas);
  elementos.btnParar.disabled = false;
  elementos.selectCamara.disabled = true;

  enMarcha = true;
  candidato = { clase: null, desde: 0 };
  ultimaGuardada = null;
  idAnimacion = requestAnimationFrame(bucle);
}

function detenerCamara() {
  enMarcha = false;
  cancelAnimationFrame(idAnimacion);

  if (webcam) {
    webcam.stop();
    webcam = null;
  }

  elementos.contenedorCamara.innerHTML = '<span class="visor-vacio">Cámara apagada</span>';
  limpiarPanel(panelCamara, 'Pulsa Iniciar para activar la cámara.');

  elementos.btnIniciar.disabled = false;
  elementos.btnParar.disabled = true;
  elementos.selectCamara.disabled = false;
}

async function bucle() {
  if (!enMarcha) return;

  webcam.update();
  const predicciones = await modelo.predict(webcam.canvas);

  // Puede que se haya pulsado Detener mientras se calculaba la predicción
  if (!enMarcha) return;

  const resultado = mostrarResultado(panelCamara, predicciones);
  guardarSiEstable(resultado);

  idAnimacion = requestAnimationFrame(bucle);
}

/* ---------- Imagen subida ---------- */

function clasificarArchivo(archivo) {
  if (!archivo || !modelo) return;

  if (!archivo.type.startsWith('image/')) {
    mostrarAviso('El archivo seleccionado no es una imagen.');
    return;
  }
  ocultarAviso();

  const imagen = new Image();

  imagen.onload = async () => {
    // Se dibuja en un canvas a tamaño real para que el modelo la recorte y escale bien
    const canvas = document.createElement('canvas');
    canvas.width = imagen.naturalWidth;
    canvas.height = imagen.naturalHeight;
    canvas.getContext('2d').drawImage(imagen, 0, 0);

    if (elementos.vistaPrevia.src) URL.revokeObjectURL(elementos.vistaPrevia.src);
    elementos.vistaPrevia.src = imagen.src;
    elementos.vistaPrevia.hidden = false;
    elementos.textoSubida.hidden = true;

    prediccionesImagen = await modelo.predict(canvas);
    const resultado = mostrarResultado(panelImagen, prediccionesImagen);
    anadirAlHistorico(resultado, 'Imagen', canvas);
  };

  imagen.onerror = () => mostrarAviso('No se ha podido abrir la imagen.');
  imagen.src = URL.createObjectURL(archivo);
}

/* ---------- Resultado ---------- */

function mostrarResultado(panel, predicciones) {
  const mejor = predicciones.reduce((a, b) => (b.probability > a.probability ? b : a));
  const porcentaje = Math.round(mejor.probability * 100);
  const aceptado = porcentaje >= umbral;

  panel.raiz.dataset.estado = aceptado ? 'ok' : 'duda';

  if (aceptado) {
    panel.titulo.textContent = mensajePara(mejor.className);
    panel.detalle.textContent = `Confianza: ${porcentaje} %`;
  } else {
    panel.titulo.textContent = CONFIG.mensajeDuda;
    panel.detalle.textContent = `La opción más probable es ${mejor.className} (${porcentaje} %), por debajo del umbral del ${umbral} %.`;
  }

  for (const p of predicciones) {
    const fila = panel.filas.get(p.className);
    if (!fila) continue;

    const pct = Math.round(p.probability * 100);
    fila.relleno.style.width = pct + '%';
    fila.porcentaje.textContent = pct + ' %';
    fila.raiz.classList.toggle('ganadora', aceptado && p === mejor);
  }

  return {
    clase: aceptado ? mejor.className : 'No concluyente',
    probabilidad: mejor.probability,
    aceptado,
  };
}

function mensajePara(clase) {
  return CONFIG.mensajes[normalizar(clase)] || `Clase detectada: ${clase}`;
}

function normalizar(texto) {
  return texto.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().trim();
}

function crearBarras(panel) {
  panel.barras.replaceChildren();
  panel.filas.clear();

  for (const nombre of etiquetas) {
    const raiz = document.createElement('div');
    raiz.className = 'barra';
    raiz.innerHTML = `
      <span class="barra-nombre"></span>
      <div class="barra-pista"><div class="barra-relleno"></div></div>
      <span class="barra-pct">0 %</span>`;
    raiz.querySelector('.barra-nombre').textContent = nombre;

    panel.barras.append(raiz);
    panel.filas.set(nombre, {
      raiz,
      relleno: raiz.querySelector('.barra-relleno'),
      porcentaje: raiz.querySelector('.barra-pct'),
    });
  }
}

function limpiarPanel(panel, texto) {
  delete panel.raiz.dataset.estado;
  panel.titulo.textContent = 'Sin resultados';
  panel.detalle.textContent = texto;

  for (const fila of panel.filas.values()) {
    fila.relleno.style.width = '0';
    fila.porcentaje.textContent = '0 %';
    fila.raiz.classList.remove('ganadora');
  }
}

/* ---------- Umbral ---------- */

function cambiarUmbral(valor) {
  umbral = valor;
  elementos.umbral.value = valor;
  elementos.valorUmbral.textContent = valor + ' %';
  document.documentElement.style.setProperty('--umbral', valor + '%');
  guardarDato('umbral', valor);

  // La última imagen subida se vuelve a evaluar con el nuevo umbral
  if (prediccionesImagen) mostrarResultado(panelImagen, prediccionesImagen);
}

/* ---------- Histórico ---------- */

// La cámara da muchas predicciones por segundo. Solo se guarda un resultado
// cuando cambia y se mantiene un tiempo, para no llenar la tabla de repetidos.
function guardarSiEstable(resultado) {
  const ahora = performance.now();

  if (resultado.clase !== candidato.clase) {
    candidato = { clase: resultado.clase, desde: ahora };
    return;
  }

  if (ahora - candidato.desde >= CONFIG.msEstabilidad && resultado.clase !== ultimaGuardada) {
    ultimaGuardada = resultado.clase;
    anadirAlHistorico(resultado, 'Cámara', webcam.canvas);
  }
}

function anadirAlHistorico(resultado, origen, fuente) {
  historico.unshift({
    fecha: new Date().toISOString(),
    origen,
    clase: resultado.clase,
    probabilidad: resultado.probabilidad,
    aceptado: resultado.aceptado,
    miniatura: crearMiniatura(fuente),
  });

  historico = historico.slice(0, CONFIG.maxHistorico);
  guardarDato('historico', JSON.stringify(historico));
  pintarHistorico();
}

function pintarHistorico() {
  const filas = historico.map((r) => {
    const fila = document.createElement('tr');
    fila.innerHTML = '<td><img alt=""></td><td class="num"></td><td></td><td></td><td class="num"></td>';

    const [miniatura, hora, origen, clase, confianza] = fila.children;
    if (r.miniatura) miniatura.firstChild.src = r.miniatura;
    hora.textContent = new Date(r.fecha).toLocaleTimeString('es-ES');
    origen.textContent = r.origen;
    clase.textContent = r.clase;
    if (!r.aceptado) clase.className = 'no-concluyente';
    confianza.textContent = Math.round(r.probabilidad * 100) + ' %';

    return fila;
  });

  const total = historico.length;
  elementos.cuerpoHistorico.replaceChildren(...filas);
  elementos.tablaHistorico.hidden = total === 0;
  elementos.historicoVacio.hidden = total > 0;
  elementos.contadorHistorico.textContent = total === 0 ? '' : `${total} ${total === 1 ? 'resultado' : 'resultados'}`;
  elementos.btnCsv.disabled = total === 0;
  elementos.btnBorrar.disabled = total === 0;
}

function vaciarHistorico() {
  if (!confirm('¿Seguro que quieres borrar el histórico?')) return;
  historico = [];
  guardarDato('historico', '[]');
  pintarHistorico();
}

function exportarCsv() {
  const lineas = [['fecha', 'hora', 'origen', 'resultado', 'confianza (%)']];

  for (const r of historico) {
    const fecha = new Date(r.fecha);
    lineas.push([
      fecha.toLocaleDateString('es-ES'),
      fecha.toLocaleTimeString('es-ES'),
      r.origen,
      r.clase,
      Math.round(r.probabilidad * 100),
    ]);
  }

  // El ﻿ inicial hace que Excel muestre bien las tildes
  const csv = '﻿' + lineas.map((l) => l.join(';')).join('\n');
  const enlace = document.createElement('a');
  enlace.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  enlace.download = 'historico.csv';
  enlace.click();
  URL.revokeObjectURL(enlace.href);
}

function crearMiniatura(fuente) {
  const lado = Math.min(fuente.width, fuente.height);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  canvas.getContext('2d').drawImage(
    fuente,
    (fuente.width - lado) / 2, (fuente.height - lado) / 2, lado, lado,
    0, 0, 64, 64
  );
  return canvas.toDataURL('image/jpeg', 0.7);
}

/* ---------- Utilidades ---------- */

function ponerEstado(estado, texto) {
  elementos.estado.dataset.estado = estado;
  elementos.estado.textContent = texto;
}

function mostrarAviso(texto) {
  elementos.aviso.textContent = texto;
  elementos.aviso.hidden = false;
}

function ocultarAviso() {
  elementos.aviso.hidden = true;
}

// localStorage puede fallar en modo privado; en ese caso simplemente no se guarda nada
function guardarDato(clave, valor) {
  try {
    localStorage.setItem('clasificador.' + clave, valor);
  } catch (e) {}
}

function leerDato(clave) {
  try {
    return localStorage.getItem('clasificador.' + clave);
  } catch (e) {
    return null;
  }
}

function cargarDatosGuardados() {
  const umbralGuardado = Number(leerDato('umbral'));
  if (umbralGuardado >= 50 && umbralGuardado <= 99) umbral = umbralGuardado;

  try {
    const datos = JSON.parse(leerDato('historico'));
    historico = Array.isArray(datos) ? datos : [];
  } catch (e) {
    historico = [];
  }
}

/* ---------- Eventos ---------- */

elementos.btnIniciar.addEventListener('click', iniciarCamara);
elementos.btnParar.addEventListener('click', detenerCamara);
elementos.umbral.addEventListener('input', (e) => cambiarUmbral(Number(e.target.value)));
elementos.btnCsv.addEventListener('click', exportarCsv);
elementos.btnBorrar.addEventListener('click', vaciarHistorico);

elementos.inputImagen.addEventListener('change', (e) => {
  clasificarArchivo(e.target.files[0]);
  e.target.value = ''; // para poder volver a elegir la misma foto
});

elementos.zonaSubida.addEventListener('dragover', (e) => {
  e.preventDefault();
  elementos.zonaSubida.classList.add('arrastrando');
});

elementos.zonaSubida.addEventListener('dragleave', () => {
  elementos.zonaSubida.classList.remove('arrastrando');
});

elementos.zonaSubida.addEventListener('drop', (e) => {
  e.preventDefault();
  elementos.zonaSubida.classList.remove('arrastrando');
  clasificarArchivo(e.dataTransfer.files[0]);
});

// Si se cambia de pestaña con la cámara encendida, se detiene
document.addEventListener('visibilitychange', () => {
  if (document.hidden && enMarcha) detenerCamara();
});

/* ---------- Inicio ---------- */

cargarDatosGuardados();
cambiarUmbral(umbral);
pintarHistorico();
cargarModelo();
