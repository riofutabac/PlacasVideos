# 🚗 Pipeline ALPR Orientado a Eventos (Ecuador - Vía Pintag)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riofutabac/PlacasVideos/blob/main/GoogleColab_ALPR_Pipeline.ipynb)

Pipeline de reconocimiento automático de matrículas vehiculares (ALPR / LPR) de alto rendimiento, optimizado para grabaciones de videovigilancia en vías de lastre (cámara lateral-frontal con ángulo y vibración).

Diseñado con una **arquitectura basada en eventos** para procesar horas continuas de video con máxima fidelidad, detección multi-carril (`ENTRADA` / `SALIDA`), filtrado de polvo/viento mediante Motion Gate de 3 estados, deduplicación temporal y exportación de auditoría a Excel con fotos incrustadas.

---

## 🌟 Características Principales

- **Arquitectura Basada en Eventos:** No ejecuta OCR cuadro por cuadro innecesariamente. Mantiene trayectorias completas por vehículo (`ByteTrack`) y selecciona únicamente las $M$ mejores tomas de alta calidad (`Top-M Ranking`) para reconocimiento de placa.
- **Soporte Dinámico de Hardware (CPU & CUDA):** Detección automática de aceleradores GPU (`CUDAExecutionProvider` de ONNX Runtime para FastALPR y PyTorch CUDA para YOLOv8). Si no hay GPU disponible, opera con fallback transparente a CPU multihilo.
- **Validación Heurística Ecuatoriana:** Validador no destructivo con reglas de provincias oficiales de Ecuador (ej. Pichincha `P`, Guayas `G`, Tungurahua `T`, etc.), formatos particulares y comerciales (`AAA-1234`), y motocicletas (`AA-123A` / `A-123A`).
- **Control de Falsos Positivos:** `Motion Gate` con amortiguación histérica (`idle_count >= 10`), umbrales de persistencia mínima de 0.6s / 8 cuadros para descartar ramas, sombras y ráfagas de polvo.
- **Auditoría Integral en Excel & SQLite:** Guarda telemetría canónica de cada corrida (`speed_ratio`, tiempos de decodificación, inferencia y OCR) e incrusta recortes del vehículo y la placa en celdas de Excel para revisión humana directa.

---

## 📁 Estructura del Proyecto

```text
PlacasVideos/
├── GoogleColab_ALPR_Pipeline.ipynb  # Notebook listo para ejecutar en Google Colab con GPU T4
├── main.py                          # Punto de entrada para ejecución por lotes
├── requirements.txt                 # Dependencias Python
├── config/
│   └── camera_config.yaml           # Calibración de ROI, zonas de cruce y umbrales de tracking
├── src/
│   ├── pipeline_runner.py           # Orquestador del pipeline, decodificación y scheduling
│   ├── motion_gate.py               # Compuerta de movimiento de 3 estados
│   ├── crossing_logic.py            # Lógica de detección de sentido (ENTRADA / SALIDA)
│   ├── quality_ranker.py            # Selección de mejores cuadros por brillo/nitidez/área
│   ├── ecuador_plate_validator.py   # Validación y corrección de sintaxis de placas
│   ├── deduplicator.py              # Agrupación de eventos repetidos por ventana de tiempo
│   ├── db_manager.py                # Persistencia relacional en SQLite y telemetría
│   ├── excel_exporter.py            # Generador del reporte de auditoría con fotos incrustadas
│   └── timer_profiler.py            # Profiling preciso por etapas del pipeline
├── tools/
│   └── calibrate_roi.py             # Herramienta visual interactiva para ajustar ROIs
├── data/                            # Directorio de SQLite (data/events.sqlite)
├── evidence/                        # Almacén de evidencias fotográficas
│   ├── vehicles/                    # Fotos completas de los vehículos
│   └── plates/                      # Recortes ampliados de las placas
└── reports/                         # Reportes generados en formato Excel (.xlsx)
```

---

## 🚀 Ejecución en Google Colab (Aceleración GPU Gratuita)

Si no cuentas con GPU local o tu servidor tiene restricciones de espacio, puedes procesar tus videos a máxima velocidad (> 8× – 12× realtime) usando Google Colab:

1. Abre [Google Colab](https://colab.research.google.com/).
2. Sube o abre el cuaderno [`GoogleColab_ALPR_Pipeline.ipynb`](file:///Users/desarrollopashq/Documents/GitHub/TestNuevoPlacas/GoogleColab_ALPR_Pipeline.ipynb).
3. Selecciona el entorno de ejecución con GPU:
   - **Entorno de ejecución** $\rightarrow$ **Cambiar tipo de entorno de ejecución** $\rightarrow$ **T4 GPU**.
4. Sube tus videos `.mp4` en el panel de archivos o conéctalos desde tu Google Drive.
5. Ejecuta las celdas en orden. Al finalizar, el notebook descargará automáticamente el reporte `reporte_auditoria.xlsx` con todas las fotos de las placas detectadas.

---

## 💻 Instalación y Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/riofutabac/PlacasVideos.git
cd PlacasVideos
```

### 2. Crear y activar entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline
Coloca tus videos `.mp4` en la raíz del proyecto y corre:
```bash
python main.py
```

Al terminar, encontrarás:
- **Base de datos:** `data/events.sqlite`
- **Fotos de evidencia:** `evidence/vehicles/` y `evidence/plates/`
- **Reporte de auditoría:** `reports/reporte_auditoria.xlsx`

---

## 📊 Métricas de Validación Calibradas

En pruebas sobre los videos de referencia (`*(60).mp4` y `*(61).mp4`):
- **Recall de Eventos:** **100% (8/8 vehículos detectados)**
- **Precisión de Dirección (`ENTRADA` / `SALIDA`):** **100%**
- **Falsos Positivos de Evento:** **0**
- **Deduplicación:** 100% de coherencia en agrupaciones de cruce continuo.
