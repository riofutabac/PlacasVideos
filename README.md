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
├── requirements.txt                 # Dependencias Python (ultralytics>=8.4.0, etc.)
├── yolov8n.onnx                     # Modelo ONNX calibrado para GPU NVIDIA
├── config/
│   └── camera_config.yaml           # Calibración de ROI, zonas de cruce y umbrales de tracking
├── src/
│   ├── pipeline_runner.py           # Orquestador del pipeline, decodificación y scheduling
│   ├── video_decoder.py             # Decodificador desacoplado con backends NVDEC y OpenCV
│   ├── motion_gate.py               # Compuerta de movimiento adaptativa (Luma 0.01ms)
│   ├── crossing_logic.py            # Lógica de detección de sentido (ENTRADA / SALIDA)
│   ├── quality_ranker.py            # Selección de mejores cuadros por brillo/nitidez/área
│   ├── ecuador_plate_validator.py   # Validación y corrección de sintaxis de placas
│   ├── deduplicator.py              # Agrupación de eventos repetidos por ventana de tiempo
│   ├── db_manager.py                # Persistencia relacional en SQLite y telemetría
│   ├── excel_exporter.py            # Generador del reporte de auditoría con fotos incrustadas
│   └── timer_profiler.py            # Profiling preciso por etapas del pipeline
├── benchmarks/
│   ├── ground_truth.json            # Ground truth auditado de referencia (8/8 eventos)
│   ├── baseline_yolov8n.json        # Métricas canónicas de referencia YOLOv8n
│   ├── baseline_yolo26n.json        # Métricas canónicas de referencia YOLO26n
│   ├── evaluate_pipeline.py         # Evaluación automática contra ground_truth.json
│   ├── cross_validate_audit.py      # Cruce automático contra 'Revisión bypass Pintag.xlsx'
│   ├── resumen_ejecutivo.py         # Reporte ejecutivo de tiempos, FPS y costos
│   ├── benchmark_yolo_comparison.py # Comparativa A/B YOLOv8 vs YOLO26 (GPU/CPU)
│   └── debug/                       # Scripts auxiliares de depuración
├── data/                            # Directorio de SQLite (data/events.sqlite)
├── evidence/                        # Almacén de evidencias fotográficas
│   ├── vehicles/                    # Fotos completas de los vehículos
│   └── plates/                      # Recortes ampliados de las placas
└── reports/                         # Reportes generados en formato Excel (.xlsx)
```

---

## 🚀 Ejecución en Google Colab (Aceleración GPU)

1. Abre [Google Colab](https://colab.research.google.com/).
2. Sube o abre el cuaderno [`GoogleColab_ALPR_Pipeline.ipynb`](file:///Users/desarrollopashq/Documents/GitHub/TestNuevoPlacas/GoogleColab_ALPR_Pipeline.ipynb).
3. Selecciona el entorno de ejecución con GPU:
   - **Entorno de ejecución** $\rightarrow$ **Cambiar tipo de entorno de ejecución** $\rightarrow$ **T4 GPU**.
4. Conecta tu Google Drive o coloca los videos `.mp4` en el entorno.
5. Ejecuta las celdas en orden. El pipeline procesa por lotes a **>6.4× Tiempo Real**.

---

## 💻 Instalación y Ejecución Local / Servidor

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

### 4. Opciones de Ejecución (`main.py`)
```bash
# Ejecutar con configuración por defecto (procesa videos en la raíz o en Drive):
python main.py

# Procesar una carpeta específica:
python main.py /ruta/a/videos/

# Procesar solo clips específicos (ej. clips 60 y 61):
python main.py /ruta/a/videos/ --clips 60 61

# Seleccionar modelo detector de vehículos:
python main.py --model yolov8n.onnx   # Recomendado para GPUs NVIDIA (6.07 ms)
python main.py --model yolo26n.onnx   # Recomendado para Servidores CPU (32% más rápido en CPU)

# Forzar decodificador:
python main.py --decoder nvdec    # GPU hardware decode
python main.py --decoder opencv   # CPU fallback
```

---

## 🤖 Selección de Modelos: YOLOv8 vs YOLO26

El pipeline soporta arquitecturas modernas de visión por computador según el hardware de despliegue:

El modelo por defecto es **`yolov8n.onnx`**. Se evaluó `yolo26n.onnx` sobre los clips de
referencia (`*(60).mp4` y `*(61).mp4`) en Tesla T4 y resultó peor en el pipeline completo:

| Característica (medido en T4, 18-sep-2026) | `yolov8n.onnx` (por defecto) | `yolo26n.onnx` |
| :--- | :---: | :---: |
| **Recall de eventos** | **100% (8/8)** 🏆 | 100% (8/8) |
| **Dirección de cruce** | **100% (8/8)** 🏆 | 87.5% (7/8) |
| **Falsos positivos** | **0** 🏆 | 1 |
| **Velocidad del pipeline completo** | **6.46× tiempo real** 🏆 | 4.17× tiempo real |
| **Tamaño de archivo** | 12.4 MB | **9.6 MB** 🏆 |

> `yolo26n` es más rápido que `yolov8n` en inferencia aislada sobre CPU, pero esa ventaja no se
> traduce al pipeline completo y cuesta precisión de dirección. Cualquier modelo se puede probar
> con `--model`, pero solo debe adoptarse si mejora en T4 **y** en CPU sin romper el 8/8.

---

## 📊 Métricas de Validación Calibradas (Honestas)

Medidas empíricas sobre el Ground Truth canónico de referencia (`ground_truth.json`, clips `*(60).mp4` y `*(61).mp4`, evaluado en Tesla T4 e Intel Xeon, Septiembre 2026):
- **Recall de Eventos:** **100% (8/8 vehículos detectados)**
- **Precisión de Dirección (`ENTRADA` / `SALIDA`):** **100% (8/8)**
- **Falsos Positivos de Evento:** **0**
- **Placas Exactas (Exact Match):** **33.3% (2 de 6 legibles)** — reproducible en T4 y en CPU
- **Exactitud de Caracteres Honesta:** **75.6% (31/41 caracteres reales)** — penalizando las placas legibles no leídas (ej. `TAA2204` cuenta como 0/7).
- **Acierto de Primer Carácter:** **50.0% (3/6)**
- **Velocidad Sostenida:** **>6.4× Tiempo Real (160 FPS equivalentes)**
- **Costo Cloud:** Menos de **$0.06 USD por hora de video analizada** en GPU T4.
