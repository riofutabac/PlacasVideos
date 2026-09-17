# PLAN MAESTRO v1.0: Pipeline ALPR Ligero Orientado a Eventos (Vía de Lastre)
> **Fuente Canónica de Verdad del Proyecto (CONGELADO v1.0)**  
> **Directorio:** `TestNuevoPlacas`  
> **Misión:** Maximizar el throughput de procesamiento de video de seguridad en vía de lastre, sujeto a límites estrictos de recall de eventos, precisión de dirección, exactitud ALPR y deduplicación, generando reportes de auditoría en Excel con evidencia visual incrustada y cero renderizado innecesario.

---

## 1. Visión y Diagnóstico del Problema

### El error clásico
Tratar un archivo de 6–7 horas (648.000–756.000 cuadros) como 756.000 problemas independientes ejecutando detección y OCR en cada cuadro a 30 FPS. Esto colapsa CPU/GPU, recalienta máquinas normales, produce cientos de miles de inferencias inútiles sobre escenas vacías y confunde polvo o tráfico exterior.

### El nuevo paradigma: Pipeline Orientado a Eventos
El pipeline no analiza cuadros; **descubre tracks, valida eventos y extrae evidencia selectiva**.

```text
VIDEO
  ↓ (decode secuencial)
CROP RECTANGULAR DE ACCESO (Ahorro masivo de cómputo/FLOPs)
  ↓
CONTROL DE MUESTREO ADAPTATIVO (Configurable: SLEEP / PRE-ACTIVE / ACTIVE)
  ↓
YOLO VEHÍCULOS (car, truck, bus, motorcycle)
  ↓
POLÍGONO DEL CARRIL (Rechazo geométrico por punto de contacto con el suelo)
  ↓
BYTETRACK (Asociación temporal liviana en cámara fija)
  ↓
MÁQUINA DE ESTADOS DE CRUCE (OUTSIDE → APPROACHING → CROSSED → COMMITTED)
  ↓
BUFFER DINÁMICO TOP-M (Conserva los mejores M vehicle frames por nitidez y tamaño)
  ↓ (solo al confirmarse COMMITTED)
DETECTOR DE PLACAS EN BATCH (Sobre los M vehicle crops seleccionados)
  ↓
RANKING ESPECÍFICO DE PLACAS TOP-K (K=2 a 3 crops de placa)
  ↓
FASTPLATEOCR (Modelos CCT especializados, sub-milisegundo)
  ↓
VOTACIÓN TEMPORAL + PRIOR ESTADÍSTICO ECUADOR (ANT) (Auditable, nunca filtro destructivo)
  ↓
DEDUPLICACIÓN EN 2 NIVELES (Intra-clip e Inter-clip con cooldown y verificación visual)
  ↓
PERSISTENCIA CANÓNICA EN SQLITE (Con soporte de reanudación / checkpoint)
  ↓
GENERADOR DESACOPLADO DE EXCEL (.xlsx con fotos incrustadas en celdas)
```

---

## 2. Decisiones Arquitectónicas y Reglas Críticas

### A. Doble ROI: Recorte Físico (Crop) + Polígono Lógico
* **Crop Rectangular Físico:** Se extrae una subregión fija del cuadro completo (ej. 800×700 px sobre 1920×1080). YOLO procesa una imagen mucho menor, ahorrando FLOPs y acelerando la inferencia.
* **Polígono Lógico de Carril:** Dentro del crop, se define el polígono que delimita la vía de lastre. Un vehículo se considera válido si su **punto de contacto inferior central de la caja** $(x_{center}, y_{bottom})$ cae dentro del polígono.
* **Criterio de Rechazo:** Rechaza geométricamente el tráfico cuya referencia de contacto no pertenece al polígono configurado (eliminando tráfico de la carretera vecina y sombras externas).

### B. Motion Gate sin Poder de Veto Destructivo (3 Estados Adaptativos)
El detector de movimiento solo ahorra cómputo cuando no hay tránsito; **nunca tiene permiso para descartar un vehículo por sí solo**.
* **`SLEEP` (Zona inactiva):** Motion gate ultra liviano en miniatura (160×90 px) + **Sentinel Watchdog YOLO a frecuencia baja** (configurable en YAML, ej. 1.0–2.0 FPS). Si un auto pasa rápido y el motion gate falla, el Sentinel lo atrapa.
* **`PRE-ACTIVE`:** Al detectar cambio o detección sentinel → detector sube a frecuencia media (ej. 5.0 FPS).
* **`ACTIVE`:** Al confirmar un vehículo en el polígono → detector sube a frecuencia plena (ej. 8.0–10.0 FPS). Se mantiene activo durante `active_hold_seconds` (ej. 1.5–2.0 s) tras la última detección.

### C. Máquina de Estados de Cruce y Fijación de Timestamps
Para evitar falsas alertas por autos que maniobran, retroceden o tocan la línea sin cruzar:
```text
[OUTSIDE] ──> [APPROACHING] ──> [CROSSED] ──> [COMMITTED]
```
1. **`OUTSIDE`:** El vehículo circula lejos de la línea de control.
2. **`APPROACHING`:** El vehículo entra a la zona de aproximación inmediata.
3. **`CROSSED`:** El segmento de trayectoria cruza la línea virtual. **En este instante exacto se congela el `event_timestamp` principal**, el punto de cruce y la dirección tentativa.
4. **`COMMITTED`:** La trayectoria subsiguiente confirma que el vehículo se alejó hacia el lado de destino (confirma `ENTRADA` o `SALIDA`). Si el auto retrocede inmediatamente antes de consolidar, no se emite el evento.
* **Regla de Unicidad:** Cada tupla `(track_id, line_id)` puede generar **máximo un evento**.

### D. Dos Rankings de Calidad Consecutivos (Vehículo vs Placa)
La calidad del auto no siempre correlaciona con la calidad de la matrícula (reflejos, ángulo, oclusión). Se aplican dos filtros:

1. **Top-M Vehicle Frames (en vuelo durante el track, $M=5$):**
   $$\text{Score}_{\text{vehículo}} = w_1 \cdot \text{LaplacianVariance} + w_2 \cdot \text{ÁreaNormalizada} + w_3 \cdot \text{YOLOConf} - w_4 \cdot \text{BordePenalty}$$
2. **Top-K Plate Crops (tras detección de placa en batch, $K=3$):**
   $$\text{Score}_{\text{placa}} = w_1 \cdot \text{PlateConf} + w_2 \cdot \text{ÁreaPlaca} + w_3 \cdot \text{NitidezPlaca} + w_4 \cdot \text{AspectPlausibility} - w_5 \cdot \text{BordeCrop} - w_6 \cdot \text{Sobreexposición}$$

### E. OCR Selectivo + Prior Estadístico Ecuador (ANT)
* **Inferencia Selectiva:** OCR se ejecuta exclusivamente 2–3 veces por evento confirmado sobre los mejores recortes de placa seleccionados.
* **FastPlateOCR:** Reconocimiento CCT especializado en caracteres de placas.
* **Trazabilidad Inmutable:**
  - `plate_raw`: Cadena exacta leída por el OCR, sin alteraciones.
  - `plate_normalized`: Limpieza sintáctica (espacios, guiones, mayúsculas).
  - `plate_corrected`: Corrección asistida por heurística ANT (desambiguación `O ↔ 0`, `I ↔ 1`, `B ↔ 8`, `S ↔ 5`, `Z ↔ 2`, series provinciales).
  - `plate_correction_reason`: Justificación explícita de cada reemplazo.
* **Estados de Placa (`plate_status`):**
  `OK`, `BAJA_CONFIANZA`, `PLACA_DETECTADA_OCR_FALLIDO`, `PLACA_NO_DETECTADA`, `PLACA_NO_LEGIBLE`, `SIN_PLACA_VISIBLE`.
* **Cero Descarte de Eventos:** Si la placa es ilegible o no existe, el evento de vehículo **se guarda y reporta igualmente** con su foto y timestamp.

### F. Deduplicación en 2 Niveles
1. **Intra-clip:** Control estricto por `track_id + line_id`.
2. **Inter-clip (Cruce de frontera entre videos consecutivos):**
   ```python
   is_duplicate = (
       same_direction
       and (delta_t < cooldown_seconds)
       and (
           exact_high_conf_plate_match
           or (
               levenshtein_distance <= 1
               and ocr_is_uncertain
               and visual_appearance_is_similar  # SSIM / perceptual hash
           )
       )
   )
   ```
   - **Auditoría:** En base de datos no se eliminan silenciosamente; se marca `duplicate_of = event_id_original`. En el reporte Excel final se filtran o marcan adecuadamente.

### G. Jerarquía de Datos Canónica y Persistencia Desacoplada
* **`SQLite` (`events.sqlite`):** Fuente canónica de verdad del sistema. Almacena metadata, eventos, auditoría de duplicados y estado de clips procesados (`processing_runs` con hash y status) para **reanudación automática sin reprocesar**.
* **`JPEGs`:** Archivos físicos de evidencia en disco (`evidence/vehicles/` y `evidence/plates/`).
* **`JSON`:** Formato opcional para intercambio, logs estructurados o debugging.
* **`XLSX` (`auditoria.xlsx`):** Producto derivado generado de forma 100% desacoplada:
  - Lee directamente de SQLite y del directorio de evidencias.
  - Incrusta miniaturas de la foto del vehículo y recorte de placa dentro de las celdas.
  - Permite regenerar, filtrar o ajustar el formato del reporte sin tocar los videos.

### H. Cero Video Renderizado
* Prohibido generar archivos `.mp4` anotados con recuadros dibujados de 6–7 horas. **Elimina el coste significativo de dibujo, copias, codificación e I/O asociado al video anotado; el ahorro exacto se medirá en B1**.
* Prohibido `cv2.imshow` en producción.
* Solo se guardan los recortes `.jpg` de los eventos confirmados.

---

## 3. Modelo de Datos Central

```python
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

@dataclass
class VehicleFrameCandidate:
    score: float
    timestamp: float
    vehicle_crop: np.ndarray
    bbox_in_full_frame: Tuple[int, int, int, int]

@dataclass
class PlateCropCandidate:
    score: float
    timestamp: float
    plate_crop: np.ndarray
    plate_bbox: Tuple[int, int, int, int]
    detection_conf: float

@dataclass
class TrackState:
    track_id: int
    first_timestamp: float
    last_timestamp: float
    vehicle_class: str  # 'car', 'truck', 'bus', 'motorcycle'
    
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    side_history: List[str] = field(default_factory=list)
    last_side: Optional[str] = None
    
    state: str = "OUTSIDE"  # OUTSIDE, APPROACHING, CROSSED, COMMITTED
    direction: Optional[str] = None  # 'ENTRADA', 'SALIDA'
    
    crossing_timestamp: Optional[float] = None
    crossing_point: Optional[Tuple[float, float]] = None
    crossing_line_id: Optional[str] = None
    
    best_vehicle_frames: List[VehicleFrameCandidate] = field(default_factory=list)

@dataclass
class VehicleEvent:
    event_id: str
    
    # Vinculación de Ejecución & Trazabilidad
    processing_run_id: str
    clip_hash: str
    
    # Timestamps desacoplados
    event_timestamp: float           # Momento exacto del cruce
    datetime_str: str                # Formato legible (YYYY-MM-DD HH:MM:SS)
    best_vehicle_timestamp: Optional[float]
    best_plate_timestamp: Optional[float]
    
    direction: str                   # 'ENTRADA' o 'SALIDA'
    vehicle_type: str                # 'car', 'truck', 'bus', 'motorcycle'
    
    # Auditoría de placa
    plate_raw: Optional[str]
    plate_normalized: Optional[str]
    plate_corrected: Optional[str]
    plate_correction_reason: Optional[str]
    plate_status: str                # OK, BAJA_CONFIANZA, PLACA_NO_DETECTADA, etc.
    
    # Confianzas
    confidence_vehicle: float
    confidence_plate: float
    confidence_ocr: float
    confidence_consensus: float
    
    # Rutas a evidencia
    vehicle_crop_path: str
    plate_crop_path: Optional[str]
    
    # Votación y trazabilidad
    ocr_votes: List[Tuple[str, float]]
    track_id: int
    line_id: str
    video_source: str
    
    # Deduplicación auditable
    dedup_key: Optional[str] = None
    duplicate_of: Optional[str] = None  # Si no es None, referencia al evento original
```

### Metadatos de Corrida en SQLite (`processing_runs`)
Para asegurar la comparabilidad entre benchmarks (ej. B6 a 10 FPS vs B6 a 5 FPS):
```sql
CREATE TABLE processing_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    model_versions TEXT NOT NULL,      -- JSON: {detector: "yolov8n", ocr: "cct-xs-v2"}
    started_at TEXT NOT NULL,
    finished_at TEXT,
    total_videos INTEGER,
    total_events INTEGER,
    status TEXT NOT NULL
);
```

---

## 4. Estructura del Proyecto

```text
TestNuevoPlacas/
├── config/
│   └── camera_config.yaml     # Coordenadas Crop, Polígono, Línea, Parámetros Sampling
├── data/
│   └── events.sqlite          # Base de datos canónica
├── evidence/
│   ├── vehicles/              # Fotos de vehículos confirmados (JPEG)
│   └── plates/                # Recortes de placas leídas (JPEG)
├── reports/
│   └── reporte_auditoria.xlsx # Reporte Excel con fotos incrustadas
├── src/
│   ├── config_loader.py       # Carga y validación de esquemas YAML
│   ├── video_decoder.py       # Lectura secuencial robusta
│   ├── motion_gate.py         # Gate adaptativo (SLEEP, PRE-ACTIVE, ACTIVE)
│   ├── vehicle_detector.py    # YOLOv8n/11n/26n con inferencia en ROI rectangular
│   ├── tracker.py             # ByteTrack + TrackState
│   ├── crossing_logic.py      # Máquina de estados (OUTSIDE → COMMITTED)
│   ├── quality_ranker.py      # Filtro Top-M (Vehículo) y Top-K (Placa)
│   ├── plate_detector.py      # Detector de placa en batch
│   ├── ocr_engine.py          # FastPlateOCR + votación + prior ANT Ecuador
│   ├── deduplicator.py        # Deduplicación intra e inter-clip con cooldown
│   ├── db_manager.py          # Persistencia SQLite canónica + control de checkpoints
│   └── excel_exporter.py      # Generador de reportes .xlsx con imágenes incrustadas
├── tools/
│   └── calibrate_roi.py       # Herramienta visual para definir Crop, Polígono y Línea
├── benchmarks/
│   ├── ground_truth.json      # Verdad de referencia manual
│   └── evaluate_pipeline.py   # Script de métricas (recall, exact-match, tiempos)
├── deep-research-report.md    # Investigación teórica de soporte
└── PLAN_MAESTRO.md            # Este documento (Fuente Canónica)
```

---

## 5. Plan de Ejecución por Fases (B0 a B11)

```text
[B0] Baseline, Instrumentación y Ground Truth
  └─ Instrumentar timers por etapa. Establecer dataset de validación manual (ground truth).
[B1] Supresión Total de Video Renderizado
  └─ Desactivar cv2.VideoWriter y cv2.imshow; medir el ahorro exacto en CPU, tiempo y disco.
[B2] Calibración de Doble ROI y Línea de Cruce
  └─ Script visual para fijar Crop rectangular, Polígono de carril y Línea virtual.
[B3] Arquitectura Track-Centric + ByteTrack + Máquina de Estados
  └─ Implementar `TrackState` con transición OUTSIDE → APPROACHING → CROSSED → COMMITTED.
  ────────────────── [PAUSA: VALIDACIÓN PRIMER BENCHMARK (B0→B3)] ──────────────────
[B4] Buffer Continuo Top-M (Calidad de Cuadro de Vehículo)
  └─ Muestreo en vuelo de los M=5 mejores recortes por nitidez y área.
[B5] Plate Batch Detector + FastPlateOCR + Prior Estadístico Ecuador
  └─ Inferencia en batch sobre Top-M, ranking de placas Top-K, votación y desambiguación ANT.
[B6] Sweep de Frecuencia Adaptativa (10 / 8 / 5 / 3 FPS)
  └─ Medir speedup Y recall de eventos + exact-match de placas simultáneamente.
[B7] Motion Gate Adaptativo con Sentinel Watchdog
  └─ Implementar estados SLEEP (sentinel), PRE-ACTIVE y ACTIVE con hold time.
[B8] Deduplicador en Dos Niveles (Intra-clip e Inter-clip con `duplicate_of`)
  └─ Fusión de eventos en frontera de videos mediante tiempo, dirección y apariencia.
[B9] Base de Datos SQLite Canónica + Exportador a Excel con Fotos Incrustadas
  └─ Persistencia de eventos + generación de `.xlsx` con imágenes incrustadas en celdas.
[B10] Profiling de Cuellos de Botella & Optimización de Runtime
  └─ Evaluación en hardware objetivo (Mac CoreML/MPS vs NVIDIA TensorRT/NVDEC según entorno).
[B11] Stress Test de Producción & Resiliencia
  └─ Escenarios críticos obligatorios: dos vehículos simultáneos cruzando muy juntos,
     fronteras de clips, reversa/maniobras, polvo extremo y reanudación tras corte.
```

---

## 6. Métricas de Éxito y Criterios de Aceptación

$$\text{Maximizar: } \text{Speed Ratio} = \frac{\text{Duración total de video}}{\text{Tiempo de procesamiento (Wall Clock)}}$$

**Sujeto a los siguientes criterios de aceptación:**
1. **Objetivo de Producción (Event Recall):** $\ge 98\%$ respecto al ground truth manual.
2. **Criterio de No Regresión:** No perder ningún evento que el baseline detectaba correctamente en el `benchmark_set`.
3. **Objetivo Ideal:** $100\%$ sobre el conjunto de aceptación.
4. **Direction Accuracy:** $\ge 99\%$ (clasificación inequívoca de Entrada vs Salida).
5. **Plate Exact Match:** Superior o igual al baseline previo sobre placas nítidas/legibles.
6. **Rechazo Geométrico:** Filtrado estricto del tráfico de la carretera vecina mediante el polígono de apoyo.
7. **Estabilidad Térmica y de Recursos:** Operación continua sin fugas de memoria (`TrackState` limpio) y apto para computadoras estándar.
