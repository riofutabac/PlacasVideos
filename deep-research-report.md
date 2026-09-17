# Investigación profunda: cómo procesar 6–7 horas de video ALPR/ANPR mucho más rápido que tu pipeline actual

## Resumen ejecutivo y diagnóstico

La conclusión principal de la investigación es que **tu mayor oportunidad de aceleración probablemente no está en encontrar “un OCR más rápido” o “un YOLO más rápido” de forma aislada, sino en cambiar la unidad de trabajo del pipeline**.

Tu pipeline no debería pensar:

> “Tengo 756.000 frames; debo analizar 756.000 frames.”

Debería pensar:

> “Tengo una secuencia de pocos eventos de vehículos; debo gastar cómputo pesado solamente alrededor de esos eventos.”

Para 6–7 horas a 30 FPS existen aproximadamente **648.000–756.000 frames**. Si hoy tardas unas 4 horas, tu throughput global es aproximadamente **1,5×–1,75× realtime**. No hay una razón arquitectónica para ejecutar detección de placa + OCR cientos de veces sobre el mismo vehículo cuando solo necesitas un registro final.

Mi recomendación principal es esta:

```text
clips ordenados cronológicamente
        ↓
decode secuencial eficiente
        ↓
ROI fija + motion gate barato
        ↓
vehículo detector a ~5–10 FPS cuando la zona está activa
        ↓
ByteTrack
        ↓
line/polygon crossing + direction
        ↓
mantener Top-M mejores frames DEL TRACK
        ↓
plate detector solamente sobre esos candidatos
        ↓
mantener Top-K mejores plate crops
        ↓
OCR 2–5 veces por vehículo
        ↓
votación temporal/confidence
        ↓
dedup entre tracks/clips
        ↓
evento + crops, SIN video anotado
```

Este patrón tiene soporte práctico en piezas ya existentes. Frigate, por ejemplo, explícitamente utiliza detección de movimiento de muy bajo coste para decidir **cuándo y dónde** ejecutar detección de objetos, usa multiprocessing y evita procesar indiscriminadamente todos los frames. citeturn26view5 El sample de NVIDIA para LPR ya usa una jerarquía `vehículo → placa → reconocimiento`, y publicó throughput de cientos de FPS agregados en GPUs NVIDIA, demostrando que el problema puede implementarse como un pipeline GPU altamente paralelo y no como un loop Python frame-a-frame. citeturn10view0

Para OCR, `fast-plate-ocr` destaca especialmente: sus benchmarks publicados en RTX 3090, batch 1, reportan aproximadamente **0,323–0,676 ms por crop**, según modelo, equivalentes a aproximadamente **1.480–3.094 placas/s** para el reconocimiento aislado. citeturn5view0turn26view1 Por tanto, en un pipeline bien diseñado, **el OCR de 3 crops por vehículo prácticamente no debería ser el cuello de botella**. El problema aparece cuando se lo llama 50, 100 o 200 veces para el mismo automóvil.

Para tracking, en una cámara fija y una entrada/salida relativamente controlada comenzaría con **ByteTrack**, no con BoT-SORT. ByteTrack está diseñado para asociar detecciones incluso de baja confianza y su implementación oficial permite alimentar directamente detecciones de otro modelo; además tiene rutas documentadas para ONNX Runtime, TensorRT y DeepStream. citeturn9view0turn25view1 BoT-SORT añade ReID y compensación de movimiento de cámara, capacidades valiosas en escenas difíciles pero que son mucho menos necesarias en una cámara fija y añaden complejidad. citeturn14view0turn25view2

Para NVIDIA, la solución de máximo rendimiento en septiembre de 2026 es especialmente interesante porque **DeepStream ya está en 9.1** y NVIDIA consolidó sus anteriores repositorios DeepStream dentro del monorepo oficial. DeepStream 9.1 integra decoding hardware, TensorRT, tracking, GStreamer y Service Maker Python/C++; la versión x86 documentada utiliza Ubuntu 24.04, CUDA 13.2 y TensorRT 10.16.x. citeturn19view0turn26view4

Un detalle importante: el repositorio standalone de TensorRT anuncia actualmente TensorRT 11.X, pero **no actualizaría arbitrariamente un entorno DeepStream 9.1 a TensorRT 11**; DeepStream 9.1 declara explícitamente TensorRT 10.16.x como parte de su matriz soportada. citeturn14view3turn19view0

Mi ranking de las optimizaciones, por probable retorno, es:

| Prioridad | Cambio | Impacto esperado |
|---|---|---|
| Muy alta | OCR solo Top-K por vehículo | Elimina una enorme cantidad de trabajo redundante |
| Muy alta | Detector a 5–10 FPS en vez de 30 FPS | 3×–6× menos inferencias del detector |
| Muy alta | ROI + motion/event gating | Evita inferencia durante largos períodos irrelevantes |
| Muy alta | No generar video anotado | Elimina render, copies y encoding innecesarios |
| Alta | TensorRT FP16 / ONNX Runtime GPU | Reduce latencia por inferencia |
| Alta | Batch de clips independientes | Eleva utilización de GPU |
| Alta | Hardware decoding cuando decode/CPU es cuello | Libera CPU y permite más streams |
| Alta | Un único GPU worker con colas | Evita replicar modelos/VRAM |
| Media | CUDA streams / zero-copy avanzado | Útil después de resolver los cuellos grandes |
| Baja inicialmente | ReID/BoT-SORT complejo | Probablemente innecesario para cámara fija |

**Objetivo que considero realista para la primera versión optimizada:** demostrar **≥8× realtime** con event recall prácticamente igual a tu baseline. Eso convertiría 6 horas en unos 45 minutos y 7 horas en unos 52,5 minutos. En una NVIDIA suficientemente potente, con escenas relativamente vacías, ROI fija y TensorRT, **10×–15× realtime es técnicamente plausible**, pero lo marco como **ESTIMACIÓN**, no como benchmark publicado. La velocidad final dependerá enormemente de resolución, codec, GPU, modelo, porcentaje de tiempo con movimiento y tamaño de la zona.

## Repositorios investigados y qué reutilizaría

Las estrellas que aparecen a continuación son una fotografía de las páginas consultadas el **16 de septiembre de 2026**; no las utilizo como criterio principal de calidad.

| Repositorio / URL | Stars / actividad observada | Stack, GPU, licencia | Evaluación para tu caso |
|---|---:|---|---|
| **FastALPR** `https://github.com/ankandrew/fast-alpr` | 799; proyecto actual consultado | Python; ONNX Runtime; detector por defecto de `open-image-models`; OCR `fast-plate-ocr`; CPU/CUDA/OpenVINO/DirectML/QNN; MIT. citeturn3view0turn26view0 | **Muy útil.** Framework ALPR modular real, no simple notebook. Reutilizar interfaces detector/OCR y modelos. No usarlo ingenuamente en todos los frames. |
| **fast-plate-ocr** `https://github.com/ankandrew/fast-plate-ocr` | 752; release reciente dentro de 2026 | Python/Keras 3 + ONNX Runtime; modelos CCT; CPU/CUDA/TensorRT EP/OpenVINO/etc.; MIT. Permite fine-tuning y export. citeturn5view0turn26view1 | **Mi primera opción de OCR.** Especializado en placas, extremadamente pequeño/rápido y entrenable. |
| **open-image-models** `https://github.com/ankandrew/open-image-models` | 112 | Python/ONNX; detectores YOLOv9 de placas; múltiples providers; MIT. citeturn26view2 | **Muy útil como plate detector inicial.** Ofrece una familia resolución/accuracy que permite escoger trade-off. |
| **PaddleOCR** `https://github.com/PaddlePaddle/PaddleOCR` | 89.654; `pushed_at` 2026-09-16 en snapshot API | Python/Paddle; PP-OCRv5; GPU; Apache-2.0; más de 100 idiomas en el proyecto general. citeturn7view0turn6view1 | **Excelente OCR genérico**, pero para tu pipeline usaría solo recognition sobre plate crops; el pipeline completo de text detection sería trabajo redundante. |
| **EasyOCR** `https://github.com/JaidedAI/EasyOCR` | 30,0k; README muestra 1.7.2 de sep-2024 | Python/PyTorch; CRAFT + CRNN/LSTM/CTC; CUDA; Apache-2.0; 80+ idiomas. citeturn6view2turn25view0 | Utilizable, instalación sencilla; **no sería mi elección para throughput máximo**. No encontré benchmark plate-specific comparable al de FastPlateOCR. |
| **Ultralytics** `https://github.com/ultralytics/ultralytics` | 61.668; push 2026-09-16 | Python; YOLO26/YOLO11/anteriores; ONNX/TensorRT/OpenVINO; tracking integrado; AGPL-3.0/Enterprise. citeturn3view3turn7view2 | **Muy práctico para entrenamiento/export y baseline.** En producción de alto rendimiento exportaría el detector a TensorRT. Revisar licencia si producto cerrado. |
| **ByteTrack** `https://github.com/FoundationVision/ByteTrack` | 6,7k | Python/C++; asociación por detecciones; ONNX/TensorRT/DeepStream deployment; MIT. citeturn9view0turn25view1 | **Mi tracker inicial.** Ligero conceptualmente, excelente para cámara fija; puedes pasarle detecciones de cualquier detector. |
| **BoT-SORT** `https://github.com/NirAharon/BoT-SORT` | 1,5k; repo pequeño, 26 commits observados | Python; motion + appearance/ReID + camera-motion compensation; MIT. citeturn14view0turn25view2 | Bueno en tracking difícil, pero su repo original es más “research implementation” y hasta marca deployment code como pendiente. Para este caso, probablemente overkill. |
| **BoxMOT** `https://github.com/mikel-brostrom/boxmot` | 8,3k; 3.841 commits observados | Python + trackers nativos C++; ByteTrack, BoT-SORT, OCSORT, DeepOCSORT, etc.; AGPL-3.0. citeturn14view1turn25view3 | **Excelente para experimentar** y comparar trackers con una API común; incluso ofrece backends C++. Muy útil en fase de benchmark. |
| **NVIDIA DeepStream** `https://github.com/NVIDIA/DeepStream` | 246; oficialmente “Maintained”; DeepStream 9.1 | C/C++ + Python Service Maker; GStreamer, NVDEC, TensorRT, trackers, multi-stream; Apache-2.0 para source + licencia NVIDIA para binarios. citeturn19view0turn26view4 | **La mejor base de máximo throughput NVIDIA.** Producción, no demo. Mayor complejidad de integración. |
| **deepstream_lpr_app** `https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app` | 233; repo legacy trasladado | C/C++; TrafficCamNet → LPDNet → LPRNet; TensorRT/DeepStream; MIT. citeturn10view0 | **Referencia arquitectónica muy valiosa**, pero no comenzaría un proyecto nuevo desde este repo antiguo. Reutilizar arquitectura/configuración, no congelarse en sus modelos. |
| **deepstream_tao_apps** `https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps` | 456; oficialmente dejó de actualizarse | C/C++/Python; TAO LPDNet/LPRNet/OCRNet; MIT. El README dice que fue consolidado en `NVIDIA/DeepStream`. citeturn18view1 | Útil para estudiar parsers/configs antiguos; para código nuevo usar el monorepo DeepStream 9.1. |
| **DeepStream-Yolo** `https://github.com/marcoslucianops/DeepStream-Yolo` | 2,1k | C++/CUDA/configs; YOLOv5→YOLO26, RT-DETR y otros; TensorRT; MIT. citeturn25view4turn26view3 | Muy útil como referencia de integración YOLO→DeepStream. **Advertencia:** README todavía centra compatibilidad en DS 8.0, mientras NVIDIA ya va por 9.1. |
| **ONNX Runtime** `https://github.com/microsoft/onnxruntime` | 21,9k; 15.462 commits observados | C++ core + Python y otros; CUDA/TensorRT EP; MIT. citeturn14view2turn25view5 | **Opción B ideal.** Más sencilla que TensorRT directo y permite cambiar Execution Provider. Usar I/O Binding para reducir transfers CPU↔GPU. citeturn2search2 |
| **TensorRT** `https://github.com/NVIDIA/TensorRT` | 13,4k; repo anuncia TensorRT 11.X actual | C++/CUDA/Python; ONNX parser; Apache-2.0 OSS components. citeturn14view3turn25view6 | **Máximo rendimiento NVIDIA para inferencia.** Más tuning/compatibilidad. En DeepStream respetar la versión TensorRT que DS soporta. |
| **Frigate** `https://github.com/blakeblackshear/frigate` | 35,9k; branch `dev`, copyright 2026 | Python y stack NVR; OpenCV/TensorFlow/aceleradores; motion gating; multiprocessing; MIT. citeturn16view1turn26view5 | No es un ALPR offline para reutilizar entero; **sí es probablemente la mejor evidencia open-source del patrón “motion first, expensive detector only where necessary”**. |
| **Supervision** `https://github.com/roboflow/supervision` | 50,5k; 5.137 commits observados | Python; model-agnostic; zonas, tracking y utilidades; MIT. citeturn16view2turn25view7 | Muy cómodo para prototipar `LineZone`, polygon filtering, ByteTrack y eventos. En versión extrema podría reemplazar esas partes por código mínimo propio. |

**Qué reutilizaría directamente.** De FastALPR reutilizaría la abstracción detector/OCR y, potencialmente, sus modelos por defecto. El proyecto permite cambiar ambos componentes, incluyendo OCR personalizado, lo cual encaja muy bien con un pipeline donde el control temporal vive fuera de FastALPR. citeturn3view0turn26view0

De `open-image-models` probaría al menos dos tamaños de plate detector. El benchmark publicado de validación muestra este trade-off:

| Plate detector | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| YOLOv9-s 608 | 0,957 | **0,917** | 0,966 | 0,772 |
| YOLOv9-t 640 | 0,966 | 0,896 | 0,958 | 0,758 |
| YOLOv9-t 512 | 0,955 | 0,901 | 0,948 | 0,724 |
| YOLOv9-t 416 | 0,940 | 0,894 | 0,940 | 0,702 |
| YOLOv9-t 384 | 0,942 | 0,863 | 0,920 | 0,687 |
| YOLOv9-t 256 | 0,937 | 0,797 | 0,858 | 0,606 |

Estos son **BENCHMARKS PUBLICADOS de accuracy**, no de velocidad. citeturn26view2

Dado que tu prioridad número uno es no perder vehículos/placas, **no elegiría automáticamente el modelo 256 o 384 solo porque sea pequeño**. Si ejecutamos plate detection únicamente 3–10 veces por track, podemos permitirnos 512 o 608 y comprar recall con muy poco coste global.

ByteTrack también merece reutilización directa. Su API permite introducir el array de detecciones de otro detector directamente en `BYTETracker`, por lo que no estás obligado a usar YOLOX. citeturn9view0turn25view1

BoxMOT es especialmente interesante para la fase experimental porque te permite intercambiar trackers sin reescribir todo el pipeline y ya ofrece implementaciones nativas C++ en algunos caminos. citeturn14view1 Pero revisaría cuidadosamente AGPL-3.0 antes de incorporarlo a un producto propietario.

## Tecnologías, benchmarks y el cuello crítico de decoding

**Comparación práctica de componentes**

| Tecnología | Para qué la usaría | Ventaja principal | Desventaja / observación |
|---|---|---|---|
| YOLO26n/s | vehicle detector | Actual generación Ultralytics; export TensorRT/ONNX | No asumir que por ser nuevo gana en tu cámara: validar recall |
| YOLO11n/s | vehicle detector | Maduro, documentado para producción | Algo anterior a YOLO26 |
| YOLOv8 | baseline existente | Ecosistema muy conocido | En 2026 ya no sería mi default para proyecto nuevo |
| ByteTrack | tracking | Simple, barato, buen fit cámara fija | Sin ReID fuerte |
| BoT-SORT | tracking difícil | ReID + camera motion compensation | Más coste y complejidad |
| FastPlateOCR | plate recognition | Especializado y muy rápido | Necesitas verificar/fine-tunear Ecuador |
| PaddleOCR | fallback OCR | Gran ecosistema y multilingüe | Genérico; pipeline completo es más pesado de lo necesario |
| EasyOCR | fallback/prototipo | Muy fácil de usar | Sin benchmark plate-specific competitivo encontrado |
| ONNX Runtime CUDA | runtime B | Fácil despliegue y portable | Puede dejar performance sobre la mesa vs TRT puro |
| ONNX Runtime TensorRT EP | runtime B+ | TensorRT con interfaz ORT | Hay que controlar compatibilidad y fallbacks |
| TensorRT | inference NVIDIA | Máximo control/rendimiento | Conversión/tuning/versionado |
| FFmpeg | decode/preprocessing | Muy robusto, rápido, fácil de perfilar | Pipe a NumPy introduce copias |
| PyAV | decode Python | API directa a libav | Principalmente CPU y más cuidado con threading |
| GStreamer | streaming pipeline | Excelente concurrencia y plugins GPU | Curva de aprendizaje |
| NVDEC | hardware decoding | Descarga video decode a hardware dedicado | Solo ayuda mucho si decode/CPU/copies son relevantes |
| DeepStream | pipeline integral | NVDEC + batching + TensorRT + tracker en GPU | Mayor complejidad |
| OpenCV VideoCapture | baseline | Sencillez | Fácil terminar con decoding/copies bloqueantes y poco control |
| OpenCV CUDA | operaciones específicas | Útil para preprocessing GPU | La disponibilidad de codecs/funciones depende del build |

Las docs actuales de Ultralytics muestran YOLO26 como generación actual y mantienen YOLO11 entre las opciones recomendadas para producción; el framework exporta a ONNX, TensorRT y otros runtimes y soporta varios trackers. citeturn3view3

**Benchmarks publicados encontrados**

| Componente | BENCHMARK PUBLICADO | Hardware / contexto | Qué significa realmente |
|---|---:|---|---|
| FastPlateOCR `cct-xs-v1-global` | **0,3232 ms / 3.094 placas/s** | RTX 3090, batch=1 | OCR aislado; no incluye plate detector |
| FastPlateOCR `cct-xs-v2-global` | **0,4664 ms / 2.144 placas/s** | RTX 3090, batch=1 | Muy atractivo como modelo inicial |
| FastPlateOCR `cct-s-v2-global` | **0,6758 ms / 1.480 placas/s** | RTX 3090, batch=1 | Modelo mayor, todavía extremadamente rápido |
| ByteTrack | ~30 FPS citado por autores; MOT17 table 29,6 FPS | V100 / benchmark MOT | No interpretar como tracker-only universal |
| DeepStream LPR legacy | 9,2 FPS | Jetson Nano, 1 stream | End-to-end sample LPR |
| DeepStream LPR legacy | 80,31 FPS total | Jetson NX, 3 streams | Throughput agregado |
| DeepStream LPR legacy | 146,43 FPS total | Jetson Xavier, 5 streams | Throughput agregado |
| DeepStream LPR legacy | 341,65 FPS total | Jetson Orin, 5 streams | Throughput agregado |
| DeepStream LPR legacy | **447,15 FPS total** | NVIDIA T4, 14 streams | Throughput agregado, no single-stream benchmark |

Los números de FastPlateOCR provienen directamente del benchmark del repositorio. citeturn5view0 Los resultados de ByteTrack provienen del repositorio/paper oficial; además del resultado de aproximadamente 30 FPS, el model zoo publica valores para configuraciones completas, por lo que **no deben tratarse como coste puro del algoritmo de asociación**. citeturn9view0 Los resultados DeepStream son los publicados por el sample oficial LPR; corresponden a una implementación/modelos legacy y a múltiples streams, por lo que son evidencia de capacidad de throughput, **no una promesa para tu cámara ni para DeepStream 9.1**. citeturn10view0

Para estos casos específicos:

- **YOLO26/YOLO11 en tu cámara y GPU:** **NO ENCONTRÉ BENCHMARK CONFIABLE directamente transferible a tu pipeline ALPR**. Los benchmarks generales de detector no incluyen tu resolución, ROI, decode, tracking y modelo fine-tuned.
- **PaddleOCR específicamente sobre placas ecuatorianas:** **NO ENCONTRÉ BENCHMARK CONFIABLE comparable.**
- **EasyOCR específicamente sobre placas ecuatorianas:** **NO ENCONTRÉ BENCHMARK CONFIABLE.**
- **OpenCV VideoCapture vs FFmpeg vs PyAV vs NVDEC para tus codecs concretos:** **NO existe un número universal válido**; debe medirse con tus archivos porque codec, GOP, resolución, CPU, GPU y conversiones de pixel format cambian completamente el resultado.

PaddleOCR 3.x mantiene PP-OCRv5 como su generación actual; PP-OCRv5 se presenta como solución multiescenario/multilingüe y existen variantes específicas de reconocimiento. citeturn24search9turn24search33 Para este problema, sin embargo, **no usaría `PaddleOCR(frame completo)`** si ya tenemos un detector de placas: utilizaría únicamente su recognizer sobre los crops.

**Decoding puede ser un cuello de botella, pero primero hay que demostrarlo.**

OpenCV `VideoCapture` no es necesariamente “lento” por definición: puede delegar a FFmpeg/GStreamer y OpenCV documenta opciones de hardware video acceleration. El problema es que abstrae gran parte del pipeline y es fácil acabar decodificando a CPU/BGR y realizando CPU↔GPU copies en cada frame. citeturn13search19

FFmpeg ofrece una baseline mucho más transparente y NVIDIA documenta integración de FFmpeg con sus paths de hardware encode/decode. citeturn13search2 NVDEC es el motor dedicado de NVIDIA para video decoding y NVIDIA publica documentación específica de throughput/capacidades; puede procesar video mucho más rápido que realtime según codec y GPU. citeturn13search14

PyAV envuelve las librerías FFmpeg/libav directamente y permite controlar threading; su documentación describe tanto slice como frame threading. Para procesamiento offline, el mayor buffering/latencia introducido por frame threading suele ser aceptable porque tu objetivo es throughput, no respuesta interactiva. citeturn13search1

Un detalle importante para **frame skipping**: no intentaría implementar “leer frame 0, hacer seek al 6, seek al 12, …” sobre H.264/H.265. Los codecs inter-frame dependen de frames anteriores y PyAV documenta que los seeks deben aterrizar alrededor de keyframes. En general, para archivos comprimidos conviene **decodificar secuencialmente y descartar/samplear**, o utilizar mecanismos específicos del decoder, en lugar de hacer cientos de seeks aleatorios. citeturn13search12

DeepStream lleva esa idea bastante más lejos. Su decoder `Gst-nvvideo4linux2` usa NVDEC para codecs soportados y expone propiedades de dropping/skipping de frames; además, `nvstreammux` permite batch de varias fuentes antes de inferencia. citeturn11search2turn11search4 DeepStream recomienda también aumentar intervalos de inferencia cuando GPU es el cuello y permite medir latencia por componente del pipeline. citeturn11search3

**Cuándo NVDEC realmente te ayudará:**

Si un benchmark “decode-only” demuestra que 6 horas de video tardan, por ejemplo, 10–20 minutos en CPU, reducirlo a 5 minutos no arreglará un pipeline total de 4 horas. En ese escenario, detector/OCR/sincronizaciones son el problema.

Si decode-only tarda 1–2 horas o tu CPU está al 100% mientras la GPU permanece subutilizada, NVDEC puede ser un cambio enorme.

Y con muchos clips concurrentes, hardware decode puede ser aún más importante porque permite que la CPU se concentre en demux, tracking, event logic y filesystem mientras la GPU/video engine decodifica.

Por eso **no migraría a DeepStream antes de obtener un breakdown de tiempos**.

## Arquitecturas recomendadas

**Opción A — modificación mínima**

Esta es la que implementaría primero porque puede darte la mayor parte del ahorro sin reescribir todo:

```text
FFmpeg / PyAV / OpenCV
        │
        ├── decode secuencial
        │
        ├── sample inicial ≈10 FPS
        │
        ▼
ROI fija + motion gate pequeño
        │
        ▼
YOLO26n/s o tu YOLO actual
        │
        ▼
ByteTrack
        │
        ▼
line-crossing / polygon
        │
        ▼
Top-M frames por track
        │
        ▼
plate detector sobre Top-M
        │
        ▼
Top-K plate crops
        │
        ▼
FastPlateOCR
        │
        ▼
temporal vote + cooldown
        │
        ▼
SQLite/Parquet/JSON + JPEG crops
```

Mantendría Python y probablemente tu detector actual al principio. Solo cambiaría **cuándo se llama cada cosa**.

Supervision puede acelerar muchísimo el prototipo porque es model-agnostic y tiene utilidades de tracking/zones; su repositorio incluso muestra ejemplos de vehicle tracking con ByteTrack. citeturn16view2turn25view7 En una segunda optimización puedes reemplazarlo por unas pocas funciones propias si el profiler demuestra overhead.

Para OCR pondría:

```text
FastALPR / open-image-models plate detector
              ↓
       FastPlateOCR
```

en lugar de un OCR genérico sobre cada frame. FastALPR fue diseñado precisamente para separar detector y recognizer y permitir intercambiar componentes. citeturn3view0

Ventajas: bajo riesgo, fácil A/B test, poco cambio de código, conserva debugging Python.

Desventajas: frames probablemente siguen entrando a memoria CPU; menor batching/zero-copy; Python sigue coordinando el pipeline.

**Opción B — alto rendimiento Python + GPU**

Esta sería mi arquitectura objetivo si tienes una NVIDIA de escritorio/datacenter y quieres mantener Python:

```text
2–4 decode workers
FFmpeg NVDEC / GStreamer / PyNvVideoCodec
        │
        ├────────────┐
        ▼            ▼
 sampled frames   next clips prefetched
        │
        ▼
bounded queue
        │
        ▼
ONE GPU inference worker
        │
        ├── dynamic batch across independent clips
        │
        ▼
YOLO TensorRT FP16
        │
        ▼
per-stream ByteTrack state
        │
        ▼
crossing + Top-M candidate buffers
        │
        ▼
batch plate detector
        │
        ▼
batch FastPlateOCR ORT CUDA/TRT EP
        │
        ▼
event aggregation / writer
```

NVIDIA mantiene PyNvVideoCodec como bindings Python sobre APIs C++ para encoding/decoding acelerado por hardware, por lo que es una alternativa relevante a extraer raw BGR mediante un subprocess FFmpeg cuando quieres una ruta Python/NVIDIA más integrada. citeturn17search2

Aquí usaría preferentemente:

- **TensorRT FP16** para el vehicle detector.
- ONNX Runtime CUDA o TensorRT EP para FastPlateOCR/plate detector.
- I/O Binding cuando sea posible para evitar viajes GPU→CPU→GPU innecesarios; ONNX Runtime documenta I/O Binding precisamente para mantener tensors en el dispositivo. citeturn2search2
- Uno o pocos contexts GPU persistentes.
- Batching entre clips porque son independientes.

No haría:

```text
process_1 -> carga YOLO en GPU
process_2 -> carga YOLO en GPU
process_3 -> carga YOLO en GPU
process_4 -> carga YOLO en GPU
```

como estrategia principal en una única GPU.

Eso replica weights, buffers y execution contexts, consume VRAM y puede convertir la GPU en una batalla de contexts.

Preferiría:

```text
decoder CPU/NVDEC workers
        ↓
bounded queue
        ↓
1 GPU inference service
        ↓
batched results
```

Los trackers permanecen independientes por `clip_id`/stream, pero las inferencias pueden ir juntas en un batch.

Las CUDA streams las introduciría **después** de batching y profiling. Pueden permitir solapar transferencias, decode e inferencia, pero añadirlas antes de eliminar OCR redundante, frame redundancy y CPU/GPU copies sería optimización prematura.

**Opción C — máximo rendimiento NVIDIA**

Para throughput extremo:

```text
filesrc
  ↓
demux/parser
  ↓
NVDEC
  ↓
nvstreammux
  ↓
PGIE TensorRT
vehicle detector
  ↓
nvtracker
  ↓
ROI / analytics / line crossing
  ↓
candidate extraction
  ↓
SGIE or custom branch
plate detector
  ↓
selective OCR
  ↓
metadata/event sink
```

DeepStream 9.1 combina precisamente hardware decode, TensorRT, object tracking, multi-stream batching y GStreamer; el repo actual también contiene YOLO integration tools y Service Maker Python/C++. citeturn19view0 NVIDIA consolidó en este repo los antiguos `deepstream_tao_apps`, reference apps y herramientas. citeturn19view0turn18view1

La antigua app NVIDIA LPR es una plantilla especialmente importante porque implementa:

```text
PGIE vehicle
  ↓
SGIE license plate detection
  ↓
SGIE license plate recognition
```

y documenta ROI support y backends TensorRT/Triton. citeturn10view0

No copiaría literalmente su comportamiento de reconocer la placa continuamente. La modificaría para que **el tracker/event state determine cuándo activar la fase de placa/OCR**.

Una variante que considero incluso más atractiva para Ecuador sería:

```text
DeepStream:
NVDEC → vehicle TRT → tracker → line crossing
                              ↓
                       Top-M crops/event
                              ↓
                      async crop queue
                              ↓
Python/C++ OCR worker:
plate detector → FastPlateOCR → vote
```

Así obtienes el throughput de DeepStream para la parte de video, que es la pesada, sin quedar atado al recognizer LPRNet pretrained de un sample que no ha sido validado en tus matrículas.

Complejidad relativa:

| Arquitectura | Velocidad potencial | Complejidad | Flexibilidad OCR | Tiempo de ingeniería |
|---|---|---|---|---|
| A Python optimizado | Alta | Baja | Excelente | Bajo |
| B Python + TRT/NVDEC | Muy alta | Media | Excelente | Medio |
| C DeepStream 9.1 | Máxima | Alta | Media-alta con branch custom | Alto |

**Mi elección:** A inmediatamente → medir → B como solución probable final. Solo saltaría a C si después necesitas exprimir una NVIDIA al máximo, procesar muchos archivos simultáneamente o escalar a múltiples cámaras.

## Frame skipping, ROI, OCR selectivo, paralelismo y deduplicación

A 30 FPS, la cantidad de inferencias potenciales es:

| Detector rate | Frames/inferencias en 6 h | En 7 h | Reducción respecto 30 FPS |
|---:|---:|---:|---:|
| 30 FPS | 648.000 | 756.000 | 1× |
| 15 FPS | 324.000 | 378.000 | 2× |
| 10 FPS | 216.000 | 252.000 | 3× |
| 5 FPS | 108.000 | 126.000 | 6× |
| 2 FPS | 43.200 | 50.400 | 15× |

Esto es una **reducción matemática de llamadas al detector**, no una predicción de speedup global. Decode, tracking, copies y almacenamiento siguen teniendo costes.

Yo empezaría con **10 FPS**.

Después probaría 5 FPS.

No bajaría directamente a 2 FPS porque tu prioridad es “no perder vehículos”. Un vehículo puede cruzar una región pequeña entre dos muestras separadas por 500 ms. El tracker puede interpolar un objeto ya observado; **no puede recuperar mágicamente un vehículo que nunca produjo una detección**.

La regla correcta no es escoger una frecuencia arbitraria sino medir:

```text
dwell_time_min =
    menor tiempo observado que un vehículo real permanece
    dentro de la zona detectable
```

Y escoger aproximadamente:

```text
sample_interval <= dwell_time_min / 3
```

para intentar obtener varias observaciones incluso para el caso rápido.

Para un driveway lento puede resultar que 5 FPS sea más que suficiente. Para una calle rápida puede no serlo.

Además, line crossing no debe exigir que el centro aparezca exactamente encima de la línea. Usa la trayectoria:

```text
previous_centroid -----> current_centroid
               X
         virtual line
```

y verifica intersección/cambio de lado. Así puedes detectar un crossing aunque el objeto “salte” de un lado al otro entre frames muestreados.

**ROI: aquí hay un matiz crítico.**

Esto:

```python
frame[outside_polygon] = 0
prediction = yolo(frame)
```

normalmente **no reduce de forma significativa los FLOPs del detector**, porque la red sigue recibiendo, por ejemplo, un tensor 640×640.

Para ahorrar de verdad:

```text
full frame 1920×1080
      ↓
crop rectangular alrededor del acceso
      ↓
resize a 416/512/etc.
      ↓
detector
```

o, todavía mejor:

```text
cheap ROI motion test
      ↓
NO MOTION ───────────► no detector
MOTION
      ↓
detector
```

La ROI también aumenta la resolución efectiva del vehículo dentro del input. Eso puede permitir bajar el `imgsz` del detector manteniendo tamaño aparente suficiente.

El mejor diseño para preservar recall sería un **motion gate con watchdog**:

```text
zona inactiva:
    motion detector barato
    +
    sentinel YOLO cada 0.5–1 s

si aparece motion o sentinel detection:
    activar modo HIGH-RATE

zona activa:
    YOLO a 5–10 FPS
    tracking
    mantener activo 1–2 s tras último movimiento
```

Esto evita depender al 100% de background subtraction. Frigate es evidencia práctica de la utilidad de hacer detección de movimiento barata antes de object detection cara. citeturn26view5

Para el motion gate usaría una miniatura de la ROI, por ejemplo 160×90 o 320×180, con:

```text
grayscale
→ absdiff / running background
→ threshold
→ morphology
→ changed-pixel fraction
```

o MOG2 si te funciona mejor.

No esperaría que background subtraction sea perfectamente robusto a:

- lluvia,
- reflejos,
- faros,
- sombras,
- movimiento de árboles,
- cambio día/noche.

No importa demasiado si produce **falsos positivos**; solo debe evitar falsos negativos peligrosos. Su función es ahorrar detección cuando está claramente vacío.

**OCR inteligente**

Yo iría incluso más lejos que tu propuesta. No tienes por qué ejecutar plate detection continuamente tampoco.

Cada `VehicleTrack` debería contener algo como:

```python
VehicleTrack:
    id
    first_pts
    last_pts
    direction
    has_crossed
    bbox_history
    best_vehicle_frames      # max 5–10
    best_plate_candidates    # max 3–5
    ocr_results
```

Durante el track:

```text
vehicle enters OCR zone
        ↓
vehicle bbox suficientemente grande?
        ↓
frame sharp?
        ↓
vehicle no cortado por borde?
        ↓
insertar en Top-M
```

Al terminar el evento:

```text
Top-M vehicle frames, por ejemplo 5
        ↓
plate detector batch=5
        ↓
rank de plate crops
        ↓
Top-K = 3
        ↓
OCR batch=3
```

Un quality score inicial podría ser:

```text
quality =
    w1 * plate_detection_conf
  + w2 * normalized_plate_area
  + w3 * sharpness
  + w4 * distance_from_image_edge
  + w5 * geometric/frontality_score
```

Para blur puedes empezar con variance-of-Laplacian o Tenengrad. No convertiría esto en un modelo neuronal adicional hasta demostrar que hace falta.

La política OCR podría ser:

```text
OCR candidate 1
OCR candidate 2

if normalized_text_1 == normalized_text_2
   and confidences are high:
       early stop

else:
       OCR candidate 3

if disagreement remains:
       candidates 4–5
```

Supongamos —**ESTIMACIÓN ilustrativa**— que un vehículo aparece durante 5 segundos a 30 FPS. Son 150 frames.

OCR cada frame:

```text
150 OCR calls
```

Top-K=3:

```text
3 OCR calls
```

Eso es **50× menos llamadas OCR para ese track**.

Y el benchmark de FastPlateOCR indica que el reconocimiento aislado ya puede ser sub-milisegundo en RTX 3090. citeturn5view0 Por tanto, la estrategia adecuada no es necesariamente optimizarlo de 0,47 ms a 0,30 ms, sino llamarlo tres veces en lugar de 150.

No encontré en los repositorios inspeccionados una implementación mantenida que combine exactamente **tracking + Top-K quality selection + 2–5 OCR + temporal vote + line event** con un benchmark público reproducible. Los componentes sí existen; **esa pequeña capa de orchestration es justamente la parte que construiría a medida**.

**Votación temporal**

No usaría simplemente:

```python
Counter(["ABC1234", "ABC1234", "A8C1234"]).most_common(1)
```

como única estrategia.

Mantendría:

```text
text
OCR confidence
plate detector confidence
quality score
timestamp
```

y puntuaría candidatos:

```text
vote_weight =
    OCR_confidence
    × plate_confidence
    × quality
```

Para resultados cercanos:

```text
ABC1234
A8C1234
ABC1234
```

puedes hacer majority vote por carácter después de alinear cadenas de longitud compatible, además de voto por string completo.

**No agregaría regex estadounidense/europeo como filtro duro.**

Como máximo utilizaría reglas ecuatorianas específicas **después de medir tu propio dataset**, y inicialmente solo como score auxiliar, nunca como condición para tirar una lectura.

FastPlateOCR permite fine-tunear modelos pretrained y exportarlos nuevamente, por lo que es una buena ruta para adaptar un recognizer a tus propios crops. citeturn26view1

Para Ecuador/Latinoamérica mi estrategia sería:

```text
modelo global
    ↓
benchmark con tus placas
    ↓
crear failure set:
  noche
  lluvia
  blur
  motos
  inclinación
  placa pequeña
  glare
  placa dañada
    ↓
fine-tune
```

Hay otro detalle metodológico importante: al crear train/validation/test, **separa por vehículo/placa, no por frame aleatorio**. Si tienes 100 frames casi idénticos de la misma matrícula y repartes 80 en train y 20 en test, el benchmark puede dar una imagen artificialmente optimista.

**Deduplicación**

Tracking ID es suficiente solo mientras el track sea estable y permanezca dentro del mismo clip.

Tus clips cortos crean este problema:

```text
clip_001.mp4
    vehículo aparece
    ↓
EOF

clip_002.mp4
    mismo vehículo sigue apareciendo
    ↓
nuevo tracker ID
```

Por tanto usaría dedup en dos niveles.

Dentro de un clip:

```text
(track_id, crossing_line_id)
```

solo puede producir un evento una vez.

Entre clips:

```text
normalized_plate
+ direction
+ timestamp proximity
+ optional appearance similarity
```

Ejemplo conceptual:

```python
same_vehicle = (
    same_direction
    and abs(timestamp_a - timestamp_b) < cooldown
    and (
        high_conf_plate_a == high_conf_plate_b
        or visual_similarity(vehicle_crop_a, vehicle_crop_b) > threshold
    )
)
```

El cooldown debe medirse con tu entrada; no fijaría arbitrariamente 30 segundos en producción.

Cuando OCR sea incierto, conservaría alternativas:

```json
{
  "plate": "ABC1234",
  "plate_confidence": 0.91,
  "alternatives": [
    ["ABC1234", 0.91],
    ["A8C1234", 0.07]
  ]
}
```

El evento final:

```text
timestamp
direction
plate
plate_confidence
vehicle_confidence
track_id
clip_id
vehicle_crop_path
plate_crop_path
ocr_votes
```

**Paralelismo con muchos clips**

Usaría una arquitectura producer/consumer con backpressure:

```text
              ┌→ decoder worker clip A ─┐
disk/files ───┼→ decoder worker clip B ─┼→ bounded frame queue
              ├→ decoder worker clip C ─┤
              └→ prefetch next file ────┘
                                         ↓
                                  GPU batcher
                                         ↓
                                TensorRT detector
                                         ↓
                       demultiplex by clip/stream ID
                              ↓       ↓       ↓
                           tracker  tracker  tracker
                              \       |       /
                               candidate queue
                                      ↓
                         batched plate/OCR worker
```

Con una sola GPU empezaría con:

```text
2–4 decoders
1 GPU inference process
batch 4–16 vehicle frames
bounded queue = 2–4 batches
```

y mediría.

Para plate OCR, como los crops son pequeños, puedes acumular eventos de varios clips y ejecutar:

```text
batch=8 / 16 / 32
```

pero **no afirmo una aceleración concreta** porque el benchmark público de FastPlateOCR citado es batch 1. citeturn5view0

Para no saturar VRAM:

- mantener un solo juego de weights;
- reutilizar TensorRT execution contexts;
- usar bounded queues;
- no almacenar frames completos de todos los tracks;
- mantener únicamente Top-M crops;
- preasignar buffers cuando sea posible;
- no producir un video anotado;
- no dejar crecer infinitamente el número de batches “in flight”.

DeepStream usa precisamente batching multi-source y NVIDIA documenta que `nvstreammux` y `nvinfer` tienen batches con semánticas diferentes; la memoria aumenta con decode buffers, scaling y batch size, por lo que conviene dimensionarlos explícitamente. citeturn11search1

## Plan de implementación, dependencias y pseudocódigo

Implementaría esto en fases para que cada optimización sea medible.

**Primero: instrumentar el pipeline actual.**

Antes de cambiar modelos añade timers a:

```text
decode
vehicle_detector
tracker
plate_detector
OCR
postprocess
image_write
database_write
video_render/encode
```

No optimices todavía.

**Segundo: eliminar output innecesario.**

Desactiva:

```text
cv2.imshow
drawing boxes
annotated full video
video writer
debug images por frame
```

Guarda únicamente:

```text
1 vehicle JPEG/event
1 plate JPEG/event
metadata
```

**Tercero: convertir de frame-centric a track-centric.**

Añade ByteTrack y crea estado persistente por vehicle ID.

ByteTrack permite consumir detecciones externas, así que esto puede hacerse sin reemplazar tu YOLO. citeturn9view0

**Cuarto: line crossing antes del OCR.**

Un track que nunca cruza una zona relevante:

```text
0 OCR calls
```

Un peatón:

```text
0 OCR calls
```

Un auto estacionado fuera de la zona:

```text
0 OCR calls
```

**Quinto: Top-M vehicle frames y Top-K OCR.**

Prueba:

```text
M = 5
K = 3
```

como primera configuración.

No son números mágicos; luego los optimizas contra exact-match plate accuracy.

**Sexto: bajar detector rate.**

A/B:

```text
30 → 15 → 10 → 5 → 2 FPS
```

y detente en el punto anterior a que event recall empiece a caer.

**Séptimo: ROI.**

Recorta físicamente la región y comprueba si puedes reducir `imgsz` del detector.

**Octavo: motion gate con sentinel.**

Solo después de tener line crossing funcionando.

**Noveno: cambiar OCR.**

Benchmark:

```text
current OCR
vs
FastPlateOCR cct-xs-v2-global
vs
FastPlateOCR cct-s-v2-global
vs
Paddle recognition-only
```

El repo FastPlateOCR proporciona directamente `m.benchmark()` y fine-tuning. citeturn26view1

**Décimo: exportar detector.**

Prueba sucesivamente:

```text
PyTorch FP16
ONNX Runtime CUDA
TensorRT FP16
```

sin cambiar otra variable.

**Undécimo: hardware decoding.**

Compara FFmpeg CPU vs NVDEC con el mismo conjunto de videos.

**Duodécimo: concurrencia entre archivos.**

Una vez que un clip funciona correctamente:

```text
2 clips → 4 clips → 8 clips
```

con un GPU worker batched hasta encontrar la saturación.

**Dependencias concretas para Opción A/B**

Un stack razonable sería:

```text
Python 3.11/3.12
ffmpeg + ffprobe
opencv-python-headless
numpy
ultralytics                # entrenamiento/export/baseline
onnxruntime-gpu
fast-alpr
fast-plate-ocr
open-image-models
ByteTrack implementation
# o boxmot solo durante benchmarking
psutil
pynvml / nvidia-ml-py
pyarrow
pandas
```

Opcionales:

```text
av                         # PyAV decode
supervision                # zones/line utilities
```

Para NVIDIA máximo:

```text
DeepStream 9.1
GStreamer
CUDA 13.2
TensorRT 10.16.x
NVIDIA driver 595+
Python 3.12 / Service Maker cuando corresponda
```

Esas versiones son las que NVIDIA documenta actualmente para DeepStream 9.1 x86/Jetson; usa la matriz exacta de la plataforma en vez de combinar wheels arbitrariamente. citeturn19view0

No instalaría TensorRT 11 sobre ese stack solo porque el repositorio TensorRT standalone ya anuncie 11.X. citeturn14view3turn19view0

**Pseudocódigo del pipeline recomendado**

```python
from dataclasses import dataclass, field
from heapq import heappush, heappushpop

TARGET_DETECT_FPS = 10.0
SENTINEL_FPS = 1.0
MAX_VEHICLE_CANDIDATES = 5
MAX_OCR_CANDIDATES = 3


@dataclass
class Candidate:
    score: float
    timestamp: float
    vehicle_crop: object
    vehicle_bbox: tuple


@dataclass
class TrackState:
    track_id: int
    first_timestamp: float
    last_timestamp: float
    crossed: bool = False
    direction: str | None = None
    candidates: list = field(default_factory=list)


def push_top_candidate(track: TrackState, candidate: Candidate) -> None:
    item = (
        candidate.score,
        candidate.timestamp,
        candidate,
    )

    if len(track.candidates) < MAX_VEHICLE_CANDIDATES:
        heappush(track.candidates, item)
    elif candidate.score > track.candidates[0][0]:
        heappushpop(track.candidates, item)


def process_clip(clip, global_event_cache):
    """
    Decoder is sequential. No random seeking every Nth compressed frame.
    """

    tracker = create_bytetrack()
    tracks: dict[int, TrackState] = {}

    motion_active = False
    last_detector_ts = -1e9
    last_sentinel_ts = -1e9

    for frame, pts in decoder(clip):

        roi = crop_detection_roi(frame)

        # Extremely cheap gate on a downscaled ROI.
        motion_score = motion_detector.update(downscale_gray(roi))

        if motion_score > MOTION_THRESHOLD:
            motion_active = True
            motion_hold_until = pts + MOTION_HOLD_SECONDS

        if motion_active and pts > motion_hold_until:
            motion_active = False

        if motion_active:
            run_detector = (
                pts - last_detector_ts >= 1.0 / TARGET_DETECT_FPS
            )
        else:
            # Safety watchdog: never trust motion gating absolutely.
            run_detector = (
                pts - last_sentinel_ts >= 1.0 / SENTINEL_FPS
            )

        if not run_detector:
            continue

        detections = vehicle_detector(roi)

        last_detector_ts = pts
        if not motion_active:
            last_sentinel_ts = pts

        detections = filter_vehicle_classes(detections)
        track_outputs = tracker.update(detections)

        for trk in track_outputs:

            state = tracks.setdefault(
                trk.id,
                TrackState(
                    track_id=trk.id,
                    first_timestamp=pts,
                    last_timestamp=pts,
                ),
            )
            state.last_timestamp = pts

            # Crossing is calculated from trajectory, not exact-line overlap.
            crossing = crossing_detector.update(
                track_id=trk.id,
                centroid=trk.centroid,
                timestamp=pts,
            )

            if crossing.just_crossed and not state.crossed:
                state.crossed = True
                state.direction = crossing.direction

            # Only collect frames from the plate-readable region.
            if inside_ocr_zone(trk.centroid):
                vehicle_crop = crop(frame, trk.full_frame_bbox)

                q = vehicle_frame_quality(
                    crop=vehicle_crop,
                    bbox=trk.full_frame_bbox,
                    frame_shape=frame.shape,
                    detector_confidence=trk.confidence,
                )

                if q >= VEHICLE_QUALITY_MIN:
                    push_top_candidate(
                        state,
                        Candidate(
                            score=q,
                            timestamp=pts,
                            vehicle_crop=vehicle_crop,
                            vehicle_bbox=trk.full_frame_bbox,
                        ),
                    )

        # Finalize disappeared tracks promptly.
        for state in expired_tracks(tracks, pts):
            if state.crossed:
                finalize_vehicle_event(state, global_event_cache)
            del tracks[state.track_id]


def finalize_vehicle_event(track: TrackState, global_event_cache):
    vehicle_candidates = sorted(
        (x[2] for x in track.candidates),
        key=lambda x: x.score,
        reverse=True,
    )

    # One batch, instead of plate detection every video frame.
    plate_predictions = plate_detector.batch_predict(
        [c.vehicle_crop for c in vehicle_candidates]
    )

    plate_candidates = []

    for vehicle_candidate, plate_pred in zip(
        vehicle_candidates,
        plate_predictions,
    ):
        if not plate_pred:
            continue

        plate_crop = extract_plate(
            vehicle_candidate.vehicle_crop,
            plate_pred.bbox,
        )

        quality = plate_quality(
            plate_crop=plate_crop,
            detection_conf=plate_pred.confidence,
        )

        plate_candidates.append(
            (
                quality,
                vehicle_candidate.timestamp,
                vehicle_candidate.vehicle_crop,
                plate_crop,
                plate_pred.confidence,
            )
        )

    plate_candidates.sort(reverse=True)
    best = plate_candidates[:MAX_OCR_CANDIDATES]

    if not best:
        # Still emit a vehicle event if required;
        # do not silently lose the entry/exit.
        emit_vehicle_without_plate(track)
        return

    # Tiny batch of only the best crops.
    ocr_results = plate_ocr.batch_predict(
        [x[3] for x in best],
        return_confidence=True,
    )

    consensus = temporal_plate_vote(
        plate_candidates=best,
        ocr_results=ocr_results,
    )

    event = {
        "timestamp": choose_event_timestamp(track, best),
        "direction": track.direction,
        "plate": consensus.text,
        "plate_confidence": consensus.confidence,
        "track_id": track.track_id,
        "vehicle_crop": best[0][2],
        "plate_crop": best[0][3],
        "ocr_votes": consensus.votes,
    }

    if not is_cross_clip_duplicate(event, global_event_cache):
        write_event(event)
        global_event_cache.add(event)
```

Un detalle intencional del pseudocódigo es este:

> **Si la placa falla, no elimines el evento de vehículo.**

Tu prioridad número uno es no perder entradas/salidas. “Vehículo detectado pero placa ilegible” es información útil y debe quedar registrada para inspección/manual retry.

**Qué NO implementaría inicialmente**

No haría OCR sobre todos los frames.

No detectaría placas sobre el frame completo continuamente.

No generaría video anotado.

No arrancaría un proceso GPU independiente por clip.

No aplicaría regex extranjera como filtro duro.

No usaría BoT-SORT/ReID antes de demostrar ID-switches que ByteTrack no resuelva.

No empezaría con CUDA streams custom.

No haría random seek cada N frames en H.264/H.265.

No migraría inmediatamente todo a C++/DeepStream antes de medir dónde se van las cuatro horas.

No almacenaría cientos de vehicle/plate crops por track.

No escogería el detector de placas más pequeño sin comprobar recall: la propia tabla publicada de `open-image-models` muestra una caída notable de recall al ir hasta 256 px. citeturn26view2

## Benchmark reproducible, aceleración esperable y riesgos

Tu benchmark debe separar **accuracy de throughput**. Optimizar un pipeline hasta 20× realtime no sirve si pierde un 5% de los vehículos.

Crearía un conjunto `benchmark_set` que contenga:

```text
día
noche
sombra
sol frontal
faros
lluvia si existe
vehículos lentos
vehículos rápidos
motos
vehículos parcialmente ocluidos
placas pequeñas
clips que cortan el vehículo entre archivos
```

Y anotaría manualmente, como mínimo:

```text
ground_truth_event_id
timestamp aproximado
direction
plate_text si legible
plate_legible = true/false
```

La métrica más importante será:

```text
event_recall =
    matched_ground_truth_events /
    all_ground_truth_vehicle_events
```

Después:

```text
direction_accuracy

plate_detection_recall

plate_exact_match_accuracy =
    exact_plate_matches /
    legible_ground_truth_plates

character_accuracy

duplicate_event_rate =
    duplicate_output_events /
    true_events
```

No usaría únicamente “OCR confidence”; una red puede estar altamente confiada y equivocada.

**Instrumentación por etapa**

Por clip recoge:

```text
video_duration_seconds
source_frame_count
decoded_frame_count
detector_frame_count

decode_seconds
motion_gate_seconds
vehicle_detection_seconds
tracking_seconds
plate_detection_seconds
ocr_seconds
postprocessing_seconds
write_seconds

processing_seconds_wall
```

Y calcula:

```text
speed_ratio =
    video_duration_seconds / processing_seconds_wall
```

Ejemplos:

```text
6 h / 4 h = 1.50× realtime
7 h / 4 h = 1.75× realtime

6 h / 45 min = 8.00× realtime
7 h / 45 min = 9.33× realtime
```

Define también **dos FPS diferentes**, porque un solo “FPS” puede ser engañoso:

```text
source_equivalent_FPS =
    total_source_frames / wall_seconds

actual_detector_FPS =
    detector_calls / detector_seconds
```

Si decodificas 30 FPS pero detectas 5 FPS, decir simplemente “500 FPS” no informa qué stage produjo ese número.

Recoge además:

```text
CPU utilization
system RAM
GPU utilization
GPU memory
NVDEC utilization, si disponible
GPU power
disk read MB/s
```

con `psutil` y NVML/`nvidia-smi dmon`.

Para GPU timing, recuerda que CUDA es asíncrono. Un patrón como:

```python
t0 = perf_counter()
model(...)
t1 = perf_counter()
```

puede medir launch time en vez de ejecución efectiva. Para microbenchmarks usa CUDA events o sincroniza correctamente antes de leer el reloj.

**Warm-up**

Excluye:

```text
model loading
TensorRT engine building
first CUDA initialization
primeros warm-up batches
```

del throughput de régimen estable, pero mide el startup por separado.

La aplicación NVIDIA TAO incluso recomienda utilizar `trtexec` para medir el tiempo de inferencia por batch de motores TensorRT. citeturn18view1

**Experimentos que ejecutaría exactamente en este orden**

| Run | Cambio respecto baseline | Qué descubre |
|---|---|---|
| B0 | Pipeline actual | Base |
| B1 | Sin annotated video | Coste de render/encode |
| B2 | OCR desactivado | Coste real de OCR actual |
| B3 | Plate detector desactivado | Coste plate stage |
| B4 | YOLO 15 FPS | sensibilidad a sampling |
| B5 | YOLO 10 FPS | idem |
| B6 | YOLO 5 FPS | idem |
| B7 | YOLO 2 FPS | límite agresivo |
| B8 | ROI crop | ahorro/accuracy ROI |
| B9 | Motion gate + 10 FPS active | ahorro por zonas vacías |
| B10 | Top-K OCR=3 | ahorro OCR real |
| B11 | FastPlateOCR | OCR backend |
| B12 | ONNX CUDA | runtime detector |
| B13 | TensorRT FP16 | runtime detector |
| B14 | FFmpeg CPU decode | decode baseline |
| B15 | NVDEC | beneficio hardware decode |
| B16 | 2 clips batched | GPU utilization |
| B17 | 4 clips batched | escalabilidad |
| B18 | 8 clips batched | punto de saturación |

Para cada run no basta con throughput. Registra:

```text
event_recall
plate_exact_match
duplicate_rate
```

y rechaza cualquier optimización que vulnere tu umbral.

Yo pondría una condición de aceptación similar a:

```text
event_recall:
    no degradation statistically observed on benchmark set

direction accuracy:
    ~100% except truly ambiguous events

plate exact-match:
    >= baseline

speed:
    materially better
```

**ESTIMACIÓN razonada del potencial de aceleración**

Partimos de:

```text
4 horas processing
para
6–7 horas source

= 1.5–1.75× realtime
```

Solo bajar detección de 30 a 10 FPS significa **3× menos detector calls**.

A 5 FPS:

```text
6× menos
```

Si motion gating determina —ejemplo puramente hipotético— que la región solo está activa el 20% del tiempo, el detector high-rate podría ejecutarse sobre una fracción adicional de esos frames. No convierto 6× × 5× directamente en “30× speedup” porque las etapas se solapan y existen costes fijos.

OCR puede reducirse aún más dramáticamente. Un track visible 150 frames con K=3 pasa de:

```text
150 → 3 calls
= 50× reducción de ese trabajo
```

Si hoy OCR representa el 50% de tus cuatro horas, reducir 50× solo ese stage **no produce 50× global**, sino aproximadamente un máximo cercano a 2× mientras el resto permanezca igual, por la lógica de Amdahl.

Por eso medir porcentajes es esencial.

Mi rango de planificación sería:

| Etapa | ESTIMACIÓN de mejora respecto tus 4 h | 6–7 h procesadas aproximadamente |
|---|---:|---:|
| Solo rediseño A: sampling + tracks + selective OCR | **2×–5×** | ~48–120 min |
| B bien optimizada: +TensorRT/batching/decode pipeline | **4×–10×** | ~24–60 min |
| C NVIDIA muy optimizada, escena favorable | **5×–15×** | ~16–48 min |

**Estas cifras son ESTIMACIONES, no benchmarks publicados de tu máquina.**

¿Por qué considero plausible la parte alta, pero no la garantizo? Porque el sample LPR de NVIDIA publicó **447,15 FPS agregados en T4 con 14 streams 1080p**, incluyendo una arquitectura completa de vehicle detection → plate detection → plate recognition. citeturn10view0 Eso demuestra que un pipeline ALPR correctamente acelerado puede entrar en un régimen de cientos de frames/s agregados. Pero ese benchmark utiliza hardware, modelos, batching y datasets distintos, y por tanto **no equivale a decir que una T4 procesará tu archivo a 447 FPS**.

Del mismo modo, DeepStream actual está diseñado específicamente alrededor de hardware decoding, batching, TensorRT y tracking en una pipeline GStreamer, por lo que su arquitectura elimina varias capas de overhead que aparecen en un loop Python/OpenCV clásico. citeturn19view0turn11search4

**Riesgos principales**

| Riesgo | Consecuencia | Mitigación |
|---|---|---|
| Frame skipping demasiado agresivo | vehículo nunca detectado | empezar 10 FPS, medir dwell time, sentinel |
| Motion gate falla | evento perdido | watchdog YOLO lento cuando “inactive” |
| ROI demasiado pequeña | vehículos parcialmente fuera | margen amplio + benchmark de bordes |
| Plate detector demasiado pequeño | placa no encontrada | probar 512/608; priorizar recall |
| OCR global falla Ecuador | placa incorrecta | recolectar dataset real + fine-tune |
| Tracker fragmenta ID | duplicados | plate/time/direction dedup |
| Clip boundary resetea tracker | doble evento | global event cache entre clips |
| Batch demasiado grande | OOM/latencia | bounded queues + sweep batch size |
| CPU decode limita GPU | baja GPU utilization | FFmpeg threading/NVDEC |
| CPU↔GPU copies | throughput inferior | I/O Binding/GStreamer/NVMM |
| múltiples GPU processes | VRAM/context contention | un inference service/batcher |
| TensorRT incompatibilidad | deployment roto | respetar matriz CUDA/TRT/DS |
| BoT-SORT/ReID innecesario | trabajo extra | ByteTrack primero |
| video annotated output | I/O/encode innecesario | events + crops solamente |
| confianza OCR mal calibrada | lectura segura pero falsa | temporal vote + exact-match validation |

La decisión técnica que más defendería es esta:

> **El detector de vehículos debe descubrir tracks; el tracker debe descubrir eventos; el evento debe decidir cuándo buscar una placa; y solo unos pocos crops de placa deben llegar al OCR.**

No:

```text
video
→ YOLO
→ plate detector
→ OCR
→ repetir todo 756.000 veces
```

sino:

```text
video
→ cheap temporal filtering
→ vehicle tracks
→ crossing events
→ ~5 candidate frames/event
→ ~3 OCR crops/event
→ one database record/event
```

Con tus requisitos —una cámara fija, solo entrada/salida, sin necesidad de video anotado y muchos clips independientes— **hay mucha estructura que aprovechar**. Frigate demuestra la validez práctica de motion-first gating; ByteTrack aporta el estado temporal barato; FastALPR/FastPlateOCR aportan un stack especializado y modular; ONNX Runtime/TensorRT resuelven el runtime acelerado; y DeepStream 9.1 proporciona la arquitectura de máximo throughput en NVIDIA. citeturn26view5turn25view1turn3view0turn5view0turn19view0

La ruta con mejor relación esfuerzo/beneficio es, por tanto:

```text
tu código actual
      ↓
instrumentación
      ↓
10 FPS + ROI
      ↓
ByteTrack + crossing
      ↓
Top-M frame selection
      ↓
FastPlateOCR Top-K=3
      ↓
sin video output
      ↓
benchmark
      ↓
TensorRT FP16
      ↓
batch de clips
      ↓
NVDEC si decode aparece en el profiler
      ↓
DeepStream solo si todavía necesitas más throughput
```

Ese orden evita una reescritura prematura y, sobre todo, optimiza primero **la cantidad de trabajo que haces**, antes de gastar semanas intentando ejecutar trabajo redundante un poco más rápido.