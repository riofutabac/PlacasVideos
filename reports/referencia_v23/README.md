# Referencia congelada — v2.3 (prefetch)

Corrida de referencia contra la que se compara cualquier cambio futuro.
**46.9 min es el tiempo observado de ESTA corrida, no una garantía.**

## Identidad de la corrida

| | |
|---|---|
| `run_id` | `RUN_20260922_192445` |
| Commit del código | `6f63b88` (`feat(video): overlap next-clip SSD staging…`) + este commit de archivo |
| `config_hash` | `d44e8c71` |
| Hardware | Google Colab, Tesla T4 (2 vCPU), videos en Google Drive |
| Inicio → fin | 2026-09-22 19:24:45 → 20:11:42 (**46.9 min**, 6.88× realtime) |
| Entrada | 63 archivos (61 procesados, 2 corruptos: clips 8 y 48) · 5 h 20 min de video |

## Modelos

```
vehicle_detector: yolov8n.onnx (imgsz=416, device=cuda)
plate_detector:   yolo-v9-t-512-license-plate-end2end (CUDA)
ocr_engine:       cct-xs-v2-global-model (CUDA)
```

## Resultados

| Métrica | Valor |
|---|---|
| Eventos únicos | **137** (+8 duplicados apartados) |
| Recall de eventos (clips 60-61) | 8/8 |
| Dirección (clips 60-61) | 8/8 |
| Falsos positivos (clips 60-61) | 0 |
| Placas exactas (clips 60-61) | 2/6 legibles |
| Caracteres (honesto, s/41) | 75.6% |
| Placas legibles sin lectura | 1 (TAA2204) |
| Primer carácter | 50% (3/6) |
| Cruce con bitácora (39 auditados) | 17 exactas · 3 cercanas · 15 solo-vehículo · 4 sin cruce |

> Las métricas de recall/dirección/falsos positivos corresponden **solo a los clips 60-61**, que son
> los únicos con ground truth. No están demostradas para todo el día.

## Desglose de tiempo (con prefetch)

```
Startup               14.68 s
Staging/Copy         438.02 s   (solapado con el procesamiento)
File Hash             30.49 s
OSD Clock Read        20.95 s
Clip Processing     2745.02 s
Wait For Next File     6.49 s
Excel Export           1.94 s
Total Wall Clock    2814.75 s
```

## Archivos de esta carpeta

- `events_RUN_20260922_192445.sqlite` — base completa de la corrida (eventos, clips, telemetría).
- `camera_config.yaml` — configuración exacta usada.

## Cómo comparar un cambio futuro contra esta referencia

```bash
python benchmarks/compare_runs.py \
    reports/referencia_v23/events_RUN_20260922_192445.sqlite \
    <base_de_la_nueva_corrida>.sqlite
```

Un cambio se acepta solo si **no hay diferencias por evento** (placa, dirección, tipo, duplicado) y
el tiempo mejora de forma reproducible. Ver `reports/experimento_fast_convert.md` para un caso que
fue rechazado exactamente por esta regla.

## Casos abiertos que toda versión nueva debe volver a comprobar

- `XBS0640` (clip 17, 12:22): al OCR nunca le llega texto que leer.
- `POD0393` (clip 14, 12:05:59): el OCR lee `TKG111` con 0.645; rechazo correcto por umbral.
- `PDW6349`: sin explicación.
- `PCC2529`: cae en el clip 48, corrupto en origen.
- 102 eventos no listados en la bitácora: **candidatos a revisar**, no omisiones confirmadas.
