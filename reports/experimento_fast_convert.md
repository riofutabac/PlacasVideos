# Experimento ARCHIVADO: `--fast-convert`

**Fecha:** 22-sep-2026 · **Veredicto: RECHAZADO.** No adoptar sin rehacer el experimento completo.

## Qué proponía

Hoy el pipeline convierte a BGR el ROI completo (2300×1064) en cada frame analizado, y Ultralytics
reduce esa imagen internamente antes de inferir. La idea era detectar sobre una imagen reducida y
pagar la conversión a resolución completa solo cuando hay un vehículo, reservando el frame original
para el ranking de calidad, los recortes de placa y la evidencia.

**Motivación medida** (corrida `RUN_20260922_192445`, 63 clips, 46.9 min):
`frame_conversion` 477.92 s (16.9% del lote) sobre 61,853 frames analizados.

## Dato técnico obtenido (sí vale la pena conservarlo)

Ultralytics **no** usa un tensor cuadrado: para un ROI de 2300×1064 con `imgsz=416` hace letterbox a
**416×192** (`[1,3,192,416]`), sin relleno adicional (192 ya es múltiplo de 32). Se midió
instrumentando `BasePredictor.preprocess`, no se supuso.

## Resultado: velocidad

| | Sin `--fast-convert` | Con `--fast-convert` |
|---|---|---|
| Lote (clips 60-61, Mac/OpenCV) | 1443 s | 1418 s |
| Decodificación | 721 s | 725 s |
| YOLO | 495 s | 470 s |

- **Mejora observada: 1.7% (25 s)** en Mac con decoder OpenCV. El criterio de aceptación era ≥10%.
- **Sin repeticiones**, no sabemos si esos 25 s superan la variabilidad entre corridas de esta Mac,
  que históricamente ha variado hasta 3× con el mismo código.
- **Rendimiento en T4/NV12 no evaluado.** El decoder de OpenCV nunca entrega NV12, así que el camino
  donde estaba el margen grande (los 477.92 s de conversión NV12) **no se ejercitó** en esta prueba.

## Resultado: precisión (el motivo real del rechazo)

`benchmarks/compare_runs.py` sobre los mismos 8 eventos de los clips 60-61:
**6 diferencias en 8 eventos.** Vehículos, direcciones y duplicados sí coinciden.

| Evento | Sin fast-convert | Con fast-convert |
|---|---|---|
| clip 60 @ 27.85 s | `PCW2497` (conf 0.865) | `PCW7497` (conf 0.957) |
| clip 60 @ 111.38 s | conf 0.952 | conf 0.925 |
| clip 60 @ 115.0 s | sin placa | `TAA244` (REVISION_MANUAL, 0.768) |
| clip 60 @ 123.44 s | conf 0.971 | conf 0.926 |
| clip 61 @ 80.1 s | `AAC2573` (OK, 0.882) | **sin placa** |
| clip 61 @ 314.25 s | conf 0.998 | conf 0.995 |

**No son seis errores nuevos, pero incumplen el criterio de conservar las salidas.** Y ninguna
lectura mejoró: `PCW7497` sigue mal frente a la real `PCW2492`, `TAA244` no recupera `TAA2204`, y
perder `AAC2573` solo reduce cobertura (esa lectura ya era incorrecta frente a `PAC2573`).

## Causa: hipótesis, no conclusión

La explicación probable es que detectar sobre la imagen reducida produce cajas de vehículo
ligeramente distintas (redondeo al reescalar), y esa diferencia se propaga: caja → recorte del
vehículo → detección de placa → recorte de placa → OCR. **Pero no está demostrado.** También pudo
influir el escalado o el color de la imagen reducida, o que el ranking de calidad seleccionara otros
frames. Para atribuirlo hay que comparar, sobre un mismo evento afectado, los frames seleccionados,
las cajas, los recortes y las lecturas.

## Lección

**La precisión depende de toda la cadena de selección y recorte, no solo del OCR.** Cualquier
optimización futura debe verificar la cadena completa (frames elegidos → cajas → recortes → texto);
no basta con que coincida el número de eventos.

## Estado del código

`src/fast_convert.py`, el flag `--fast-convert` y `video.fast_convert: false` quedan en el
repositorio **apagados por defecto**, junto con sus tests (los 172 de la suite pasan). Con el flag
apagado el camino de ejecución es exactamente el original.

## Alternativas que quedan abiertas

1. **Saltar la conversión solo cuando el motion gate no ve movimiento** — conserva píxeles idénticos
   en todo frame que sí se analiza; menos ahorro, riesgo mucho menor.
2. Localizar vehículos en la imagen reducida y **refinar las cajas con la ruta original** al capturar
   evidencia. Añade inferencias y tampoco garantiza las mismas lecturas: necesita su propio
   experimento.
3. Buscar operaciones redundantes en preparación, posprocesamiento y copias de recortes, que no
   alteren los píxeles que recibe el modelo.
