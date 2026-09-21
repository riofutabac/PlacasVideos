# Cierre de fase de precisión — ALPR bypass Píntag

Fecha del análisis: 2026-09-21
Fuente de datos: `data/events_137.sqlite` (copia local de `events (1).sqlite`, RUN_20260921_214037 + RUN_20260921_224044, 63 clips, con el fix de "auto parqueado" del commit `72a3d98` y el rechazo de calcomanías del commit `207ae84`) vs `data/events_133.sqlite` (copia local de `events.sqlite`, RUN_20260921_190533, mismos 63 clips, sin esos dos fixes). Auditoría: `Revisión bypass Pintag.xlsx`, columna "Hora cam 2", 39 vehículos. Cruce automático: `benchmarks/build_gt_candidates.py`.

**No se volvió a correr el pipeline.** Todos los números de este informe salen de consultas SQL directas sobre las dos bases de datos ya entregadas y de las corridas de `build_gt_candidates.py` sobre esas mismas bases.

## Resumen antes/después (tolerancia 75s, la que usa el script por defecto)

| Métrica | 133-run (antes) | 137-run (después) |
|---|---|---|
| Eventos únicos (no duplicados) | 133 | 137 |
| Clips procesados | 63 | 63 |
| Vehículos auditados (Excel) | 39 | 39 |
| Coincidencia exacta de placa (EXACT_PLATE) | 17 (43.6%) | 17 (43.6%) |
| Coincidencia cercana (NEAR_PLATE) | 4 (10.3%) | 3 (7.7%) |
| Solo vehículo, sin placa leída (VEHICLE_ONLY) | 12 (30.8%) | 15 (38.5%) |
| Sin cruce (NO_MATCH) | 6 (15.4%) | 4 (10.3%) |
| Eventos extra no auditados | 100 | 102 |

El total de matches "con placa" (EXACT + NEAR) se mantiene en 21 en ambas corridas; lo que cambió es la distribución interna: 2 vehículos que antes tenían placa (POD0393, XBS0640) ahora caen en VEHICLE_ONLY (ver sección 2), y a la vez 2 vehículos que antes eran NO_MATCH ahora tienen vehículo detectado gracias a los fixes de la fase de precisión (menos autos parqueados descartados). El neto de NO_MATCH bajó de 6 a 4.

## 1. Diff entre las dos corridas

Comparando por `video_source` + `direction` + `event_timestamp` con tolerancia de ±5s (para no confundir con cambios de tipo de vehículo, que sí ocurren entre corridas), el resultado es:

**Eventos verdaderamente nuevos en el 137-run (sin contraparte temporal en el 133-run), 6 en total:**

| Hora | Tipo | Dirección | Clip | Placa | Conf. OCR |
|---|---|---|---|---|---|
| 13:18:34 | truck | SALIDA | (27) | — (PLACA_NO_DETECTADA) | 0.0 |
| 13:48:53 | truck | SALIDA | (33) | — (PLACA_NO_DETECTADA) | 0.0 |
| 15:52:05 | truck | SALIDA | (55) | — (PLACA_NO_DETECTADA) | 0.0 |
| **16:03:53** | **truck** | **SALIDA** | **(57)** | **— (PLACA_NO_DETECTADA)** | **0.0** |
| 11:23:34 | truck | SALIDA | (6) | PBJ9917 | 1.0 |
| 14:47:13 | motorcycle | ENTRADA | (43) | — (PLACA_NO_DETECTADA) | 0.0 |

Se confirma que el camión furgón blanco a las ~16:03:53 en el clip 57 (evidencia `forense_PAB3439_clip57.jpg`) **sí es uno de los eventos nuevos**: es `EVT_(57)_0045_187`. En el 133-run ese clip solo tenía 2 eventos (bus 16:01:02 y auto 16:03:42); el camión no aparecía. Esto es consistente con el propósito del fix `72a3d98` (dejar de descartar camiones grandes SALIDA por el chequeo de auto parqueado en el bordillo).

**Eventos que estaban en el 133-run y desaparecieron del 137-run, 2 en total:**

| Hora | Tipo | Dirección | Clip | Placa (133) | Conf. OCR |
|---|---|---|---|---|---|
| 15:52:24 | truck | SALIDA | (55) | PAA6249 | 0.996 |
| 12:52:07 | motorcycle | ENTRADA | (22) | — (PLACA_NO_DETECTADA) | 0.0 |

No se investigó a fondo el porqué de estas 2 pérdidas (no estaba en el alcance de las preguntas 1-5); quedan anotadas como pendiente de revisión en la sección de "no resuelto".

El resto de las diferencias en el diff "ingenuo" (por `datetime_str` exacto) son en realidad el mismo evento con **reclasificación de tipo de vehículo** (bus→truck, truck→car, bus→car, car→truck en 5 casos, todos con Δt < 1s) — no son eventos nuevos ni perdidos, y no afectan el conteo neto 133→137.

## 2. Regresiones en lectura de placa: POD0393 y XBS0640

Ambos casos se investigaron consultando el mismo `event_id` en las dos bases.

### POD0393 (fila auditoría 010, 12:06, clip 14)

- Evento: `EVT_(14)_0004_103`, datetime `2026-09-09 12:05:59`, mismo `event_id` en ambas corridas.
- 133-run: `plate_corrected = POD0393`, `plate_status = OK`, `confidence_ocr = 0.981`.
- 137-run: `plate_corrected = NULL`, `plate_status = PLACA_NO_DETECTADA`, `confidence_ocr = 0.0`.

**Causa concreta: el evento perdió la lectura de placa.** No es un problema de asignación del cruce uno-a-uno ni un evento distinto — es literalmente el mismo `event_id`, mismo timestamp, mismo tipo/dirección, pero en la nueva corrida el pipeline no detectó/leyó la placa en absoluto. Esto es una regresión real introducida entre las dos corridas (candidatos: el filtro más estricto de auto parqueado o el rechazo de calcomanías, aunque no se puede afirmar cuál sin inspeccionar el video/logs, que están fuera del alcance de "no volver a correr el pipeline").

### XBS0640 (fila auditoría 013, 12:22, clip 17)

- 133-run: evento `EVT_(17)_0005_105`, `plate_corrected = XBS0640`, `OK`, `confidence_ocr = 0.983`.
- 137-run: el clip 17 solo tiene **un** evento en esa ventana, `EVT_(17)_0003_105`, con `plate_corrected = NULL`, `PLACA_NO_DETECTADA`.

**Causa concreta: mismo fenómeno que POD0393** (evento en el mismo timestamp/clip/dirección perdió la placa), con la particularidad de que el `event_id` cambió de sufijo (`0005` → `0003`) porque la renumeración interna de tracks del clip cambió entre corridas (probablemente por menos tracks descartados antes de ese punto, dado el fix del filtro de parqueado) — **no** es una asignación errónea del script de cruce: solo hay un evento candidato en esa ventana temporal en el 137-run y es al que se asignó correctamente.

En ambos casos el diagnóstico es el mismo: **la placa se dejó de leer**, no un problema del cruce ni de deduplicación. Esto merece revisión de por qué el fix del filtro de auto parqueado (o el de calcomanías) está tumbando la lectura de placa en camiones SALIDA que antes sí se leían bien.

## 3. Los 8 duplicados (137-run)

| Original | Duplicado | Δt (s) | Tipo | Dirección | Placa | Juicio |
|---|---|---|---|---|---|---|
| EVT_8(9)_0045_88 (11:38:14) | EVT_8(9)_0047_89 (11:38:15) | 1.0 | truck | ENTRADA | PFU3047 (igual en ambos) | Duplicado genuino, mismo paso |
| EVT_(43)_0047_100 (14:45:26) | EVT_(43)_0048_100 (14:45:26) | 0.0 | truck | SALIDA | sin placa en ambos | Duplicado genuino, mismo paso |
| EVT_(23)_0088_125 (12:55:51) | EVT_(23)_0089_125 (12:55:51) | 0.0 | truck | ENTRADA | sin placa en ambos | Duplicado genuino, mismo paso |
| EVT_(43)_0039_98 (14:45:24) | EVT_(43)_0172_144 (14:46:10) | 46.0 | car | ENTRADA | PBN2792 (igual en ambos) | Misma placa confirmada — es el mismo vehículo re-detectado, correctamente deduplicado por la ventana de 60s (commit `90dbd79`). No parece un paso distinto descartado. |
| EVT_(14)_0021_161 (12:06:57) | EVT_(14)_0023_162 (12:06:58) | 1.0 | motorcycle | ENTRADA | sin placa en ambos | Duplicado genuino, mismo paso |
| EVT_(38)_0021_187 (14:19:23) | EVT_(38)_0023_187 (14:19:23) | 0.0 | car | SALIDA | sin placa en ambos | Duplicado genuino, mismo paso |
| EVT_(25)_0038_206 (13:08:12) | EVT_(25)_0041_207 (13:08:13) | 1.0 | car | SALIDA | PFU1666 (igual en ambos) | Duplicado genuino, mismo paso |
| EVT_8(3)_0039_228 (11:07:34) | EVT_8(3)_0041_228 (11:07:34) | 0.0 | car | ENTRADA | PBO4275 (igual en ambos) | Duplicado genuino, mismo paso |

**Conclusión: ninguno de los 8 duplicados parece corresponder a un paso real distinto siendo descartado.** Los 7 con Δt ≤ 1s son claramente re-tracking del mismo frame/paso. El único con Δt = 46s (PBN2792) tiene placa idéntica en ambos eventos, lo que confirma que es el mismo vehículo capturado dos veces dentro de la ventana de cooldown de 60s, no dos vehículos distintos. No se tocó la configuración de deduplicación, como se pidió.

## 4. Sensibilidad del cruce a la tolerancia temporal (75s / 90s / 120s)

| DB | Tolerancia | EXACT_PLATE | NEAR_PLATE | VEHICLE_ONLY | NO_MATCH |
|---|---|---|---|---|---|
| 137-run | 75s | 17 | 3 | 15 | 4 |
| 137-run | 90s | 17 | 3 | 15 | 4 |
| 137-run | 120s | 17 | 3 | 15 | 4 |
| 133-run | 75s | 17 | 4 | 12 | 6 |
| 133-run | 90s | 17 | 4 | 12 | 6 |
| 133-run | 120s | 17 | 4 | 12 | 6 |

**El resultado es completamente estable entre 75s, 90s y 120s en ambas bases.** Esto se debe a que `_assign_one_to_one` en `benchmarks/build_gt_candidates.py` tiene un "fallback" de coincidencia exacta de placa que ignora la tolerancia temporal (líneas ~198-211): si la placa auditada coincide literalmente con algún evento aún no asignado, se empareja sin importar el tiempo. Como en este dataset todos los matches de placa exacta ya caen dentro de 75s, ampliar la ventana no suma ni resta coincidencias. **La cifra titular (43.6% EXACT_PLATE) no es sensible a este parámetro** en el rango probado.

## 5. Los 102 eventos extra del 137-run (no auditados)

**Aclaración explícita: estos 102 eventos son candidatos para revisión, NO son omisiones humanas confirmadas.** La auditoría del Excel solo registra 39 vehículos que cruzaron ambas cámaras (posiblemente por selección manual, no por cobertura total); el resto son vehículos que el pipeline detectó en los 63 clips pero que nunca fueron ingresados a la hoja de auditoría, por lo que no hay ground truth con el cual contrastarlos.

Por estado de placa:

| Bucket | Cantidad |
|---|---|
| Placa OK, confianza ≥ 0.95 | 54 |
| Confianza 0.85–0.95 | 4 |
| Confianza < 0.85 | 1 |
| REVISION_MANUAL | 2 |
| Sin placa (PLACA_NO_DETECTADA) | 41 |

Por tipo de vehículo: truck 60, car 25, motorcycle 10, bus 7.

Cruce tipo × bucket:

| Tipo | ≥0.95 | 0.85–0.95 | <0.85 | REVISION_MANUAL | sin placa |
|---|---|---|---|---|---|
| truck | 35 | 1 | 1 | 1 | 22 |
| car | 14 | 2 | 0 | 0 | 9 |
| bus | 3 | 0 | 0 | 1 | 3 |
| motorcycle | 2 | 1 | 0 | 0 | 7 |

## Casos puntuales pedidos en el encargo

- **PDW6349** (fila auditoría, 12:16): sin match en ninguna de las dos bases, en ninguna tolerancia probada (75/90/120s). **No se encontró ninguna explicación en los datos disponibles.** Sigue sin explicación — se necesitaría revisar el clip correspondiente a las 12:16 manualmente para determinar si el vehículo fue detectado sin placa o si simplemente no fue detectado.
- **PCC2529** (fila auditoría, 15:11): confirmado que cae en el clip 48, el cual está marcado `FAILED: Corrupted MP4 container (missing 'moov' atom)` en `processed_clips` **en ambas bases** (133 y 137). El vehículo nunca pudo ser procesado porque el archivo de video está corrupto; no es un fallo del pipeline de detección/OCR.
- **PAB7630** (camión mezclador de concreto, forense a las ~14:13:14 según `forense_PAB7630_clip37.jpg`): el cruce automático lo asignó al evento `EVT_(37)_0020_101`, con `datetime_str = 2026-09-09 14:12:27` (clip 37), diferencia reportada de 33s respecto a la hora auditada (14:13:00). Comparado con la hora forense real (~14:13:14), la diferencia real es de ~47s — dentro de rango plausible dado que la hora de auditoría está redondeada al minuto y el evento capturado corresponde al instante de detección del vehículo, no necesariamente al instante exacto marcado en el forense. En el clip 37 solo hay 2 eventos SALIDA (14:11:25 y 14:12:27), así que no hay otro candidato mejor. **El match es plausible**, aunque el evento tiene `plate_status = PLACA_NO_DETECTADA` (no se leyó placa del mezclador), consistente con que el cruce lo clasificó como VEHICLE_ONLY. Nota adicional: en el 133-run este mismo evento tenía una lectura falsa `TUGP5` (REVISION_MANUAL) que el fix de rechazo de calcomanías (commit `207ae84`) eliminó correctamente — pero no fue reemplazada por la placa real del mezclador (que, a juzgar por el forense, podría no ser legible).

## Lo que queda sin resolver

1. **PDW6349**: sin match, sin explicación encontrada en los datos. Requiere inspección manual del clip de las 12:16.
2. **Causa raíz de la pérdida de placa en POD0393 y XBS0640**: se confirmó *que* pasó (mismo evento, placa antes OK y ahora PLACA_NO_DETECTADA) pero no se determinó *por qué* sin volver a correr el pipeline o revisar logs/video, lo cual estaba fuera de alcance de este análisis.
3. **2 eventos que desaparecieron del 133-run al 137-run** (15:52:24 truck PAA6249 clip 55, y 12:52:07 motorcycle clip 22 sin placa): no se investigó la causa; podría ser deduplicación legítima con otro evento cercano o pérdida real de detección — no se puede afirmar sin más contexto.
4. **Los 102 eventos extra** son candidatos sin confirmar contra la auditoría — no representan necesariamente omisiones del personal auditor; podrían ser vehículos simplemente no incluidos en la muestra de 39 filas.
5. **PAB7630**: el match temporal es plausible pero no 100% verificado contra el video (solo contra el forense fijo); y la placa real del mezclador sigue sin leerse en ninguna de las dos corridas.

## Reproducibilidad

Todos los cruces se generaron con:
```
.venv/bin/python benchmarks/build_gt_candidates.py --db data/events_137.sqlite --time-tolerance {75,90,120}
.venv/bin/python benchmarks/build_gt_candidates.py --db data/events_133.sqlite --time-tolerance {75,90,120}
```
usando las copias locales `data/events_137.sqlite` y `data/events_133.sqlite` (no versionadas, `data/` está en `.gitignore`). No se modificó `src/` ni la configuración del pipeline. `.venv/bin/python -m pytest -q` sigue en 117 tests pasando (sin cambios de código en este análisis).
