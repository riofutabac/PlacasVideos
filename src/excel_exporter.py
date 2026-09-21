"""
Professional Excel Report Exporter for ALPR Pipeline.
Generates an audit-ready Excel workbook (.xlsx) from SQLite canonical database.
Features:
- Categorizes audit certainty: ALTA (SEGURA), MEDIA, DUDOSA / REVISIÓN, SIN PLACA.
- Displays consensus reading evidence: e.g. "7/7 lecturas" or "1 sola lectura".
- Physically embeds vehicle and plate crop images directly inside worksheet cells.
- Auto-sizes row heights and column widths to fit images perfectly.
- Clean header styles, conditional colors, and table auto-filters.
- Exports filtered duplicates into a secondary sheet ("Duplicados Descartados").
"""
import os
import json
import sqlite3
from typing import Optional, Dict, Any, Tuple
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.drawing.image import Image as OpenPyXLImage

def compute_audit_certainty(
    final_plate: Optional[str],
    status: str,
    confidence_ocr: float,
    ocr_votes_str: Optional[str],
    reason: Optional[str]
) -> Tuple[str, str, PatternFill, Font]:
    votes = []
    if ocr_votes_str:
        try:
            votes = json.loads(ocr_votes_str)
        except Exception:
            votes = []

    total_reads = len(votes)
    norm_final = (final_plate or "").strip().upper()
    matching_reads = 0
    for v in votes:
        v_txt = v[0] if isinstance(v, (list, tuple)) else v.get("text", "")
        if v_txt and v_txt.strip().upper() == norm_final:
            matching_reads += 1

    if total_reads > 0:
        consensus_lbl = f"{matching_reads}/{total_reads} lecturas"
    elif norm_final and norm_final != "SIN PLACA":
        consensus_lbl = "1 sola lectura"
    else:
        consensus_lbl = "-"

    f_high = Font(name="Calibri", size=10, bold=True, color="006100")
    f_med = Font(name="Calibri", size=10, bold=False, color="7F6000")
    f_low = Font(name="Calibri", size=10, bold=True, color="C65911")
    f_none = Font(name="Calibri", size=10, bold=False, color="7F7F7F")

    fill_high = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    fill_med = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    fill_low = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    fill_none = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

    is_body_text = "carrocería" in str(reason).lower() or "marca" in str(reason).lower()

    if not norm_final or norm_final == "SIN PLACA" or status == "PLACA_NO_DETECTADA":
        return "SIN PLACA", consensus_lbl, fill_none, f_none
    elif is_body_text or status in ("REVISION_MANUAL", "FORMATO_INVALIDO"):
        return "REVISIÓN MANUAL", consensus_lbl, fill_low, f_low
    elif (matching_reads >= 2 and confidence_ocr >= 0.88) or (confidence_ocr >= 0.96 and status == "OK"):
        return "ALTA (SEGURA)", consensus_lbl, fill_high, f_high
    elif confidence_ocr >= 0.75 and status == "OK":
        return "MEDIA", consensus_lbl, fill_med, f_med
    else:
        return "DUDOSA", consensus_lbl, fill_low, f_low

class ExcelReportExporter:
    def __init__(self, db_path: str = "data/events.sqlite"):
        self.db_path = db_path

    def export_report(self, output_path: str = "reports/reporte_auditoria.xlsx", run_id: Optional[str] = None, include_duplicates: bool = False) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch non-duplicate rows for main sheet
        q_main = "SELECT * FROM events WHERE duplicate_of IS NULL"
        params_main = []
        if run_id:
            q_main += " AND processing_run_id = ?"
            params_main.append(run_id)
        q_main += " ORDER BY event_timestamp ASC"
        cursor.execute(q_main, params_main)
        rows_main = cursor.fetchall()

        # Fetch duplicate rows
        q_dup = "SELECT * FROM events WHERE duplicate_of IS NOT NULL"
        params_dup = []
        if run_id:
            q_dup += " AND processing_run_id = ?"
            params_dup.append(run_id)
        q_dup += " ORDER BY event_timestamp ASC"
        cursor.execute(q_dup, params_dup)
        rows_dup = cursor.fetchall()
        conn.close()

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Auditoria Transito"
        ws.views.sheetView[0].showGridLines = True

        self._populate_sheet(ws, rows_main, is_dup_sheet=False)

        if rows_dup and (include_duplicates or True):
            ws_dup = wb.create_sheet(title="Duplicados Descartados")
            ws_dup.views.sheetView[0].showGridLines = True
            self._populate_sheet(ws_dup, rows_dup, is_dup_sheet=True)

        wb.save(output_path)
        print(f"Excel report successfully generated at: {output_path} ({len(rows_main)} eventos únicos, {len(rows_dup)} duplicados)")
        return output_path

    def _populate_sheet(self, ws, rows, is_dup_sheet: bool = False):
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F4E79" if not is_dup_sheet else "7030A0", end_color="1F4E79" if not is_dup_sheet else "7030A0", fill_type="solid")
        cell_font = Font(name="Calibri", size=10)
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"), right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"), bottom=Side(style="thin", color="D9D9D9")
        )

        headers = [
            "Nº", "ID Evento", "Fecha y Hora", "Sentido", "Tipo",
            "Placa Raw", "Placa Validada", "Certeza Auditoría", "Consenso OCR", "Conf. OCR",
            "Estado Placa", "Foto Vehículo", "Recorte Placa", "Detalle / Reglas ANT", "Video Origen"
        ]
        if is_dup_sheet:
            headers.insert(2, "Duplicado De")

        ws.append(headers)
        ws.row_dimensions[1].height = 28
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align

        col_widths = {
            1: 6, 2: 14, 3: 20, 4: 12, 5: 12,
            6: 14, 7: 16, 8: 18, 9: 16, 10: 12,
            11: 16, 12: 22, 13: 18, 14: 30, 15: 30, 16: 18
        }
        for col_idx, width in col_widths.items():
            if col_idx <= len(headers):
                ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = width

        DATA_ROW_HEIGHT = 75
        IMG_MAX_HEIGHT = 90
        IMG_MAX_WIDTH = 150

        for idx, row in enumerate(rows, start=1):
            row_num = idx + 1
            ws.row_dimensions[row_num].height = DATA_ROW_HEIGHT
            col = 1

            ws.cell(row=row_num, column=col, value=idx).alignment = center_align
            col += 1
            ws.cell(row=row_num, column=col, value=row['event_id']).alignment = center_align
            col += 1

            if is_dup_sheet:
                dup_cell = ws.cell(row=row_num, column=col, value=row['duplicate_of'] or "-")
                dup_cell.alignment = center_align
                dup_cell.font = Font(name="Calibri", size=10, bold=True, color="7030A0")
                col += 1

            ws.cell(row=row_num, column=col, value=row['datetime_str']).alignment = center_align
            col += 1

            # Direction
            dir_cell = ws.cell(row=row_num, column=col, value=row['direction'])
            dir_cell.alignment = center_align
            if row['direction'] == 'ENTRADA':
                dir_cell.font = Font(name="Calibri", size=10, bold=True, color="006100")
                dir_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            else:
                dir_cell.font = Font(name="Calibri", size=10, bold=True, color="9C6500")
                dir_cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
            col += 1

            ws.cell(row=row_num, column=col, value=row['vehicle_type'].upper()).alignment = center_align
            col += 1
            ws.cell(row=row_num, column=col, value=row['plate_raw'] or "-").alignment = center_align
            col += 1

            # Plate Validated
            plt_val = row['plate_corrected'] or "SIN PLACA"
            p_cell = ws.cell(row=row_num, column=col, value=plt_val)
            p_cell.alignment = center_align
            p_cell.font = Font(name="Calibri", size=11, bold=True)
            col += 1

            # Audit Certainty & Consensus
            cert_lbl, cons_lbl, cert_fill, cert_font = compute_audit_certainty(
                plt_val, row['plate_status'], float(row['confidence_ocr'] or 0.0),
                row['ocr_votes'], row['plate_correction_reason']
            )
            cert_cell = ws.cell(row=row_num, column=col, value=cert_lbl)
            cert_cell.alignment = center_align
            cert_cell.fill = cert_fill
            cert_cell.font = cert_font
            col += 1

            cons_cell = ws.cell(row=row_num, column=col, value=cons_lbl)
            cons_cell.alignment = center_align
            col += 1

            # OCR Confidence
            conf_val = f"{row['confidence_ocr']:.1%}" if row['confidence_ocr'] > 0 else "-"
            ws.cell(row=row_num, column=col, value=conf_val).alignment = center_align
            col += 1

            # Status
            status_cell = ws.cell(row=row_num, column=col, value=row['plate_status'])
            status_cell.alignment = center_align
            if row['plate_status'] == 'OK':
                status_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
            elif row['plate_status'] in ('BAJA_CONFIANZA', 'REVISION_MANUAL'):
                status_cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
            col += 1

            # Images
            veh_col_idx = col
            col += 1
            plt_col_idx = col
            col += 1

            # Detail & Video source
            ws.cell(row=row_num, column=col, value=row['plate_correction_reason'] or "").alignment = left_align
            col += 1
            ws.cell(row=row_num, column=col, value=row['video_source']).alignment = left_align

            # Set fonts and borders
            for c in range(1, len(headers) + 1):
                cell = ws.cell(row=row_num, column=c)
                if not cell.font.bold:
                    cell.font = cell_font
                cell.border = thin_border

            # Embed Vehicle Image
            veh_path = row['vehicle_crop_path']
            if veh_path and os.path.exists(veh_path):
                try:
                    img_veh = OpenPyXLImage(veh_path)
                    img_veh.height = IMG_MAX_HEIGHT
                    img_veh.width = int(img_veh.width * (IMG_MAX_HEIGHT / max(1, img_veh.height)))
                    if img_veh.width > IMG_MAX_WIDTH:
                        img_veh.width = IMG_MAX_WIDTH
                    col_letter = openpyxl.utils.get_column_letter(veh_col_idx)
                    ws.add_image(img_veh, f"{col_letter}{row_num}")
                except Exception:
                    ws.cell(row=row_num, column=veh_col_idx, value="[Error img]")

            # Embed Plate Image
            plt_path = row['plate_crop_path']
            if plt_path and os.path.exists(plt_path):
                try:
                    img_plt = OpenPyXLImage(plt_path)
                    img_plt.height = int(IMG_MAX_HEIGHT * 0.65)
                    img_plt.width = int(img_plt.width * (img_plt.height / max(1, img_plt.height)))
                    if img_plt.width > 120:
                        img_plt.width = 120
                    col_letter = openpyxl.utils.get_column_letter(plt_col_idx)
                    ws.add_image(img_plt, f"{col_letter}{row_num}")
                except Exception:
                    ws.cell(row=row_num, column=plt_col_idx, value="[Error img]")

        # AutoFilter
        end_letter = openpyxl.utils.get_column_letter(len(headers))
        ws.auto_filter.ref = f"A1:{end_letter}{len(rows) + 1}"
