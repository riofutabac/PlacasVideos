"""
Professional Excel Report Exporter for ALPR Pipeline.
Generates an audit-ready Excel workbook (.xlsx) from SQLite canonical database.
Features:
- Physically embeds vehicle and plate crop images directly inside worksheet cells.
- Auto-sizes row heights and column widths to fit images perfectly.
- Clean header styles, conditional colors, and table auto-filters.
- Fully decoupled: Can regenerate reports anytime without touching video files.
"""
import os
import sqlite3
from typing import Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.drawing.image import Image as OpenPyXLImage
from PIL import Image as PILImage

class ExcelReportExporter:
    def __init__(self, db_path: str = "data/events.sqlite"):
        self.db_path = db_path

    def export_report(self, output_path: str = "reports/reporte_auditoria.xlsx", run_id: Optional[str] = None, include_duplicates: bool = False) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Connect to DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM events"
        params = []
        conds = []
        if run_id:
            conds.append("processing_run_id = ?")
            params.append(run_id)
        if not include_duplicates:
            conds.append("duplicate_of IS NULL")
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY event_timestamp ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Auditoria Transito"
        ws.views.sheetView[0].showGridLines = True

        # Styles
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        cell_font = Font(name="Calibri", size=10)
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9")
        )

        headers = [
            "Nº", "ID Evento", "Fecha y Hora", "Sentido", "Tipo",
            "Placa Raw", "Placa Validada", "Estado Placa", "Conf. OCR",
            "Foto Vehículo", "Recorte Placa", "Detalle / Reglas ANT", "Video Origen"
        ]

        # Write Headers
        ws.append(headers)
        ws.row_dimensions[1].height = 28
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align

        # Column widths
        col_widths = {
            1: 6, 2: 14, 3: 20, 4: 12, 5: 12,
            6: 14, 7: 16, 8: 16, 9: 12,
            10: 22, 11: 18, 12: 28, 13: 30
        }
        for col_idx, width in col_widths.items():
            ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = width

        # Row height for data rows (to fit images nicely)
        DATA_ROW_HEIGHT = 75
        IMG_MAX_HEIGHT = 90
        IMG_MAX_WIDTH = 150

        for idx, row in enumerate(rows, start=1):
            row_num = idx + 1
            ws.row_dimensions[row_num].height = DATA_ROW_HEIGHT

            # Values
            ws.cell(row=row_num, column=1, value=idx).alignment = center_align
            ws.cell(row=row_num, column=2, value=row['event_id']).alignment = center_align
            ws.cell(row=row_num, column=3, value=row['datetime_str']).alignment = center_align
            
            # Direction with styling
            dir_cell = ws.cell(row=row_num, column=4, value=row['direction'])
            dir_cell.alignment = center_align
            if row['direction'] == 'ENTRADA':
                dir_cell.font = Font(name="Calibri", size=10, bold=True, color="006100")
                dir_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            else:
                dir_cell.font = Font(name="Calibri", size=10, bold=True, color="9C6500")
                dir_cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")

            ws.cell(row=row_num, column=5, value=row['vehicle_type'].upper()).alignment = center_align
            ws.cell(row=row_num, column=6, value=row['plate_raw'] or "-").alignment = center_align
            
            # Corrected plate
            plate_cell = ws.cell(row=row_num, column=7, value=row['plate_corrected'] or "SIN PLACA")
            plate_cell.alignment = center_align
            plate_cell.font = Font(name="Calibri", size=11, bold=True)

            # Status
            status_cell = ws.cell(row=row_num, column=8, value=row['plate_status'])
            status_cell.alignment = center_align
            if row['plate_status'] == 'OK':
                status_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
            elif row['plate_status'] == 'BAJA_CONFIANZA':
                status_cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

            # OCR Confidence
            conf_val = f"{row['confidence_ocr']:.1%}" if row['confidence_ocr'] > 0 else "-"
            ws.cell(row=row_num, column=9, value=conf_val).alignment = center_align

            # Detail & Video source
            ws.cell(row=row_num, column=12, value=row['plate_correction_reason'] or "").alignment = left_align
            ws.cell(row=row_num, column=13, value=row['video_source']).alignment = left_align

            # Set fonts and borders
            for c in range(1, len(headers) + 1):
                cell = ws.cell(row=row_num, column=c)
                if not cell.font.bold:
                    cell.font = cell_font
                cell.border = thin_border

            # Embed Vehicle Image (Column 10)
            veh_path = row['vehicle_crop_path']
            if veh_path and os.path.exists(veh_path):
                try:
                    img_veh = OpenPyXLImage(veh_path)
                    img_veh.height = IMG_MAX_HEIGHT
                    img_veh.width = int(img_veh.width * (IMG_MAX_HEIGHT / max(1, img_veh.height)))
                    if img_veh.width > IMG_MAX_WIDTH:
                        img_veh.width = IMG_MAX_WIDTH
                    img_cell = f"J{row_num}"
                    ws.add_image(img_veh, img_cell)
                except Exception as e:
                    ws.cell(row=row_num, column=10, value="[Error img]")

            # Embed Plate Image (Column 11)
            plt_path = row['plate_crop_path']
            if plt_path and os.path.exists(plt_path):
                try:
                    img_plt = OpenPyXLImage(plt_path)
                    img_plt.height = int(IMG_MAX_HEIGHT * 0.65)
                    img_plt.width = int(img_plt.width * (img_plt.height / max(1, img_plt.height)))
                    if img_plt.width > 120:
                        img_plt.width = 120
                    plt_cell = f"K{row_num}"
                    ws.add_image(img_plt, plt_cell)
                except Exception:
                    ws.cell(row=row_num, column=11, value="[Error img]")

        # Add AutoFilter
        ws.auto_filter.ref = f"A1:M{len(rows) + 1}"
        wb.save(output_path)
        print(f"Excel report successfully generated at: {output_path} ({len(rows)} events)")
        return output_path
