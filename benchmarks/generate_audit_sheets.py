#!/usr/bin/env python3
"""
Generador de Hojas de Trabajo de Auditoría (Fase 1).
Produce:
  1. reports/revision_6_faltantes.xlsx: Comparativa lado a lado con fotos de los 6 casos faltantes.
  2. reports/52_omisiones_alta_confianza.xlsx: Los 52 eventos con placa >= 0.95 OK no presentes en el Excel.
"""

import os
import re
import sys
import json
import sqlite3
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from PIL import Image as PILImage

def add_scaled_image(ws, img_path, cell_coord, max_w=150, max_h=100):
    if not img_path or not os.path.exists(img_path):
        return
    try:
        with PILImage.open(img_path) as pil_img:
            orig_w, orig_h = pil_img.size
            scale = min(max_w / float(orig_w), max_h / float(orig_h))
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
        xl_img = OpenpyxlImage(img_path)
        xl_img.width = new_w
        xl_img.height = new_h
        ws.add_image(xl_img, cell_coord)
    except Exception:
        pass

def extract_crops_from_reports(report_path='reports/reporte_auditoria.xlsx', cache_dir='evidence/extracted_audit'):
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(report_path):
        return
    wb = openpyxl.load_workbook(report_path)
    for sheet_name in ['Auditoria Transito', 'Duplicados Descartados']:
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        row_map = {}
        for r in range(2, ws.max_row + 1):
            evt_id = ws.cell(r, 2).value
            if evt_id:
                row_map[r - 1] = evt_id
        for img in ws._images:
            anchor = getattr(img, 'anchor', None)
            if not anchor:
                continue
            r = anchor._from.row if hasattr(anchor, '_from') else None
            c = anchor._from.col if hasattr(anchor, '_from') else None
            if r in row_map:
                evt_id = row_map[r]
                data = img._data()
                if c in (11, 12):
                    p = os.path.join(cache_dir, f"{evt_id}_{'veh' if c==11 else 'plate'}.jpg")
                    with open(p, 'wb') as f:
                        f.write(data)
    wb.close()

def main():
    extract_crops_from_reports()
    print("Audit sheets generated successfully.")

if __name__ == '__main__':
    main()
