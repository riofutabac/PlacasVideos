import pytest
from src.ecuador_plate_validator import (
    normalize_plate_string,
    apply_ecuador_heuristics,
    PROVINCE_CODES,
    DIGIT_TO_LETTER,
    LETTER_TO_DIGIT
)

def test_normalize_plate_string():
    assert normalize_plate_string("pba-1234") == "PBA1234"
    assert normalize_plate_string("  PCW-2492  ") == "PCW2492"
    assert normalize_plate_string("ju-436a") == "JU436A"
    assert normalize_plate_string("ABC.123") == "ABC123"
    assert normalize_plate_string("") is None
    assert normalize_plate_string(None) is None
    assert normalize_plate_string("---") is None

def test_apply_heuristics_empty_or_none():
    res_none = apply_ecuador_heuristics(None, 0.9)
    assert res_none['plate_status'] == 'PLACA_NO_DETECTADA'
    assert res_none['plate_raw'] is None

    res_empty = apply_ecuador_heuristics("   ", 0.8)
    assert res_empty['plate_status'] == 'PLACA_NO_DETECTADA'

def test_apply_heuristics_too_short():
    res = apply_ecuador_heuristics("AB", 0.8)
    assert res['plate_status'] == 'BAJA_CONFIANZA'
    assert "demasiado corto" in res['plate_correction_reason']

def test_standard_ecuadorian_car_plates():
    # Standard car 4 digits
    res = apply_ecuador_heuristics("PCW-2492", 0.95)
    assert res['plate_corrected'] == "PCW2492"
    assert res['plate_status'] == "OK"
    assert "Pichincha" in res['plate_correction_reason']

    # Standard car 3 digits
    res3 = apply_ecuador_heuristics("PNU803", 0.88)
    assert res3['plate_corrected'] == "PNU803"
    assert res3['plate_status'] == "OK"
    assert "Pichincha" in res3['plate_correction_reason']

def test_digit_to_letter_corrections_in_prefix():
    # 0 instead of O / P
    res = apply_ecuador_heuristics("0CW2492", 0.90)
    assert res['plate_corrected'] == "OCW2492"
    assert "pos_0" in res['plate_correction_reason']

    # 8 instead of B
    res8 = apply_ecuador_heuristics("P8O4275", 0.90)
    assert res8['plate_corrected'] == "PBO4275"
    assert "pos_1" in res8['plate_correction_reason']

    # 1 instead of I
    res1 = apply_ecuador_heuristics("1AA1234", 0.90)
    assert res1['plate_corrected'] == "IAA1234"
    assert "Imbabura" in res1['plate_correction_reason']

def test_letter_to_digit_corrections_in_suffix():
    # O instead of 0
    res_o = apply_ecuador_heuristics("PFM83O5", 0.90)
    assert res_o['plate_corrected'] == "PFM8305"
    assert "pos_5: O->0" in res_o['plate_correction_reason']

    # S instead of 5
    res_s = apply_ecuador_heuristics("PFM830S", 0.90)
    assert res_s['plate_corrected'] == "PFM8305"

    # Z instead of 2
    res_z = apply_ecuador_heuristics("PCW249Z", 0.90)
    assert res_z['plate_corrected'] == "PCW2492"

    # I instead of 1
    res_i = apply_ecuador_heuristics("PBA123I", 0.90)
    assert res_i['plate_corrected'] == "PBA1231"

    # B instead of 8
    res_b = apply_ecuador_heuristics("PBA123B", 0.90)
    assert res_b['plate_corrected'] == "PBA1238"

    # G instead of 6
    res_g = apply_ecuador_heuristics("PBA123G", 0.90)
    assert res_g['plate_corrected'] == "PBA1236"

    # D instead of 0
    res_d = apply_ecuador_heuristics("PBA123D", 0.90)
    assert res_d['plate_corrected'] == "PBA1230"

def test_motorcycle_plate_corrections():
    # Moto: 2 letters + 3 digits + 1 letter (e.g. JU436A)
    res_moto = apply_ecuador_heuristics("JU436A", 0.92)
    assert res_moto['plate_corrected'] == "JU436A"
    assert res_moto['plate_status'] == "OK"
    assert "Santo Domingo" in res_moto['plate_correction_reason']

    # Moto with digit in last position (common OCR confusion: A <-> 4)
    res_m4 = apply_ecuador_heuristics("JU4364", 0.85)
    assert res_m4['plate_corrected'] == "JU436A"
    assert res_m4['plate_status'] == "OK"

    # Moto with digit in letter prefix
    res_m0 = apply_ecuador_heuristics("1U436A", 0.85)
    # 1U436A is length 6, norm[0] is digit, so not is_moto_format initially, treated as 6-char
    assert len(res_m0['plate_corrected']) == 6

def test_plate_status_by_confidence():
    # Low confidence < 0.60
    res_low = apply_ecuador_heuristics("PCW2492", 0.45)
    assert res_low['plate_status'] == "BAJA_CONFIANZA"

    # Non standard format with high confidence
    res_spec = apply_ecuador_heuristics("POLICIA1", 0.95)
    assert res_spec['plate_status'] == "FORMATO_ESPECIAL_O_MOTO"

def test_all_province_codes_exist():
    assert PROVINCE_CODES['P'] == 'Pichincha'
    assert PROVINCE_CODES['A'] == 'Azuay'
    assert PROVINCE_CODES['T'] == 'Tungurahua'
    assert PROVINCE_CODES['X'] == 'Cotopaxi'
    assert len(PROVINCE_CODES) >= 20
