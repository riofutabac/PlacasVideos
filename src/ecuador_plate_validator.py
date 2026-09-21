"""
Ecuadorian License Plate Normalizer and Contextual Validator (ANT Heuristics).
Features:
- Provincial prefix awareness (P=Pichincha, G=Guayas, etc.).
- Letter/Digit position disambiguation (O<->0, I<->1, B<->8, S<->5, Z<->2).
- Non-destructive prior: Keeps plate_raw intact, provides plate_normalized, plate_corrected,
  and plate_correction_reason.
- Never discards an event if the plate is unusual or non-standard.
"""
import re
from typing import Tuple, Optional, Dict, Any

PROVINCE_CODES = {
    'A': 'Azuay', 'B': 'Bolivar', 'U': 'Canar', 'C': 'Carchi', 'X': 'Cotopaxi',
    'H': 'Chimborazo', 'O': 'El Oro', 'E': 'Esmeraldas', 'W': 'Galapagos',
    'I': 'Imbabura', 'L': 'Loja', 'R': 'Los Rios', 'M': 'Manabi', 'V': 'Morona Santiago',
    'N': 'Napo', 'S': 'Pastaza', 'P': 'Pichincha', 'Q': 'Orellana', 'K': 'Sucumbios',
    'T': 'Tungurahua', 'Z': 'Zamora Chinchipe', 'Y': 'Santa Elena', 'J': 'Santo Domingo'
}

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A', '7': 'T'
}

LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'D': '0', 'Q': '0', 'A': '4', 'T': '7', 'L': '1'
}

def normalize_plate_string(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    return cleaned if cleaned else None

def apply_ecuador_heuristics(
    raw_text: Optional[str],
    ocr_conf: float,
    needs_manual_review: bool = False
) -> Dict[str, Any]:
    """
    Applies Ecuadorian ANT heuristics gently as a prior.
    Returns:
      plate_raw: unmodified input
      plate_normalized: uppercase alphanumeric
      plate_corrected: contextual correction if applicable
      plate_correction_reason: justification string
      plate_status: OK, BAJA_CONFIANZA, REVISION_MANUAL, PLACA_NO_LEGIBLE, etc.
    """
    if not raw_text or len(raw_text.strip()) == 0:
        return {
            'plate_raw': None,
            'plate_normalized': None,
            'plate_corrected': None,
            'plate_correction_reason': None,
            'plate_status': 'PLACA_NO_DETECTADA'
        }

    norm = normalize_plate_string(raw_text)
    if not norm or len(norm) < 4:
        return {
            'plate_raw': raw_text,
            'plate_normalized': norm,
            'plate_corrected': norm,
            'plate_correction_reason': "Texto demasiado corto para placa estándar",
            'plate_status': 'BAJA_CONFIANZA'
        }

    reasons = []
    corrected_chars = list(norm)
    length = len(norm)

    # Ecuadorian motorcycle plate format: 2 letters + 3 digits + 1 letter (e.g. JU436A)
    # or 2 letters + 4 digits (e.g. AB1234)
    is_moto_format = (length == 6 and norm[0].isalpha() and norm[1].isalpha() and norm[2].isdigit())

    if is_moto_format:
        # pos 0, 1 should be letters
        for i in (0, 1):
            ch = corrected_chars[i]
            if ch.isdigit() and ch in DIGIT_TO_LETTER:
                corrected_chars[i] = DIGIT_TO_LETTER[ch]
                reasons.append(f"moto_pos_{i}: {ch}->{corrected_chars[i]}")
        # pos 2, 3, 4 should be digits
        for i in (2, 3, 4):
            ch = corrected_chars[i]
            if ch.isalpha() and ch in LETTER_TO_DIGIT:
                corrected_chars[i] = LETTER_TO_DIGIT[ch]
                reasons.append(f"moto_pos_{i}: {ch}->{corrected_chars[i]}")
        # pos 5 should be letter
        ch5 = corrected_chars[5]
        if ch5.isdigit() and ch5 in DIGIT_TO_LETTER:
            corrected_chars[5] = DIGIT_TO_LETTER[ch5]
            reasons.append(f"moto_pos_5: {ch5}->{corrected_chars[5]}")

    # Standard car/truck plate format: 3 letters + 3 or 4 digits (e.g. ABC1234 or ABC123)
    elif length in (6, 7):
        # First 3 should be letters
        for i in range(3):
            ch = corrected_chars[i]
            if ch.isdigit() and ch in DIGIT_TO_LETTER:
                corrected_chars[i] = DIGIT_TO_LETTER[ch]
                reasons.append(f"pos_{i}: {ch}->{corrected_chars[i]}")

        # Remaining should be digits
        for i in range(3, length):
            ch = corrected_chars[i]
            if ch.isalpha() and ch in LETTER_TO_DIGIT:
                corrected_chars[i] = LETTER_TO_DIGIT[ch]
                reasons.append(f"pos_{i}: {ch}->{corrected_chars[i]}")

    corrected_str = "".join(corrected_chars)

    # Validate province code
    first_letter = corrected_str[0] if len(corrected_str) > 0 and corrected_str[0].isalpha() else None
    province = PROVINCE_CODES.get(first_letter)
    if province:
        reasons.append(f"Provincia: {province}")

    # Determine status
    is_standard_car = bool(re.match(r'^[A-Z]{3}[0-9]{3,4}$', corrected_str))
    is_standard_moto = bool(re.match(r'^[A-Z]{2}[0-9]{3}[A-Z]$', corrected_str))
    is_standard_other = bool(re.match(r'^[A-Z]{2}[0-9]{4}$', corrected_str))
    is_standard = is_standard_car or is_standard_moto or is_standard_other

    # Non-plate text (body text / brand / logos / noise)
    is_non_plate = (
        len(corrected_str) < 5 or
        len(corrected_str) > 8 or
        not any(c.isalpha() for c in corrected_str) or
        not any(c.isdigit() for c in corrected_str)
    )

    if is_non_plate:
        status = 'REVISION_MANUAL'
        reasons.append("Formato no corresponde a placa ecuatoriana (posible texto de carrocería o marca)")
    elif needs_manual_review:
        status = 'REVISION_MANUAL'
        reasons.append("Ambigüedad en lectura: requiere revisión manual")
    elif is_standard and ocr_conf >= 0.75:
        status = 'OK'
    elif ocr_conf < 0.60:
        status = 'BAJA_CONFIANZA'
    elif is_standard:
        status = 'OK'
    else:
        status = 'REVISION_MANUAL'
        reasons.append("Formato no estándar")

    return {
        'plate_raw': raw_text,
        'plate_normalized': norm,
        'plate_corrected': corrected_str,
        'plate_correction_reason': "; ".join(reasons) if reasons else "Lectura directa sin cambios",
        'plate_status': status
    }
