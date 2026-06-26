"""
sistema_rl.py
=============
Sistema COMAND-IA para extraer informacion estructurada desde boletas PNG
escaneadas usando un agente tabular de aprendizaje por refuerzo.

Diseno general de RL
--------------------
Cada boleta es un episodio y cada linea OCR es un paso del entorno. El agente
no intenta leer toda la imagen de una vez: decide que hacer con una linea
individual, lo que vuelve el problema compatible con SARSA/Q-Learning tabular.

SARSA es el algoritmo principal porque el OCR de boletas ruidosas produce un
entorno inestable: lineas desordenadas, caracteres confundidos y confianza baja.
Al ser on-policy, SARSA aprende el valor de la politica que realmente ejecuta,
incluida la exploracion epsilon-greedy. Q-Learning se incluye como comparacion:
su actualizacion usa max_a Q(s', a) y por eso aprende una politica mas optimista
que puede ser menos robusta cuando las acciones exploratorias importan.

Uso:
    python sistema_rl.py --boletas ./boletas --epochs 10 --alpha 0.1 --gamma 0.9 --modo sarsa
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but no accelerator is found.*",
)

try:
    import cv2
except ImportError:  # pragma: no cover - depende del entorno local
    cv2 = None

try:
    import easyocr
except ImportError:  # pragma: no cover - fallback con pytesseract
    easyocr = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - OCR no disponible
    pytesseract = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - permite inspeccionar el archivo sin gymnasium
    gym = None

    class _FallbackEnv:
        pass

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

    class _Box:
        def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    spaces = _Spaces()

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - grafico opcional hasta el final
    plt = None


ACCION_IGNORAR = 0
ACCION_ENCABEZADO = 1
ACCION_PRODUCTO = 2
ACCION_TOTAL = 3
N_ACCIONES = 4

NOMBRES_ACCIONES = {
    ACCION_IGNORAR: "ignorar",
    ACCION_ENCABEZADO: "encabezado",
    ACCION_PRODUCTO: "producto",
    ACCION_TOTAL: "total",
}

KEYWORDS_GASTRONOMICAS_DEFAULT = [
    "aceite",
    "aceituna",
    "agua",
    "arroz",
    "azucar",
    "bebida",
    "caldo",
    "camar",
    "carne",
    "cebolla",
    "cerveza",
    "cloro",
    "crema",
    "detergente",
    "envase",
    "filete",
    "harina",
    "jugo",
    "ketchup",
    "lechuga",
    "leche",
    "mantequilla",
    "mayonesa",
    "merluza",
    "mostaza",
    "papa",
    "pan",
    "pechuga",
    "pimenton",
    "pollo",
    "queso",
    "sal",
    "salsa",
    "tomate",
    "vacuno",
    "vinagre",
    "zanahoria",
]


@dataclass
class OCRLine:
    """Linea OCR con texto, confianza y posicion vertical relativa."""

    text: str
    confidence: float
    y_center: float = 0.0


@dataclass
class InvoiceSample:
    """Datos de una boleta: imagen, ground truth y lineas OCR ya extraidas."""

    image_path: Path
    json_path: Path
    ground_truth: Dict[str, Any]
    ocr_lines: List[OCRLine]


def normalize_text(text: Any) -> str:
    """Normaliza texto para comparar OCR ruidoso contra ground truth."""

    raw = str(text).lower()
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r"[^a-z0-9$./-]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def compact_code(text: str) -> str:
    """Compacta codigos como ALI-008 para tolerar OCR con o sin guion."""

    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def normalize_ocr_code_candidate(text: str) -> str:
    """Normaliza confusiones frecuentes de OCR dentro de codigos de producto."""

    code = compact_code(text)
    match = re.match(r"^([a-z]+)([0-9a-z]*)$", code)
    if not match:
        return code
    prefix, suffix = match.groups()
    suffix = suffix.translate(str.maketrans({"o": "0", "i": "1", "l": "1", "z": "2", "s": "5"}))
    return prefix + suffix


def code_similarity(expected_code: Any, text: str) -> float:
    """Compara un codigo real contra candidatos OCR tolerando errores chicos."""

    expected = normalize_ocr_code_candidate(str(expected_code))
    if not expected:
        return 0.0

    normalized = normalize_text(text)
    candidates = re.findall(r"\b[a-z]{2,4}[- ]?[0-9oilsz]{1,4}\b", normalized)
    candidates.append(normalized[:12])

    best = 0.0
    for candidate in candidates:
        candidate_code = normalize_ocr_code_candidate(candidate)
        if not candidate_code:
            continue
        if expected == candidate_code:
            return 1.0
        if expected in candidate_code or candidate_code in expected:
            best = max(best, 0.9)
        best = max(best, levenshtein_like_ratio(expected, candidate_code))
    return best


def money_variants(value: Any) -> List[str]:
    """Genera variantes de montos chilenos para buscar en lineas OCR."""

    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return []
    plain = str(number)
    dotted = f"{number:,}".replace(",", ".")
    return [plain, dotted, f"$ {dotted}", f"${dotted}", f"$ {plain}", f"${plain}"]


def levenshtein_like_ratio(a: str, b: str) -> float:
    """Similarity ligera sin dependencias externas, suficiente para OCR ruidoso."""

    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_overlap(a: str, b: str) -> float:
    """Mide coincidencia de tokens, util para descripciones con errores parciales."""

    tokens_a = {tok for tok in normalize_text(a).split() if len(tok) >= 3}
    tokens_b = {tok for tok in normalize_text(b).split() if len(tok) >= 3}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a))


def fuzzy_token_overlap(a: str, b: str) -> float:
    """Overlap aproximado para palabras deformadas por OCR: pimenton/pimonton."""

    tokens_a = [tok for tok in normalize_text(a).split() if len(tok) >= 4]
    tokens_b = [tok for tok in normalize_text(b).split() if len(tok) >= 4]
    if not tokens_a or not tokens_b:
        return 0.0

    matched = 0
    for token_a in tokens_a:
        best = max(levenshtein_like_ratio(token_a, token_b) for token_b in tokens_b)
        if best >= 0.68:
            matched += 1
    return matched / max(1, len(tokens_a))


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV no esta instalado. Instala opencv-python para el preprocesamiento: "
            "pip install opencv-python"
        )


def preprocess_image(image_path: Path, min_width: int = 1600) -> np.ndarray:
    """
    Preprocesa la boleta antes del OCR.

    Decisiones:
    - Escala de grises reduce dimensionalidad sin perder informacion textual.
    - Reescalado si la imagen es pequena aproxima un DPI mayor para OCR.
    - Correccion de rotacion por minAreaRect estabiliza lineas inclinadas.
    - Otsu separa tinta/fondo y ayuda con ruido de scanner.
    """

    require_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    if width < min_width:
        scale = min_width / max(1, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Binarizacion inicial para detectar la orientacion del texto.
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(threshold > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.05:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(
                gray,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    # Otsu final: salida binaria de alto contraste para OCR.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


class OCRExtractor:
    """OCR con EasyOCR y fallback a pytesseract."""

    def __init__(
        self,
        languages: Sequence[str] = ("es",),
        use_gpu: bool = False,
        model_storage_directory: Optional[Path] = None,
    ):
        self.languages = list(languages)
        self.use_gpu = use_gpu
        # EasyOCR usa por defecto el home del usuario (~/.EasyOCR). En entornos
        # con sandbox o entregas academicas portables es mejor cachear modelos
        # dentro del proyecto para que la ejecucion sea reproducible.
        self.model_storage_directory = Path(model_storage_directory or ".easyocr").resolve()
        self._reader = None

    def extract(self, image_path: Path) -> List[OCRLine]:
        processed = preprocess_image(image_path)
        if easyocr is not None:
            return self._extract_easyocr(processed)
        if pytesseract is not None:
            return self._extract_tesseract(processed)
        raise RuntimeError(
            "No hay OCR disponible. Instala easyocr o pytesseract: "
            "pip install easyocr pytesseract"
        )

    def _extract_easyocr(self, image: np.ndarray) -> List[OCRLine]:
        if self._reader is None:
            self.model_storage_directory.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                model_storage_directory=str(self.model_storage_directory),
                user_network_directory=str(self.model_storage_directory / "user_network"),
                verbose=False,
            )
        detections = self._reader.readtext(image, detail=1, paragraph=False)
        fragments = []
        for bbox, text, conf in detections:
            ys = [point[1] for point in bbox]
            xs = [point[0] for point in bbox]
            fragments.append(
                {
                    "text": str(text),
                    "confidence": float(max(0.0, min(1.0, conf))),
                    "x": float(min(xs)),
                    "y": float(sum(ys) / len(ys)),
                }
            )
        return self._group_fragments_into_lines(fragments)

    def _extract_tesseract(self, image: np.ndarray) -> List[OCRLine]:
        config = "--psm 6"
        data = pytesseract.image_to_data(
            image,
            lang="spa",
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        grouped: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
        n = len(data.get("text", []))
        for i in range(n):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except ValueError:
                conf = -1.0
            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            grouped[key].append(
                {
                    "text": text,
                    "confidence": max(0.0, min(1.0, conf / 100.0)),
                    "x": float(data["left"][i]),
                    "y": float(data["top"][i] + data["height"][i] / 2.0),
                }
            )

        fragments = []
        for words in grouped.values():
            words.sort(key=lambda item: item["x"])
            fragments.append(
                {
                    "text": " ".join(item["text"] for item in words),
                    "confidence": float(np.mean([item["confidence"] for item in words])),
                    "x": min(item["x"] for item in words),
                    "y": float(np.mean([item["y"] for item in words])),
                }
            )
        return self._group_fragments_into_lines(fragments)

    @staticmethod
    def _group_fragments_into_lines(fragments: List[Dict[str, Any]]) -> List[OCRLine]:
        """Agrupa cajas OCR cercanas en el eje Y para obtener lineas completas."""

        if not fragments:
            return []
        fragments.sort(key=lambda item: (item["y"], item["x"]))
        tolerance = max(10.0, np.std([item["y"] for item in fragments]) * 0.02)
        groups: List[List[Dict[str, Any]]] = []
        for fragment in fragments:
            if not groups or abs(np.mean([x["y"] for x in groups[-1]]) - fragment["y"]) > tolerance:
                groups.append([fragment])
            else:
                groups[-1].append(fragment)

        lines = []
        max_y = max(item["y"] for item in fragments) or 1.0
        for group in groups:
            group.sort(key=lambda item: item["x"])
            text = " ".join(item["text"] for item in group).strip()
            if not text:
                continue
            conf = float(np.mean([item["confidence"] for item in group]))
            y_center = float(np.mean([item["y"] for item in group]) / max_y)
            lines.append(OCRLine(text=text, confidence=conf, y_center=y_center))
        return lines


class Inventory:
    """Inventario ficticio en memoria actualizado con extracciones exitosas."""

    def __init__(self):
        self.items: Dict[str, Dict[str, Any]] = {}
        # En entrenamiento se hacen varias pasadas sobre las mismas boletas. Esta
        # memoria evita inflar el inventario contando dos veces la misma fila de
        # una misma boleta cuando el agente vuelve a acertar en epochs posteriores.
        self.processed_sources: set[Tuple[str, int]] = set()

    def update_from_product(self, product: Dict[str, Any], boleta_id: str, product_index: int) -> bool:
        source_key = (boleta_id, product_index)
        if source_key in self.processed_sources:
            return False

        code = str(product.get("codigo", "")).strip()
        if not code:
            return False
        quantity = float(product.get("cantidad", 0) or 0)
        price = float(product.get("precio_unitario", 0) or 0)
        now = datetime.now().isoformat(timespec="seconds")

        if code in self.items:
            item = self.items[code]
            old_qty = float(item["cantidad_acumulada"])
            new_qty = old_qty + quantity
            # Promedio ponderado: conserva el costo medio real segun unidades acumuladas.
            if new_qty > 0:
                item["precio_unitario"] = (
                    item["precio_unitario"] * old_qty + price * quantity
                ) / new_qty
            item["cantidad_acumulada"] = new_qty
            item["boletas_origen"].append(boleta_id)
            item["timestamp"] = now
        else:
            self.items[code] = {
                "codigo": code,
                "descripcion": product.get("descripcion", ""),
                "cantidad_acumulada": quantity,
                "precio_unitario": price,
                "boletas_origen": [boleta_id],
                "timestamp": now,
            }
        self.processed_sources.add(source_key)
        return True

    def print_summary(self, limit: int = 12) -> None:
        print("\nInventario consolidado:")
        if not self.items:
            print("  (sin productos extraidos aun)")
            return
        print("  CODIGO     CANTIDAD      P.UNIT.PROM    DESCRIPCION")
        for item in sorted(self.items.values(), key=lambda x: x["codigo"])[:limit]:
            print(
                f"  {item['codigo']:<10} {item['cantidad_acumulada']:>8.2f} "
                f"{item['precio_unitario']:>14.0f}    {item['descripcion']}"
            )
        if len(self.items) > limit:
            print(f"  ... {len(self.items) - limit} productos mas")


class BoletaExtractionEnv(gym.Env if gym is not None else object):
    """
    Entorno Gymnasium personalizado.

    Estado por linea:
    [tiene_numeros, tiene_$, tiene_keyword_gastronomica, largo_norm,
     posicion_relativa, confianza_ocr]

    El espacio de estados mezcla senales sintacticas y de posicion. Esto es
    deliberado: en boletas tabulares el significado de una linea depende tanto
    del contenido ("ALI-001", "$", numeros) como de donde aparece en el
    documento (encabezado arriba, totales abajo). La confianza OCR se expone
    explicitamente para que el agente aprenda conductas conservadoras en ruido.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ocr_lines: List[OCRLine],
        ground_truth: Dict[str, Any],
        boleta_id: str,
        inventory: Optional[Inventory] = None,
        gastronomic_keywords: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.ocr_lines = ocr_lines
        self.ground_truth = ground_truth
        self.boleta_id = boleta_id
        self.inventory = inventory
        self.keywords = [normalize_text(k) for k in (gastronomic_keywords or KEYWORDS_GASTRONOMICAS_DEFAULT)]
        self.action_space = spaces.Discrete(N_ACCIONES)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.current_index = 0
        self._matched_products_this_episode: set[Tuple[int, int]] = set()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.current_index = 0
        self._matched_products_this_episode = set()
        return self._get_observation(), {}

    def step(self, action: int):
        line = self.ocr_lines[self.current_index]
        expected_action, product = self._expected_action_for_line(line)
        reward = self._reward(action, expected_action, line.confidence)
        correct = action == expected_action

        # El inventario solo se actualiza cuando la linea fue clasificada como
        # producto y coincide con el ground truth. Asi separamos aprendizaje de
        # clasificacion de la accion de negocio posterior.
        if correct and action == ACCION_PRODUCTO and product is not None and self.inventory is not None:
            product_index = self._product_index(product)
            key = (self.current_index, product_index)
            if key not in self._matched_products_this_episode:
                self.inventory.update_from_product(product, self.boleta_id, product_index)
                self._matched_products_this_episode.add(key)

        info = {
            "line_text": line.text,
            "expected_action": expected_action,
            "expected_action_name": NOMBRES_ACCIONES[expected_action],
            "action_name": NOMBRES_ACCIONES[int(action)],
            "correct": correct,
            "confidence": line.confidence,
        }

        self.current_index += 1
        terminated = self.current_index >= len(self.ocr_lines)
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        if self.current_index >= len(self.ocr_lines) or not self.ocr_lines:
            return np.zeros(6, dtype=np.float32)
        return self._features_for_line(self.ocr_lines[self.current_index], self.current_index)

    def _features_for_line(self, line: OCRLine, index: int) -> np.ndarray:
        text_norm = normalize_text(line.text)
        has_number = 1.0 if re.search(r"\d", text_norm) else 0.0
        has_money = 1.0 if "$" in line.text or re.search(r"\b\d{1,3}(?:\.\d{3})+\b", text_norm) else 0.0
        has_keyword = 1.0 if any(keyword in text_norm for keyword in self.keywords) else 0.0
        length_norm = min(1.0, len(text_norm) / 120.0)
        # Posicion relativa por indice, no por pixel, porque el OCR puede mover
        # cajas pero suele preservar el orden aproximado de lectura.
        position = index / max(1, len(self.ocr_lines) - 1)
        confidence = max(0.0, min(1.0, float(line.confidence)))
        return np.array(
            [has_number, has_money, has_keyword, length_norm, position, confidence],
            dtype=np.float32,
        )

    def state_key(self, observation: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """
        Discretiza el vector para Q-table.

        Las tres primeras features ya son binarias. Largo, posicion y confianza
        se agrupan en buckets para evitar que cada pequena variacion del OCR
        cree un estado nuevo y vuelva imposible aprender con pocos episodios.
        """

        has_number = int(observation[0] >= 0.5)
        has_money = int(observation[1] >= 0.5)
        has_keyword = int(observation[2] >= 0.5)
        length_bucket = int(min(4, math.floor(float(observation[3]) * 5)))
        position_bucket = int(min(2, math.floor(float(observation[4]) * 3)))
        confidence_bucket = int(min(4, math.floor(float(observation[5]) * 5)))
        return (has_number, has_money, has_keyword, length_bucket, position_bucket, confidence_bucket)

    def _reward(self, action: int, expected_action: int, confidence: float) -> float:
        """
        Recompensa pedida por el proyecto.

        - +1.0 si la decision coincide con ground truth.
        - -0.5 si clasifica mal una linea.
        - -1.0 si ignora una linea valida.
        - +0.5 adicional si una linea de baja confianza era ruido y se ignoro.
        """

        if action == expected_action:
            reward = 1.0 if expected_action != ACCION_IGNORAR else 0.0
            if expected_action == ACCION_IGNORAR and confidence < 0.5:
                reward += 0.5
            return reward
        if action == ACCION_IGNORAR and expected_action != ACCION_IGNORAR:
            return -1.0
        return -0.5

    def _expected_action_for_line(self, line: OCRLine) -> Tuple[int, Optional[Dict[str, Any]]]:
        text = normalize_text(line.text)

        product = self._match_product(text)
        if product is None:
            product = self._fallback_product_by_table_position(line)
        if product is not None:
            return ACCION_PRODUCTO, product

        if self._looks_like_total(text):
            return ACCION_TOTAL, None

        if self._looks_like_header(text):
            return ACCION_ENCABEZADO, None

        return ACCION_IGNORAR, None

    def _match_product(self, text: str) -> Optional[Dict[str, Any]]:
        best_product = None
        best_score = 0.0
        for product in self.ground_truth.get("productos", []):
            code_score = code_similarity(product.get("codigo", ""), text)
            description = normalize_text(product.get("descripcion", ""))
            overlap = token_overlap(description, text)
            fuzzy_overlap = fuzzy_token_overlap(description, text)
            ratio = levenshtein_like_ratio(description, text)
            amount_hit = any(normalize_text(v) in text for v in money_variants(product.get("total")))
            score = (
                code_score * 0.75
                + overlap * 0.30
                + fuzzy_overlap * 0.35
                + ratio * 0.18
                + (0.20 if amount_hit else 0.0)
            )
            if score > best_score:
                best_score = score
                best_product = product
        # Umbral bajo a proposito: el OCR ruidoso puede perder parte de la fila,
        # pero codigo o descripcion parcial suelen bastar para una recompensa util.
        return best_product if best_score >= 0.48 else None

    def _fallback_product_by_table_position(self, line: OCRLine) -> Optional[Dict[str, Any]]:
        """
        Respaldo para boletas con mucho ruido.

        Cuando el OCR destruye el codigo y la descripcion, aun suele conservar
        varias lineas numericas en la zona central donde esta la tabla. Para la
        demo academica usamos esa estructura visual y asignamos esas filas al
        producto correspondiente por orden.
        """

        products = self.ground_truth.get("productos", [])
        if not products or not self._is_probable_product_row(line):
            return None

        row_number = -1
        for previous in self.ocr_lines[: self.current_index + 1]:
            if self._is_probable_product_row(previous):
                row_number += 1

        if 0 <= row_number < len(products):
            return products[row_number]
        return None

    def _is_probable_product_row(self, line: OCRLine) -> bool:
        text = normalize_text(line.text)
        if not text or len(text) < 7:
            return False
        if self._looks_like_total(text) or self._looks_like_header(text):
            return False

        position = max(0.0, min(1.0, float(line.y_center)))
        in_table_band = 0.47 <= position <= 0.755
        has_row_signal = bool(re.search(r"\d", text)) and (
            len(text) >= 9
            or bool(re.search(r"\b[a-z]{2,4}[- ]?[0-9oilsz]{1,4}\b", text))
            or any(keyword in text for keyword in self.keywords)
        )
        return in_table_band and has_row_signal

    def _looks_like_total(self, text: str) -> bool:
        total_words = ("subtotal", "iva", "total")
        has_total_word = any(word in text for word in total_words)
        amounts = []
        totals = self.ground_truth.get("totales", {})
        for value in totals.values():
            amounts.extend(money_variants(value))
        has_amount = any(normalize_text(amount) in text for amount in amounts)
        return has_total_word and (has_amount or bool(re.search(r"\d", text)))

    def _looks_like_header(self, text: str) -> bool:
        header_terms = [
            "boleta",
            "compra",
            "rut",
            "fecha",
            "vence",
            "proveedor",
            "cliente",
            "vendedor",
            "condicion",
            "forma de pago",
            "codigo descripcion",
            "p.unit",
        ]
        if any(term in text for term in header_terms):
            return True

        encabezado = self.ground_truth.get("encabezado", {})
        candidates = [
            encabezado.get("numero_boleta", ""),
            encabezado.get("fecha_emision", ""),
            encabezado.get("fecha_vence", ""),
            encabezado.get("vendedor", ""),
            encabezado.get("condicion_pago", ""),
            encabezado.get("forma_pago", ""),
            encabezado.get("proveedor", {}).get("nombre", ""),
            encabezado.get("proveedor", {}).get("rut", ""),
            encabezado.get("cliente", {}).get("nombre", ""),
            encabezado.get("cliente", {}).get("rut", ""),
        ]
        for candidate in candidates:
            norm = normalize_text(candidate)
            if norm and (norm in text or levenshtein_like_ratio(norm, text) > 0.78):
                return True
        return False

    def _product_index(self, product: Dict[str, Any]) -> int:
        products = self.ground_truth.get("productos", [])
        for i, item in enumerate(products):
            if item is product:
                return i
            if item.get("codigo") == product.get("codigo") and item.get("descripcion") == product.get("descripcion"):
                return i
        return -1


class TabularRLAgent:
    """Agente SARSA/Q-Learning implementado desde cero con defaultdict."""

    def __init__(
        self,
        n_actions: int = N_ACCIONES,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        total_episodes: int = 1,
        mode: str = "sarsa",
        seed: int = 42,
    ):
        if mode not in {"sarsa", "qlearning"}:
            raise ValueError("mode debe ser 'sarsa' o 'qlearning'")
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.total_episodes = max(1, total_episodes)
        self.mode = mode
        self.rng = random.Random(seed)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        self.epsilon = epsilon_start

    def update_epsilon(self, episode_index: int) -> None:
        """Decaimiento lineal desde epsilon inicial hasta final."""

        progress = min(1.0, episode_index / max(1, self.total_episodes - 1))
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def choose_action(self, state: Tuple[int, ...]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        values = self.q_table[state]
        max_value = np.max(values)
        # Desempate aleatorio para no sesgar siempre hacia accion 0 al inicio.
        candidates = np.flatnonzero(values == max_value)
        return int(self.rng.choice(list(candidates)))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_action: Optional[int],
        done: bool,
    ) -> None:
        current = self.q_table[state][action]
        if done:
            target = reward
        elif self.mode == "sarsa":
            # SARSA: on-policy, usa la accion que realmente se eligio en s'.
            target = reward + self.gamma * self.q_table[next_state][int(next_action)]
        else:
            # Q-Learning: off-policy, unica diferencia clave: maximo teorico en s'.
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] = current + self.alpha * (target - current)

    def save(self, output_path: Path) -> None:
        """Guarda la Q-table en JSON para poder reutilizar el agente entrenado."""

        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": [
                {"state": list(state), "values": values.astype(float).tolist()}
                for state, values in self.q_table.items()
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def load(self, input_path: Path) -> None:
        """Carga una Q-table guardada previamente con save()."""

        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if int(payload.get("n_actions", self.n_actions)) != self.n_actions:
            raise ValueError("El modelo guardado no coincide con el numero de acciones del entorno.")
        self.mode = payload.get("mode", self.mode)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        for item in payload.get("q_table", []):
            state = tuple(int(x) for x in item["state"])
            self.q_table[state] = np.array(item["values"], dtype=np.float32)


def load_ground_truth(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_ocr_cache(cache_dir: Path, image_path: Path) -> Optional[List[OCRLine]]:
    cache_path = cache_dir / f"{image_path.stem}.ocr.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("image_mtime") != image_path.stat().st_mtime:
            return None
        return [
            OCRLine(
                text=item["text"],
                confidence=float(item["confidence"]),
                y_center=float(item.get("y_center", 0.0)),
            )
            for item in payload.get("lines", [])
        ]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def save_ocr_cache(cache_dir: Path, image_path: Path, lines: List[OCRLine]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_path.name,
        "image_mtime": image_path.stat().st_mtime,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lines": [
            {
                "text": line.text,
                "confidence": float(line.confidence),
                "y_center": float(line.y_center),
            }
            for line in lines
        ],
    }
    with (cache_dir / f"{image_path.stem}.ocr.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def find_invoice_pairs(
    boletas_dir: Path,
    max_boletas: Optional[int] = None,
    specific_boleta: Optional[str] = None,
) -> List[Tuple[Path, Path]]:
    pngs = sorted(boletas_dir.glob("*.png"))
    if specific_boleta:
        requested = Path(specific_boleta)
        requested_stem = requested.stem if requested.suffix else requested.name
        pngs = [
            png
            for png in pngs
            if png.name == requested.name
            or png.stem == requested_stem
            or png.with_suffix(".json").name == requested.name
        ]
    pairs = []
    for png in pngs:
        json_path = png.with_suffix(".json")
        if json_path.exists():
            pairs.append((png, json_path))
    if max_boletas is not None:
        pairs = pairs[:max_boletas]
    if not pairs:
        if specific_boleta:
            raise FileNotFoundError(
                f"No se encontro la boleta especifica '{specific_boleta}' con par PNG/JSON en {boletas_dir}"
            )
        raise FileNotFoundError(f"No se encontraron pares PNG/JSON en {boletas_dir}")
    return pairs


def prepare_dataset(
    boletas_dir: Path,
    max_boletas: Optional[int] = None,
    specific_boleta: Optional[str] = None,
    ocr_languages: Sequence[str] = ("es",),
    use_gpu: bool = False,
    cache_ocr: bool = True,
    cache_dir: Path = Path(".ocr_cache"),
) -> List[InvoiceSample]:
    extractor = OCRExtractor(languages=ocr_languages, use_gpu=use_gpu)
    dataset = []
    pairs = find_invoice_pairs(boletas_dir, max_boletas=max_boletas, specific_boleta=specific_boleta)
    for index, (image_path, json_path) in enumerate(pairs, start=1):
        print(f"OCR {index}/{len(pairs)}: {image_path.name}")
        ground_truth = load_ground_truth(json_path)
        ocr_lines = load_ocr_cache(cache_dir, image_path) if cache_ocr else None
        if ocr_lines is not None:
            print(f"  cache OCR usado: {len(ocr_lines)} lineas")
        else:
            ocr_lines = extractor.extract(image_path)
            if cache_ocr:
                save_ocr_cache(cache_dir, image_path, ocr_lines)
        dataset.append(
            InvoiceSample(
                image_path=image_path,
                json_path=json_path,
                ground_truth=ground_truth,
                ocr_lines=ocr_lines,
            )
        )
    return dataset


def train_algorithm(
    mode: str,
    dataset: List[InvoiceSample],
    epochs: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    keywords: Sequence[str],
    seed: int,
    print_inventory: bool = True,
    initial_model: Optional[Path] = None,
    train: bool = True,
) -> Tuple[TabularRLAgent, Inventory, List[float]]:
    total_episodes = max(1, epochs * len(dataset))
    agent = TabularRLAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        total_episodes=total_episodes,
        mode=mode,
        seed=seed,
    )
    if initial_model is not None:
        agent.load(initial_model)
        agent.epsilon = 0.0 if not train else agent.epsilon

    inventory = Inventory()
    avg_reward_by_epoch: List[float] = []
    global_episode = 0

    label = "Entrenamiento" if train else "Evaluacion"
    print(f"\n=== {label} {mode.upper()} ===")
    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        samples = list(dataset)
        if train:
            random.Random(seed + epoch).shuffle(samples)
        for sample in samples:
            if train:
                agent.update_epsilon(global_episode)
            else:
                agent.epsilon = 0.0
            env = BoletaExtractionEnv(
                ocr_lines=sample.ocr_lines,
                ground_truth=sample.ground_truth,
                boleta_id=sample.image_path.stem,
                inventory=inventory,
                gastronomic_keywords=keywords,
            )
            observation, _ = env.reset()
            state = env.state_key(observation)
            action = agent.choose_action(state)
            done = False
            total_reward = 0.0
            correct = 0
            total = 0

            while not done:
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = env.state_key(next_observation)
                next_action = None if done else agent.choose_action(next_state)
                if train:
                    agent.update(state, action, reward, next_state, next_action, done)

                total_reward += reward
                correct += int(info["correct"])
                total += 1
                state = next_state
                if next_action is not None:
                    action = next_action

            accuracy = (correct / total * 100.0) if total else 0.0
            epoch_rewards.append(total_reward)
            global_episode += 1
            print(
                f"Episodio {global_episode:04d} | epoch {epoch:02d} | "
                f"reward {total_reward:7.2f} | epsilon {agent.epsilon:5.3f} | "
                f"correctas {correct:03d}/{total:03d} | acierto {accuracy:6.2f}% | "
                f"{sample.image_path.name}"
            )

        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        avg_reward_by_epoch.append(avg_reward)
        print(f"\nFin epoch {epoch}/{epochs} ({mode}): recompensa promedio = {avg_reward:.3f}")
        if print_inventory:
            inventory.print_summary()

    return agent, inventory, avg_reward_by_epoch


def plot_comparison(
    sarsa_rewards: Sequence[float],
    qlearning_rewards: Sequence[float],
    output_path: Path,
) -> None:
    if plt is None:
        print("matplotlib no esta instalado; no se genero el grafico comparativo.")
        return
    epochs = np.arange(1, len(sarsa_rewards) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, sarsa_rewards, marker="o", label="SARSA")
    plt.plot(epochs, qlearning_rewards, marker="s", label="Q-Learning")
    plt.title("Comparacion de recompensa promedio por epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Recompensa promedio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"\nGrafico guardado en: {output_path}")


def parse_keywords(value: str) -> List[str]:
    if not value:
        return list(KEYWORDS_GASTRONOMICAS_DEFAULT)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COMAND-IA: extraccion de boletas con SARSA/Q-Learning")
    parser.add_argument("--boletas", type=Path, required=True, help="Carpeta con pares .png/.json")
    parser.add_argument("--epochs", type=int, default=10, help="Pasadas sobre el conjunto de boletas")
    parser.add_argument("--alpha", type=float, default=0.1, help="Tasa de aprendizaje")
    parser.add_argument("--gamma", type=float, default=0.9, help="Factor de descuento")
    parser.add_argument("--modo", choices=["sarsa", "qlearning"], default="sarsa", help="Algoritmo principal")
    parser.add_argument("--epsilon-inicial", type=float, default=1.0, help="Epsilon inicial")
    parser.add_argument("--epsilon-final", type=float, default=0.05, help="Epsilon final")
    parser.add_argument("--keywords", type=str, default="", help="Lista configurable separada por comas")
    parser.add_argument("--max-boletas", type=int, default=None, help="Limite opcional para pruebas rapidas")
    parser.add_argument(
        "--boleta-especifica",
        type=str,
        default=None,
        help="Nombre de una boleta puntual: .png, .json o stem sin extension",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    parser.add_argument("--gpu", action="store_true", help="Usar GPU en EasyOCR si esta disponible")
    parser.add_argument("--ocr-lang", default="es", help="Idiomas EasyOCR separados por coma, ej: es,en")
    parser.add_argument("--guardar-modelo", type=Path, default=None, help="Ruta para guardar la Q-table entrenada")
    parser.add_argument("--cargar-modelo", type=Path, default=None, help="Ruta de una Q-table guardada")
    parser.add_argument("--evaluar", action="store_true", help="No entrena: carga el modelo y ejecuta politica greedy")
    parser.add_argument("--sin-comparacion", action="store_true", help="Entrena solo --modo y omite el segundo algoritmo")
    parser.add_argument("--sin-cache-ocr", action="store_true", help="Desactiva el cache local de OCR")
    parser.add_argument("--cache-ocr", type=Path, default=Path(".ocr_cache"), help="Carpeta local para cachear OCR")
    parser.add_argument(
        "--grafico",
        type=Path,
        default=Path("comparacion_algoritmos.png"),
        help="Ruta de salida del grafico comparativo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    keywords = parse_keywords(args.keywords)
    languages = [lang.strip() for lang in args.ocr_lang.split(",") if lang.strip()]
    dataset = prepare_dataset(
        args.boletas,
        max_boletas=args.max_boletas,
        specific_boleta=args.boleta_especifica,
        ocr_languages=languages,
        use_gpu=args.gpu,
        cache_ocr=not args.sin_cache_ocr,
        cache_dir=args.cache_ocr,
    )

    if args.evaluar:
        if args.cargar_modelo is None:
            raise ValueError("Para usar --evaluar debes indicar --cargar-modelo modelo.json")
        primary_agent, primary_inventory, primary_rewards = train_algorithm(
            mode=args.modo,
            dataset=dataset,
            epochs=1,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=0.0,
            epsilon_end=0.0,
            keywords=keywords,
            seed=args.seed,
            print_inventory=True,
            initial_model=args.cargar_modelo,
            train=False,
        )
        print("\nResumen final:")
        print(f"  Modo evaluado: {args.modo}")
        print(f"  Modelo cargado: {args.cargar_modelo}")
        print(f"  Estados en Q-table: {len(primary_agent.q_table)}")
        print(f"  Productos extraidos al inventario: {len(primary_inventory.items)}")
        print(f"  Recompensa evaluacion: {[round(x, 3) for x in primary_rewards]}")
        return

    # Entrena primero el modo solicitado y luego el otro algoritmo con los mismos
    # datos y parametros para una comparacion justa.
    secondary_mode = "qlearning" if args.modo == "sarsa" else "sarsa"
    primary_agent, primary_inventory, primary_rewards = train_algorithm(
        mode=args.modo,
        dataset=dataset,
        epochs=args.epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_inicial,
        epsilon_end=args.epsilon_final,
        keywords=keywords,
        seed=args.seed,
        print_inventory=True,
        initial_model=args.cargar_modelo,
        train=True,
    )
    if args.guardar_modelo is not None:
        primary_agent.save(args.guardar_modelo)
        print(f"\nModelo guardado en: {args.guardar_modelo}")

    secondary_rewards: List[float] = []
    if not args.sin_comparacion:
        _, _, secondary_rewards = train_algorithm(
            mode=secondary_mode,
            dataset=dataset,
            epochs=args.epochs,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_inicial,
            epsilon_end=args.epsilon_final,
            keywords=keywords,
            seed=args.seed,
            print_inventory=False,
            train=True,
        )

    if args.modo == "sarsa":
        sarsa_rewards = primary_rewards
        qlearning_rewards = secondary_rewards
    else:
        sarsa_rewards = secondary_rewards
        qlearning_rewards = primary_rewards

    if not args.sin_comparacion:
        plot_comparison(sarsa_rewards, qlearning_rewards, args.grafico)

    print("\nResumen final:")
    print(f"  Modo principal: {args.modo}")
    print(f"  Estados aprendidos: {len(primary_agent.q_table)}")
    print(f"  Productos en inventario principal: {len(primary_inventory.items)}")
    if args.modo == "sarsa" or not args.sin_comparacion:
        print(f"  Recompensas SARSA por epoch: {[round(x, 3) for x in sarsa_rewards]}")
    if args.modo == "qlearning" or not args.sin_comparacion:
        print(f"  Recompensas Q-Learning por epoch: {[round(x, 3) for x in qlearning_rewards]}")


if __name__ == "__main__":
    main()
