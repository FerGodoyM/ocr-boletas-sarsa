"""
generador_boletas.py
====================
Genera N boletas de proveedor chilenas con:
  - Productos aleatorios del rubro gastronómico
  - Proveedores y clientes distintos cada vez
  - 4 niveles de "escáner": limpia / poco ruido / medio / mucho ruido
  - Defectos visuales: rotación, manchas, sombras, desenfoque, arrugas, tinta baja

Uso:
    python3 generador_boletas.py --cantidad 20 --salida ./boletas
    python3 generador_boletas.py          # genera 12 con defaults
"""

import argparse, os, random, math, json
from datetime import datetime
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as rl_canvas

# ══════════════════════════════════════════════════════════════════════════════
#  DATOS ALEATORIOS
# ══════════════════════════════════════════════════════════════════════════════

PROVEEDORES = [
    ("Distribuidora Alimentos del Norte Ltda.", "76.432.891-5", "Av. Balmaceda 1420, La Serena"),
    ("Comercial Frigorífico Central S.A.",      "78.234.567-3", "Ruta 5 Norte Km 12, Coquimbo"),
    ("Mayorista La Cosecha SpA",                "77.891.234-6", "Los Aromos 890, Ovalle"),
    ("Distribuidora El Granero Ltda.",          "76.123.456-7", "Av. del Mar 345, La Serena"),
    ("Proveedor Gastro Sur S.A.",               "78.567.890-2", "Panamericana 2210, Vicuña"),
    ("Alimentos y Bebidas Coquimbo SpA",        "77.345.678-9", "El Peñón 110, Coquimbo"),
]

CLIENTES = [
    ("Restaurante El Rincón del Sabor",   "12.345.678-9", "Los Carrera 523, La Serena"),
    ("Café y Cocina La Esquina",          "15.678.234-1", "Av. Francisco de Aguirre 840, La Serena"),
    ("Picada Don Memo",                   "11.234.567-8", "Calle Prat 120, Coquimbo"),
    ("Emprendimiento Sabores del Mar",    "14.567.890-3", "Los Pescadores 45, Tongoy"),
    ("Bistró Los Aromos",                 "16.789.012-4", "Av. El Santo 312, La Serena"),
    ("Fuente de Soda La Estrella",        "13.456.789-5", "O'Higgins 78, Ovalle"),
    ("Cevichería Puerto Chico",           "17.890.123-6", "Av. Costanera 560, Coquimbo"),
]

VENDEDORES = [
    "Carlos Mendoza Ríos", "Patricia Soto Vega", "Rodrigo Fuentes León",
    "Gabriela Muñoz Torres", "Andrés Castillo Díaz", "Valentina Rojas Pinto",
]

PRODUCTOS = [
    # (código, descripción, unidad, precio_min, precio_max, cant_min, cant_max)
    ("ALI-001", "Aceite vegetal canola 1 litro",          "Unid.",  1800,  2500,  6, 36),
    ("ALI-002", "Harina de trigo 50 kg (saco)",           "Saco",  15000, 22000,  1,  6),
    ("ALI-003", "Arroz grado 1 - 25 kg",                  "Saco",  11000, 16000,  1,  8),
    ("ALI-004", "Azúcar blanca 50 kg",                    "Saco",  19000, 25000,  1,  4),
    ("ALI-005", "Sal fina yodada 1 kg",                   "Unid.",   400,   600, 6, 24),
    ("ALI-006", "Vinagre de vino tinto 750ml",            "Unid.",   950,  1400,  6, 24),
    ("ALI-007", "Salsa de tomate 3 kg (tarro)",           "Unid.",  3200,  4500,  3, 12),
    ("ALI-008", "Aceitunas verdes 3 kg (balde)",          "Unid.",  7800,  9800,  2,  8),
    ("BEV-001", "Jugo natural naranja 1 lt (caja x12)",   "Caja",   8000, 11000,  2, 10),
    ("BEV-002", "Agua mineral sin gas 500ml (caja x24)",  "Caja",   4800,  6500,  4, 16),
    ("BEV-003", "Bebida cola 1.5 lt (caja x12)",          "Caja",   9500, 12500,  2,  8),
    ("BEV-004", "Cerveza lager 350ml (caja x24)",         "Caja",  18000, 24000,  2,  6),
    ("CAR-001", "Filete de merluza congelado 5 kg",       "Bandej",10000, 14000,  2, 10),
    ("CAR-002", "Pechuga de pollo sin hueso 5 kg",        "Bandej", 9500, 13000,  3, 12),
    ("CAR-003", "Carne molida vacuno 5 kg",               "Bandej",12000, 17000,  2,  8),
    ("CAR-004", "Pulpo cocido 1 kg (bolsa)",              "Bolsa", 11000, 16000,  2,  6),
    ("CAR-005", "Camarón ecuatoriano crudo 1 kg",         "Bolsa",  9000, 13500,  3,  9),
    ("VEG-001", "Tomate calibre 1 caja 10 kg",            "Caja",   5500,  8500,  2,  8),
    ("VEG-002", "Cebolla blanca caja 20 kg",              "Caja",   5000,  7500,  2,  6),
    ("VEG-003", "Papa blanca bolsa 25 kg",                "Bolsa",  7000, 10000,  2,  8),
    ("VEG-004", "Zanahoria bolsa 10 kg",                  "Bolsa",  3500,  5500,  2,  8),
    ("VEG-005", "Lechuga costina caja 12 unid.",          "Caja",   4800,  7200,  1,  5),
    ("VEG-006", "Pimentón rojo kg",                       "Kg",     1800,  2800,  3, 12),
    ("LAC-001", "Leche entera UHT 1 lt (caja x12)",       "Caja",   9000, 12000,  2,  8),
    ("LAC-002", "Queso gauda 1 kg (bloque)",              "Unid.",  5500,  7800,  2,  8),
    ("LAC-003", "Crema de leche 200ml (caja x24)",        "Caja",  14000, 18000,  1,  4),
    ("LAC-004", "Mantequilla 250g (caja x20)",            "Caja",  22000, 28000,  1,  3),
    ("PAN-001", "Pan de molde 600g (caja x12)",           "Caja",  12000, 16000,  2,  6),
    ("PAN-002", "Pan pita 300g (bolsa x6)",               "Bolsa",  2800,  4200,  4, 12),
    ("CON-001", "Caldo de vacuno en polvo 1 kg",          "Unid.",  4500,  6500,  2,  8),
    ("CON-002", "Mayonesa 3 kg (balde)",                  "Unid.",  8500, 11500,  2,  6),
    ("CON-003", "Ketchup 3 kg (balde)",                   "Unid.",  7000, 10000,  2,  6),
    ("CON-004", "Mostaza 1 kg (frasco)",                  "Unid.",  3200,  4800,  2,  8),
    ("DES-001", "Detergente industrial 5 lt",             "Unid.",  6800,  9500,  1,  4),
    ("DES-002", "Cloro multiusos 5 lt",                   "Unid.",  3500,  5500,  2,  6),
    ("DES-003", "Papel absorbente caja x4 rollos",        "Caja",   8500, 12000,  1,  4),
    ("EMB-001", "Bolsa plástica biodegradable (paq x100)","Paq.",   2800,  4200,  2,  8),
    ("EMB-002", "Envase térmico 500ml (caja x50)",        "Caja",   9500, 14000,  1,  4),
]

CONDICIONES = ["Contado", "Crédito 15 días", "Crédito 30 días", "Crédito 60 días"]
FORMAS_PAGO = ["Transferencia bancaria", "Cheque a 30 días", "Efectivo", "Débito automático"]

def fmt(n):
    return f"{int(n):,}".replace(",", ".")

def rut_random():
    base = random.randint(5_000_000, 25_000_000)
    digits = [int(d) for d in str(base)]
    factors = [2, 3, 4, 5, 6, 7]
    s = sum(d * factors[i % 6] for i, d in enumerate(reversed(digits)))
    remainder = 11 - (s % 11)
    dv = {11: "0", 10: "K"}.get(remainder, str(remainder))
    s = str(base)
    return f"{s[:-6]}.{s[-6:-3]}.{s[-3:]}-{dv}"

# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR DE PDF
# ══════════════════════════════════════════════════════════════════════════════

def generar_pdf(filepath, numero):
    W, H = A4
    c = rl_canvas.Canvas(filepath, pagesize=A4)

    prov  = random.choice(PROVEEDORES)
    cli   = random.choice(CLIENTES)
    vend  = random.choice(VENDEDORES)
    cond  = random.choice(CONDICIONES)
    forma = random.choice(FORMAS_PAGO)

    dia   = random.randint(1, 28)
    mes   = random.randint(1, 12)
    anio  = 2025 + random.randint(0, 1)
    fecha = f"{dia:02d}/{mes:02d}/{anio}"
    mes2  = mes + 1 if mes < 12 else 1
    anio2 = anio if mes < 12 else anio + 1
    vence = f"{dia:02d}/{mes2:02d}/{anio2}"

    n_productos = random.randint(4, 13)
    seleccion = random.sample(PRODUCTOS, n_productos)
    filas = []
    for cod, desc, unidad, pmin, pmax, cmin, cmax in seleccion:
        precio = random.randrange(pmin, pmax, 50)
        cant   = random.randint(cmin, cmax)
        total  = precio * cant
        filas.append((cod, desc, str(cant), unidad, fmt(precio), fmt(total), total))

    subtotal = sum(f[6] for f in filas)
    iva      = round(subtotal * 0.19)
    total_doc = subtotal + iva

    def txt(x, y, s, sz=9, bold=False, col=colors.black, align="left"):
        c.setFillColor(col)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", sz)
        {"right": c.drawRightString, "center": c.drawCentredString}.get(align, c.drawString)(x, y, str(s))

    def ln(x1, y1, x2, y2, w=0.5, col=colors.black):
        c.setStrokeColor(col); c.setLineWidth(w); c.line(x1, y1, x2, y2)

    AZUL = colors.HexColor("#1a3a5c")

    # Encabezado proveedor
    txt(2*cm, H-2.2*cm, prov[0], 11, True)
    txt(2*cm, H-2.9*cm, f"RUT: {prov[1]}")
    txt(2*cm, H-3.5*cm, prov[2])
    txt(2*cm, H-4.1*cm, f"Tel: +56 51 2 {random.randint(200000,399999)}  |  ventas@proveedor.cl")
    txt(2*cm, H-4.7*cm, "Giro: Venta al por mayor de alimentos y bebidas")

    # Caja número
    bx, by, bw, bh = 13.5*cm, H-5.2*cm, 5.2*cm, 3.2*cm
    c.setStrokeColor(AZUL); c.setLineWidth(1.5); c.rect(bx, by, bw, bh)
    txt(bx+bw/2, by+bh-0.7*cm,  "BOLETA DE COMPRA", 10, True, AZUL, "center")
    txt(bx+bw/2, by+bh-1.35*cm, f"N° {numero:06d}", 13, True, AZUL, "center")
    ln(bx+0.3*cm, by+1.55*cm, bx+bw-0.3*cm, by+1.55*cm, col=AZUL)
    txt(bx+0.4*cm, by+1.1*cm,  "Fecha emisión:", 8, True)
    txt(bx+0.4*cm, by+0.55*cm, fecha)
    txt(bx+3.0*cm, by+1.1*cm,  "Vence:", 8, True)
    txt(bx+3.0*cm, by+0.55*cm, vence)

    ln(2*cm, H-5.5*cm, W-2*cm, H-5.5*cm, 1, AZUL)

    # Datos cliente
    y = H-6.0*cm
    txt(2*cm, y,           "DATOS DEL CLIENTE", 9, True, AZUL)
    for label, val, lx, vx in [
        ("Razón social:", cli[0],  2*cm, 5.0*cm),
        ("RUT:",          cli[1],  2*cm, 5.0*cm),
        ("Dirección:",    cli[2],  2*cm, 5.0*cm),
        ("Giro:",         "Restaurante / Servicios de alimentación", 2*cm, 5.0*cm),
    ]:
        offset = (["Razón social:","RUT:","Dirección:","Giro:"].index(label)+1)*0.55*cm
        txt(lx, y-offset, label, 8, True); txt(vx, y-offset, val)

    txt(11*cm, y-0.55*cm, "Condición de pago:", 8, True); txt(15.2*cm, y-0.55*cm, cond)
    txt(11*cm, y-1.10*cm, "Forma de pago:",     8, True); txt(15.2*cm, y-1.10*cm, forma)
    txt(11*cm, y-1.65*cm, "Vendedor:",          8, True); txt(15.2*cm, y-1.65*cm, vend)

    ln(2*cm, H-9.2*cm, W-2*cm, H-9.2*cm, 0.5, colors.HexColor("#cccccc"))

    # Tabla
    table_top = H-9.6*cm
    headers   = ["CÓDIGO","DESCRIPCIÓN DEL PRODUCTO","CANT.","UNIDAD","P.UNIT.($)","TOTAL($)"]
    col_w     = [2.0*cm, 7.5*cm, 1.5*cm, 1.8*cm, 2.5*cm, 2.5*cm]
    hx = 2*cm; hr = 0.65*cm

    c.setFillColor(AZUL)
    c.rect(hx, table_top-hr, sum(col_w), hr, fill=1, stroke=0)
    cx = hx
    col_cx = []
    for w in col_w:
        col_cx.append(cx + w/2); cx += w
    for i, h in enumerate(headers):
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 7.5)
        c.drawCentredString(col_cx[i], table_top-hr+0.18*cm, h)

    row_h  = 0.55*cm
    zebra  = [colors.HexColor("#f0f4f8"), colors.white]
    for ri, row in enumerate(filas):
        ry = table_top - hr - (ri+1)*row_h
        c.setFillColor(zebra[ri%2])
        c.rect(hx, ry, sum(col_w), row_h, fill=1, stroke=0)
        lx2 = hx
        c.setStrokeColor(colors.HexColor("#dddddd")); c.setLineWidth(0.3)
        for w in col_w[:-1]:
            lx2 += w; c.line(lx2, ry, lx2, ry+row_h)
        c.setFillColor(colors.black); c.setFont("Helvetica", 7.5)
        cx2 = hx
        for ci, cell in enumerate(row[:6]):
            if ci in (2, 4, 5):
                c.drawRightString(cx2+col_w[ci]-0.2*cm, ry+0.13*cm, cell)
            elif ci == 1:
                c.drawString(cx2+0.2*cm, ry+0.13*cm, cell)
            else:
                c.drawCentredString(cx2+col_w[ci]/2, ry+0.13*cm, cell)
            cx2 += col_w[ci]

    last_y = table_top - hr - len(filas)*row_h
    c.setStrokeColor(AZUL); c.setLineWidth(0.8)
    c.rect(hx, last_y, sum(col_w), hr+len(filas)*row_h, fill=0, stroke=1)

    # Totales
    ty  = last_y - 0.4*cm
    tlx = 14.5*cm; trx = W-2*cm
    txt(tlx, ty,           "Subtotal neto:",  9, True, align="right")
    txt(trx, ty,           f"$ {fmt(subtotal)}",      9, align="right")
    txt(tlx, ty-0.55*cm,   "IVA (19%):",     9, True, align="right")
    txt(trx, ty-0.55*cm,   f"$ {fmt(iva)}",           9, align="right")
    ln(13*cm, ty-0.85*cm, W-2*cm, ty-0.85*cm, 1, AZUL)
    txt(tlx, ty-1.2*cm, "TOTAL A PAGAR:", 11, True, AZUL, align="right")
    txt(trx, ty-1.2*cm, f"$ {fmt(total_doc)}",  11, True, AZUL, align="right")

    # Observaciones
    oy2 = ty-2.0*cm
    txt(2*cm, oy2,          "Observaciones:", 8, True)
    txt(2*cm, oy2-0.5*cm,   "Entrega en bodega del local. Conservar cadena de frío donde corresponda.")
    txt(2*cm, oy2-1.0*cm,   f"Transferir a Cta Cte BancoEstado N° 00-{random.randint(100,999)}-{random.randint(10000,99999)}-{random.randint(0,9)} / RUT: {prov[1]}")

    ln(2*cm,    oy2-2.5*cm, 7*cm,    oy2-2.5*cm)
    ln(12.5*cm, oy2-2.5*cm, W-2*cm, oy2-2.5*cm)
    txt(4.5*cm,  oy2-2.9*cm, "Firma del receptor",      8, align="center")
    txt(15.3*cm, oy2-2.9*cm, "Timbre y firma proveedor",8, align="center")

    ln(2*cm, 1.6*cm, W-2*cm, 1.6*cm, 0.5, AZUL)
    txt(W/2, 1.0*cm, "Documento generado con fines académicos — Proyecto COMAND-IA",
        7, col=colors.HexColor("#888888"), align="center")

    c.save()

    # ── Retornar datos estructurados para exportar el ground truth ────────────
    datos_boleta = {
        "numero":        f"{numero:06d}",
        "fecha_emision": fecha,
        "fecha_vence":   vence,
        "proveedor":     {"nombre": prov[0], "rut": prov[1], "direccion": prov[2]},
        "cliente":       {"nombre": cli[0],  "rut": cli[1],  "direccion": cli[2]},
        "vendedor":      vend,
        "condicion_pago": cond,
        "forma_pago":    forma,
        # Cada producto guarda los valores ya calculados sin formateo de miles
        "productos": [
            {
                "codigo":          row[0],
                "descripcion":     row[1],
                "cantidad":        int(row[2]),
                "unidad":          row[3],
                "precio_unitario": int(row[4].replace(".", "")),
                "total":           int(row[5].replace(".", "")),
            }
            for row in filas
        ],
        "subtotal": subtotal,
        "iva":      iva,
        "total":    total_doc,
    }
    return datos_boleta

# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

def guardar_ground_truth(datos, nombre_base, carpeta, nivel, angulo):
    """
    Exporta el ground truth de una boleta como JSON con el mismo nombre
    base que el PNG. El agente SARSA carga este archivo para calcular
    recompensas durante el entrenamiento.

    Estructura:
      metadata  → info del documento (nivel de ruido, ángulo, timestamp)
      encabezado → proveedor, cliente, número, fechas
      productos  → lista de filas de la tabla (objetivo principal del agente)
      totales    → subtotal, IVA, total (para validación de coherencia)
    """
    gt = {
        "metadata": {
            "archivo_imagen":  f"{nombre_base}.png",
            "nivel_ruido":     nivel,
            "angulo_rotacion": round(angulo, 3),
            "generado_en":     datetime.now().isoformat(timespec="seconds"),
        },
        "encabezado": {
            "numero_boleta":  datos["numero"],
            "fecha_emision":  datos["fecha_emision"],
            "fecha_vence":    datos["fecha_vence"],
            "proveedor":      datos["proveedor"],
            "cliente":        datos["cliente"],
            "vendedor":       datos["vendedor"],
            "condicion_pago": datos["condicion_pago"],
            "forma_pago":     datos["forma_pago"],
        },
        "productos": datos["productos"],
        "totales": {
            "subtotal_neto": datos["subtotal"],
            "iva_19":        datos["iva"],
            "total":         datos["total"],
        },
    }
    json_path = os.path.join(carpeta, f"{nombre_base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    return json_path

# ══════════════════════════════════════════════════════════════════════════════
#  NIVELES DE RUIDO
# ══════════════════════════════════════════════════════════════════════════════

NIVELES = ["limpia", "poco", "medio", "mucho"]

def aplicar_ruido(pil_img, nivel):
    """Aplica degradación realista según nivel: limpia/poco/medio/mucho"""
    rng = np.random.default_rng()
    img = pil_img.convert("RGB")
    w, h = img.size

    # ── Parámetros según nivel ────────────────────────────────────────────────
    cfg = {
        "limpia": dict(rot=0.2,  contraste=0.97, amarillo=(2,1,-2),
                       ruido_std=2,   manchas_small=4,  manchas_med=0,
                       blur=0.0,  sombra=0.03, perspectiva=False, tinta=False,
                       pliegue=False, jpeg_q=None),
        "poco":   dict(rot=0.8,  contraste=0.88, amarillo=(8,4,-8),
                       ruido_std=6,   manchas_small=30, manchas_med=2,
                       blur=0.55, sombra=0.12, perspectiva=False, tinta=False,
                       pliegue=False, jpeg_q=None),
        "medio":  dict(rot=2.5,  contraste=0.78, amarillo=(15,8,-14),
                       ruido_std=14,  manchas_small=65, manchas_med=5,
                       blur=1.0,  sombra=0.22, perspectiva=True,  tinta=True,
                       pliegue=True,  jpeg_q=55),
        "mucho":  dict(rot=5.5,  contraste=0.62, amarillo=(25,12,-22),
                       ruido_std=28,  manchas_small=130,manchas_med=10,
                       blur=2.0,  sombra=0.35, perspectiva=True,  tinta=True,
                       pliegue=True,  jpeg_q=30),
    }[nivel]

    # 1. Rotación
    angle = random.uniform(-cfg["rot"], cfg["rot"])
    img = img.rotate(angle, fillcolor=(240, 238, 228), expand=False)

    # 2. Perspectiva trapezoidal (simula foto tomada con ángulo)
    if cfg["perspectiva"]:
        skew = random.randint(5, 25)
        side = random.choice(["left", "right", "top"])
        cx_off = skew if side == "right" else -skew if side == "left" else 0
        cy_off = skew if side == "top" else 0
        coeffs = (1, cx_off/w, -cx_off/2,
                  cy_off/h, 1, -cy_off/2)
        img = img.transform(img.size, Image.AFFINE, coeffs,
                            resample=Image.BILINEAR,
                            fillcolor=(240, 238, 228))

    # 3. Contraste
    img = ImageEnhance.Contrast(img).enhance(cfg["contraste"])

    # 4. Tono papel (amarillento)
    arr = np.array(img, dtype=np.float32)
    dr, dg, db = cfg["amarillo"]
    arr[:,:,0] = np.clip(arr[:,:,0]+dr, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]+dg, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]+db, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    # 5. Ruido gaussiano
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + rng.normal(0, cfg["ruido_std"], arr.shape), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    # 6. Manchas pequeñas (polvo/suciedad escáner)
    draw = ImageDraw.Draw(img)
    for _ in range(cfg["manchas_small"]):
        x = random.randint(0, w); y = random.randint(0, h)
        r = random.randint(1, 4)
        g = random.randint(60, 190)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(g, g, g-8))

    # 7. Manchas medianas (huellas/gotas)
    for _ in range(cfg["manchas_med"]):
        x = random.randint(int(w*0.05), int(w*0.95))
        y = random.randint(int(h*0.05), int(h*0.95))
        rw = random.randint(15, 55); rh = random.randint(8, 28)
        g  = random.randint(170, 225)
        draw.ellipse([x-rw, y-rh, x+rw, y+rh], fill=(g, g-4, g-14))

    # 8. Tinta baja (líneas horizontales levemente más claras)
    if cfg["tinta"]:
        arr = np.array(img, dtype=np.float32)
        for _ in range(random.randint(3, 12)):
            y_l = random.randint(0, h)
            thickness = random.randint(1, 3)
            arr[y_l:y_l+thickness, :, :] = np.clip(
                arr[y_l:y_l+thickness, :, :] + random.randint(20, 50), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # 9. Pliegue (línea diagonal levemente más oscura)
    if cfg["pliegue"]:
        draw2 = ImageDraw.Draw(img)
        x0 = random.randint(0, w//2); y0 = 0
        x1 = random.randint(w//2, w); y1 = h
        for off in range(-2, 3):
            draw2.line([(x0+off, y0), (x1+off, y1)],
                       fill=(180, 178, 165), width=1)

    # 10. Sombra de borde
    sw = int(w * cfg["sombra"])
    if sw > 0:
        arr = np.array(img, dtype=np.float32)
        side = random.choice(["left", "right", "top", "bottom"])
        for i in range(sw):
            factor = 1.0 - 0.35 * (1 - i/sw)
            if   side == "left":   arr[:, i, :]    *= factor
            elif side == "right":  arr[:, -(i+1), :] *= factor
            elif side == "top":    arr[i, :, :]    *= factor
            elif side == "bottom": arr[-(i+1), :, :] *= factor
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 11. Desenfoque
    if cfg["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=cfg["blur"]))

    # 12. Compresión JPEG (artefactos de digitalización barata)
    if cfg["jpeg_q"]:
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, "JPEG", quality=cfg["jpeg_q"])
        buf.seek(0)
        img = Image.open(buf).copy()

    return img, angle, nivel

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generador de boletas con ruido para COMAND-IA")
    parser.add_argument("--cantidad", type=int, default=12,
                        help="Número de boletas a generar (default: 12)")
    parser.add_argument("--salida", type=str, default="/mnt/user-data/outputs/boletas",
                        help="Carpeta de salida")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla aleatoria para reproducibilidad")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    os.makedirs(args.salida, exist_ok=True)

    # Distribuir niveles equitativamente (y el resto al azar)
    base  = args.cantidad // 4
    extra = args.cantidad % 4
    niveles_lista = NIVELES * base + random.sample(NIVELES, extra)
    random.shuffle(niveles_lista)

    print(f"\n{'='*55}")
    print(f"  COMAND-IA — Generador de Boletas de Prueba")
    print(f"{'='*55}")
    print(f"  Cantidad : {args.cantidad}")
    print(f"  Salida   : {args.salida}")
    print(f"{'='*55}\n")

    resumen = {n: 0 for n in NIVELES}

    for i, nivel in enumerate(niveles_lista, start=1):
        numero = random.randint(1000, 99999)
        nombre_base = f"boleta_{i:03d}_N{numero}_{nivel}"
        pdf_tmp  = f"/tmp/{nombre_base}.pdf"
        png_out  = os.path.join(args.salida, f"{nombre_base}.png")

        # Generar PDF limpio y capturar datos estructurados para ground truth
        datos_boleta = generar_pdf(pdf_tmp, numero)

        # Convertir a imagen
        pages = convert_from_path(pdf_tmp, dpi=180)
        img_limpia = pages[0]

        # Aplicar ruido
        img_final, angle, _ = aplicar_ruido(img_limpia, nivel)

        # Guardar PNG
        img_final.save(png_out, "PNG", dpi=(180, 180))
        os.remove(pdf_tmp)

        # Exportar ground truth como JSON (mismo nombre base que el PNG)
        guardar_ground_truth(datos_boleta, nombre_base, args.salida, nivel, angle)

        resumen[nivel] += 1
        n_prod = len(datos_boleta["productos"])
        print(f"  [{i:>3}/{args.cantidad}] {nombre_base}.png  (rot: {angle:+.1f}° | {n_prod} prods | GT: ✓)")

    print(f"\n{'='*55}")
    print("  Resumen por nivel:")
    for n, cnt in resumen.items():
        bar = "█" * cnt
        print(f"    {n:<8} {bar} ({cnt})")
    print(f"\n  Archivos en: {args.salida}")
    print(f"  Formato:     boleta_NNN_NXXXXX_nivel.png  +  .json")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()