# COMAND-IA: extraccion de boletas con aprendizaje por refuerzo

Proyecto en Python para extraer informacion estructurada desde boletas escaneadas
en formato PNG usando OCR y un agente tabular de aprendizaje por refuerzo. El
sistema compara SARSA y Q-Learning sobre pares de boleta/ground truth (`.png` y
`.json`) y puede guardar o cargar la Q-table entrenada.

## Contenido del proyecto

- `sistema_rl.py`: script principal del sistema OCR + entorno + agente RL.
- `boletas/`: dataset sintetico de boletas en PNG con sus archivos JSON de
  referencia.
- `modelos/`: modelos/Q-tables entrenadas en formato JSON.
- `generador de boletas/generador_boletas.py`: generador de boletas sinteticas.
- `requirements_rl.txt`: dependencias principales para ejecutar el sistema RL.

## Requisitos

- Python 3.10 o superior recomendado.
- Tesseract OCR instalado si quieres usar el fallback con `pytesseract`.
- Dependencias de Python listadas en `requirements_rl.txt`.

En Windows, si `pytesseract` no encuentra Tesseract, instala Tesseract OCR y
agrega su carpeta al `PATH`.

## Instalacion

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_rl.txt
```

Para usar el generador de boletas tambien instala sus dependencias:

```powershell
pip install -r "generador de boletas\requirements.txt"
```

## Uso rapido

Entrenar SARSA con una muestra pequena:

```powershell
python sistema_rl.py --boletas .\boletas --epochs 5 --modo sarsa --max-boletas 20 --guardar-modelo .\modelos\sarsa_demo.json
```

Entrenar Q-Learning:

```powershell
python sistema_rl.py --boletas .\boletas --epochs 5 --modo qlearning --max-boletas 20 --guardar-modelo .\modelos\qlearning_demo.json
```

Evaluar un modelo guardado sin entrenar:

```powershell
python sistema_rl.py --boletas .\boletas --modo sarsa --cargar-modelo .\modelos\sarsa_demo.json --evaluar --max-boletas 20
```

Analizar una boleta especifica:

```powershell
python sistema_rl.py --boletas .\boletas --modo sarsa --cargar-modelo .\modelos\sarsa_200_mejorado.json --evaluar --boleta-especifica boleta_001_N52073_mucho
```

Tambien puedes indicar el archivo con extension:

```powershell
python sistema_rl.py --boletas .\boletas --modo sarsa --cargar-modelo .\modelos\sarsa_200_mejorado.json --evaluar --boleta-especifica boleta_001_N52073_mucho.png
```

Entrenar sin generar grafico comparativo:

```powershell
python sistema_rl.py --boletas .\boletas --epochs 10 --modo sarsa --sin-comparacion
```

## Parametros principales

- `--boletas`: carpeta con pares `.png` y `.json`.
- `--epochs`: numero de pasadas sobre el dataset.
- `--modo`: algoritmo principal, `sarsa` o `qlearning`.
- `--alpha`: tasa de aprendizaje.
- `--gamma`: factor de descuento.
- `--epsilon-inicial` y `--epsilon-final`: control de exploracion.
- `--max-boletas`: limite para pruebas rapidas.
- `--boleta-especifica`: procesa una boleta puntual por nombre.
- `--guardar-modelo`: ruta para guardar la Q-table entrenada.
- `--cargar-modelo`: ruta de una Q-table existente.
- `--evaluar`: carga un modelo y ejecuta politica greedy sin entrenar.
- `--gpu`: usa GPU con EasyOCR si esta disponible.
- `--ocr-lang`: idiomas de EasyOCR separados por coma, por ejemplo `es,en`.
- `--cache-ocr`: carpeta usada para cachear resultados OCR.
- `--sin-cache-ocr`: desactiva el cache OCR.
- `--grafico`: ruta de salida para el grafico comparativo.

## Generar boletas sinteticas

El proyecto incluye un generador de boletas chilenas con diferentes niveles de
ruido visual. Ejemplo:

```powershell
python "generador de boletas\generador_boletas.py" --cantidad 20 --salida .\boletas --seed 42
```

Cada boleta se genera como imagen `.png` y archivo `.json` con los datos
esperados para entrenamiento/evaluacion.

## Flujo recomendado

1. Crear o actualizar el dataset de boletas.
2. Entrenar con `--max-boletas` para validar rapido.
3. Entrenar con el dataset completo.
4. Guardar el modelo en `modelos/`.
5. Evaluar el modelo con `--evaluar`.

## Notas

- EasyOCR puede descargar modelos la primera vez que se ejecuta.
- El cache OCR (`.ocr_cache/`) acelera entrenamientos repetidos sobre las mismas
  boletas.
- Los artefactos temporales, caches, entorno virtual y salidas generadas se
  excluyen del repositorio mediante `.gitignore`.
