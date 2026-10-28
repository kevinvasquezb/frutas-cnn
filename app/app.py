# =====================================================================
# app/app.py — Demo web con Gradio: sube una foto o usa la cámara y el modelo
# dice qué fruta es y si está apta para venta o hay que descartarla.
# Uso:  python app/app.py   ->  abre http://127.0.0.1:7860
# =====================================================================
import sys  # Para ajustar el path
from pathlib import Path  # Rutas

import gradio as gr  # Librería para crear interfaces web de modelos de ML con pocas líneas

RAIZ = Path(__file__).resolve().parent.parent  # Raíz del proyecto
sys.path.append(str(RAIZ))  # Para poder importar src/
from src.predecir import cargar_config, predecir as predecir_probs  # Misma lógica de predicción que la terminal

config = cargar_config()  # Clases, nombres en español y modelo usado
nombres = config["nombres_es"]
UMBRAL_CONFIANZA = 0.60  # Si la clase más probable tiene menos de 60%, avisamos que la foto no es confiable


def analizar(img):
    """Recibe la imagen de Gradio (PIL) y devuelve (probabilidades para gr.Label, veredicto en Markdown)."""
    if img is None:  # Si el usuario presionó el botón sin imagen
        return {}, "Sube una foto o toma una con la cámara."
    probs = predecir_probs(img)  # {clase: probabilidad}, ordenado de mayor a menor
    top, conf = next(iter(probs.items()))  # Clase más probable y su probabilidad
    resultado = {nombres[c]: p for c, p in probs.items()}  # Con nombres en español para mostrar

    if top == "nada":  # Clase "sin fruta" (existe si el modelo se reentrenó con fotos propias, notebook 03)
        return resultado, "# 🔍 No se detecta fruta\nPon una manzana, banana o naranja frente a la cámara."
    if top.startswith("rotten"):
        veredicto = f"# ❌ Descartar (podrida)\n**{nombres[top]}** con {conf:.1%} de confianza."
    else:
        veredicto = f"# ✅ Apta para venta\n**{nombres[top]}** con {conf:.1%} de confianza."
    if conf < UMBRAL_CONFIANZA:  # El modelo duda: puede ser otra fruta, mala luz o fondo raro
        veredicto += ("\n\n⚠️ **Confianza baja.** Puede que la foto no sea de manzana, banana o naranja, "
                      "o que haya mala luz o un fondo complicado. Prueba con una sola fruta sobre fondo claro.")
    return resultado, veredicto


demo = gr.Interface(
    fn=analizar,  # Función que se ejecuta al enviar
    inputs=gr.Image(type="pil", label="Foto de la fruta", sources=["upload", "webcam"]),  # Subir archivo o usar la cámara
    outputs=[gr.Label(num_top_classes=3, label="Predicción (top 3)"), gr.Markdown(label="Veredicto")],
    examples=sorted(str(p) for p in (RAIZ / "app" / "ejemplos").glob("*.png")),  # Imágenes de ejemplo para probar con 1 clic
    title="🍎🍌🍊 Clasificador de Frutas Frescas vs Podridas",
    description=(f"Modelo: **{config['modelo']}** (transfer learning, PyTorch). Sube una foto o usa la cámara: "
                 "una sola fruta (manzana, banana o naranja), centrada y sobre fondo claro. "
                 "Proyecto de IA — UTEPSA."),
    flagging_mode="never",  # Sin el botón "Flag" (no guardamos datos de los usuarios)
)

if __name__ == "__main__":
    demo.launch()  # Servidor local en http://127.0.0.1:7860
