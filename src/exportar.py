# =====================================================================
# src/exportar.py — Exporta a ONNX el modelo que indica models/config.json
# y verifica que ONNX y PyTorch den la misma salida.
# Uso:  python src/exportar.py      (o desde un notebook: from src.exportar import exportar_onnx)
# =====================================================================
import sys  # Para ajustar el path
from pathlib import Path  # Rutas

import numpy as np  # Para comparar las salidas
import onnxruntime as ort  # Motor que ejecuta modelos .onnx (sin PyTorch)
import torch  # PyTorch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))
from src.predecir import cargar_modelo  # La misma función de carga que usa la app

MODELS_DIR = RAIZ / "models"


def exportar_onnx():
    modelo, config, _ = cargar_modelo()  # Modelo del config, con sus pesos, en CPU y modo eval
    dummy = torch.randn(1, 3, config["img_size"], config["img_size"])  # Entrada de ejemplo: define la forma
    ruta_onnx = MODELS_DIR / config["onnx"]
    torch.onnx.export(
        modelo, (dummy,), str(ruta_onnx),
        input_names=["imagen"], output_names=["logits"],  # Nombres de la entrada y la salida
        dynamic_shapes=({0: torch.export.Dim("batch")},),  # Tamaño de batch variable
        external_data=False,  # Un solo archivo .onnx (necesario para la web)
    )
    print(f"ONNX exportado: {ruta_onnx} ({ruta_onnx.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(ruta_onnx))  # Carga el .onnx con onnxruntime
    x = torch.randn(4, 3, config["img_size"], config["img_size"])  # 4 imágenes (prueba el batch dinámico)
    salida_onnx = sess.run(None, {"imagen": x.numpy()})[0]
    with torch.no_grad():
        salida_torch = modelo(x).numpy()
    iguales = np.allclose(salida_onnx, salida_torch, atol=1e-4)
    print("¿Coinciden PyTorch y ONNX?", iguales, f"(diferencia máxima: {np.abs(salida_onnx - salida_torch).max():.2e})")
    return iguales


if __name__ == "__main__":
    exportar_onnx()
