# =====================================================================
# src/exportar.py — Escribe models/config.json para el modelo elegido (ResNet18),
# lo exporta a ONNX y verifica que ONNX y PyTorch den la misma salida.
# Uso:  python src/exportar.py
# =====================================================================
import json  # Para escribir el config
import sys  # Para ajustar el path
from pathlib import Path  # Rutas

import numpy as np  # Para comparar las salidas
import onnxruntime as ort  # Motor que ejecuta modelos .onnx (sin PyTorch)
import torch  # PyTorch
import torchvision  # Para leer el orden real de las clases con ImageFolder

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))
from src.predecir import cargar_modelo  # Reutilizamos la misma función de carga que usa la app

MODELS_DIR = RAIZ / "models"

# ---- 1. config.json del modelo elegido ----
clases = torchvision.datasets.ImageFolder(RAIZ / "data" / "train").classes  # El orden SALE de ImageFolder, no se escribe a mano
config = {
    "modelo": "resnet18",  # Modelo elegido tras la comparación del notebook 02
    "pesos": "frutas_resnet18.pth",  # Archivo de pesos (state_dict)
    "onnx": "frutas_resnet18.onnx",  # Versión ONNX (para la web y otros lenguajes)
    "img_size": 224,  # ResNet18 trabaja con 224x224
    "mean": [0.485, 0.456, 0.406],  # Normalización de ImageNet
    "std": [0.229, 0.224, 0.225],
    "clases": clases,
    "nombres_es": {
        "freshapples": "Manzana fresca", "freshbanana": "Banana fresca", "freshoranges": "Naranja fresca",
        "rottenapples": "Manzana podrida", "rottenbanana": "Banana podrida", "rottenoranges": "Naranja podrida",
    },
}
(MODELS_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print("config.json escrito:", config["modelo"], config["clases"])

# ---- 2. Exportar a ONNX ----
modelo, config, _ = cargar_modelo()  # Carga ResNet18 con nuestros pesos, en CPU y en modo eval
dummy = torch.randn(1, 3, config["img_size"], config["img_size"])  # Entrada de ejemplo: define la forma
ruta_onnx = MODELS_DIR / config["onnx"]
torch.onnx.export(
    modelo, (dummy,), str(ruta_onnx),
    input_names=["imagen"], output_names=["logits"],  # Nombres de la entrada y la salida
    dynamic_shapes=({0: torch.export.Dim("batch")},),  # Tamaño de batch variable (1 o varias imágenes)
    external_data=False,  # Un solo archivo .onnx (sin un .onnx.data aparte)
)
print(f"ONNX exportado: {ruta_onnx} ({ruta_onnx.stat().st_size / 1e6:.1f} MB)")

# ---- 3. Verificar que ONNX y PyTorch coinciden ----
sess = ort.InferenceSession(str(ruta_onnx))  # Carga el .onnx con onnxruntime
x = torch.randn(4, 3, config["img_size"], config["img_size"])  # 4 imágenes aleatorias (prueba también el batch dinámico)
salida_onnx = sess.run(None, {"imagen": x.numpy()})[0]
with torch.no_grad():
    salida_torch = modelo(x).numpy()
print("¿Coinciden PyTorch y ONNX?", np.allclose(salida_onnx, salida_torch, atol=1e-4),
      f"(diferencia máxima: {np.abs(salida_onnx - salida_torch).max():.2e})")
