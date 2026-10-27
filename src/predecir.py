# =====================================================================
# src/predecir.py — Cargar el modelo entrenado y predecir una foto.
# Uso desde la terminal:   python src/predecir.py ruta/a/foto.jpg
# Uso desde Python:        from src.predecir import cargar_modelo, predecir
# =====================================================================
import json  # Para leer models/config.json
import sys  # Para leer argumentos de la terminal y ajustar el path
from pathlib import Path  # Rutas de archivos

import torch  # PyTorch
import torch.nn as nn  # Para reemplazar la capa final de ResNet18
import torchvision.transforms as transforms  # Preprocesamiento (el mismo que el test)
from PIL import Image, ImageOps  # Abrir fotos y corregir su orientación

RAIZ = Path(__file__).resolve().parent.parent  # Raíz del proyecto (src/ está un nivel abajo)
sys.path.append(str(RAIZ))  # Para poder hacer "from src.modelo import Net" aunque se ejecute como script
MODELS_DIR = RAIZ / "models"  # Carpeta con los pesos y el config


def cargar_config():
    """Lee models/config.json: qué modelo es, sus clases, el tamaño y la normalización."""
    return json.loads((MODELS_DIR / "config.json").read_text(encoding="utf-8"))


def construir_modelo(config):
    """Crea la arquitectura indicada en el config (sin pesos entrenados todavía)."""
    n_clases = len(config["clases"])
    if config["modelo"] == "resnet18":
        from torchvision.models import resnet18
        modelo = resnet18(weights=None)  # Misma arquitectura; los pesos los cargamos de nuestro .pth
        modelo.fc = nn.Linear(modelo.fc.in_features, n_clases)  # Misma capa final que en el notebook 02
        return modelo
    from src.modelo import Net  # La CNN propia del notebook 01
    return Net(n_clases=n_clases)


def cargar_modelo():
    """Devuelve (modelo listo para predecir, config, transform)."""
    config = cargar_config()
    modelo = construir_modelo(config)
    pesos = torch.load(MODELS_DIR / config["pesos"], map_location="cpu")  # map_location: funciona aunque no haya GPU
    modelo.load_state_dict(pesos)  # Carga los pesos entrenados
    modelo.eval()  # Modo evaluación (desactiva Dropout; BatchNorm usa las estadísticas aprendidas)

    transform = transforms.Compose([  # EXACTAMENTE el mismo preprocesamiento que el test del notebook
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    return modelo, config, transform


_cache = {}  # Para cargar el modelo una sola vez aunque predecir() se llame muchas veces


def predecir(img_pil):
    """Recibe una imagen PIL y devuelve {clase: probabilidad} ordenado de mayor a menor."""
    if not _cache:
        _cache["modelo"], _cache["config"], _cache["transform"] = cargar_modelo()
    modelo, config, transform = _cache["modelo"], _cache["config"], _cache["transform"]

    img = ImageOps.exif_transpose(img_pil).convert("RGB")  # Corrige la rotación de fotos de celular y asegura RGB
    x = transform(img).unsqueeze(0)  # Preprocesa y agrega la dimensión de batch: (1, 3, H, W)
    with torch.no_grad():  # Sin gradientes: solo inferencia
        probs = torch.softmax(modelo(x), dim=1)[0]  # Logits -> probabilidades (suman 1)
    resultado = {c: float(p) for c, p in zip(config["clases"], probs)}
    return dict(sorted(resultado.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/predecir.py ruta/a/foto.jpg [otra_foto.jpg ...]")
        sys.exit(1)
    nombres = cargar_config()["nombres_es"]
    for ruta in sys.argv[1:]:
        probs = predecir(Image.open(ruta))
        top = next(iter(probs))  # La clase con mayor probabilidad
        veredicto = "DESCARTAR (podrida)" if top.startswith("rotten") else "APTA PARA VENTA"
        print(f"\n{Path(ruta).name}  ->  {nombres[top]} ({probs[top]:.1%})  =>  {veredicto}")
        for clase, p in list(probs.items())[:3]:  # Top 3
            print(f"   {nombres[clase]:<16} {p:6.1%}")
