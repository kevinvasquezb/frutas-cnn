# CLAUDE.md — Clasificador de Frutas Frescas vs Podridas (CNN con PyTorch)

> **Para Claude Code:** este archivo es el contexto del proyecto. Léelo entero antes de empezar.
> Trabaja **por fases** (sección 6), confirma con Mirko al final de cada fase y **no avances a la siguiente sin su OK**.

---

## 0. Para Mirko: cómo usar este archivo

1. Crea la carpeta del proyecto y pon este archivo en la raíz como `CLAUDE.md`:
   ```powershell
   mkdir C:\dev\frutas-cnn
   cd C:\dev\frutas-cnn
   # copia aquí este CLAUDE.md
   ```
2. Copia los notebooks de CNN de tu maestro a `referencia/`:
   - `1Case_Study_CNN_Multi-ClassClassification_PyTorch.ipynb`
   - `2CNN_BinaryClassification_PyTorch_Assignment4.ipynb`
3. Abre una terminal en la carpeta y ejecuta `claude`.
4. Primer mensaje sugerido:
   > Lee CLAUDE.md y los notebooks de `referencia/`. Empecemos con la Fase 1.
5. En cada fase, cuando Claude Code termine, revisa, corre lo que te pida y dile "ok, siguiente fase" o pásale el error.

---

## 1. Contexto

- **Alumno:** Mirko, estudiante de Ingeniería de Sistemas (UTEPSA, Santa Cruz, Bolivia). Proyecto final de la materia de IA (Deep Learning).
- **Objetivo académico:** un proyecto de CNN que se entrene en minutos u horas, que **siga los pasos del maestro** y que Mirko **entienda y pueda defender** celda por celda.
- **Proyecto:** clasificar fotos de frutas en 6 clases: manzana, banana y naranja, cada una **fresca** o **podrida**.
- **Caso de uso que se presenta:** control de calidad en mercados o packing (descartar fruta podrida automáticamente).
- **Idioma:** todo en **español**: comentarios, markdown del notebook, prints, README y app.
- **Hardware de entrenamiento:** PC con Windows, RTX 4070 de 8 GB VRAM, 32 GB de RAM, Intel i7 de 14.ª generación.

### Reglas de estilo (importante)
- **Imitar el estilo del maestro** (ver `referencia/`): PyTorch puro, `torchvision.datasets.ImageFolder`, `transforms.Compose`, una clase `nn.Module` con capas en `__init__` y `forward`, y un loop de entrenamiento explícito (`zero_grad → forward → loss → backward → step`).
- **Comentar casi cada línea** en español, explicando *qué hace* y *por qué*, como en los notebooks del maestro.
- **Nada de frameworks que oculten el loop** (no Lightning, no fastai, no Keras). La idea es que se vea todo.
- Antes de cada bloque de código, una celda markdown que explique la teoría en 3 a 6 líneas.
- Simple antes que clever. Si algo es "extra", márcalo como **(Opcional)**.

---

## 2. Patrón del maestro (resumen de `referencia/`)

El Assignment 4 (perros vs gatos) sigue este pipeline. **El proyecto debe seguir las mismas secciones:**

1. Importar librerías (torch, torchvision, numpy, matplotlib, seaborn, sklearn.metrics).
2. Transforms:
   - train: `Resize((128,128))`, `RandomHorizontalFlip()`, `RandomRotation(10)`, `ToTensor()`, `Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`
   - test: `Resize((128,128))`, `ToTensor()`, `Normalize(...)`
3. `ImageFolder` para train y test, y `DataLoader` con `batch_size=32`.
4. Función `imshow` para desnormalizar (`img/2 + 0.5`) y mostrar un grid con `make_grid`.
5. Clase `Net(nn.Module)`: 3 bloques `Conv2d(3x3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2,2)` con filtros 32 → 64 → 128, `Dropout(0.25)`, `Linear(128*16*16, 512) → BatchNorm1d → ReLU → Dropout → Linear(512, n_clases)`.
6. `CrossEntropyLoss` + `Adam(lr=0.001)`, 20 epochs, imprimiendo la loss promedio por epoch.
7. Evaluación: `net.eval()`, `torch.no_grad()`, `accuracy_score` y matriz de confusión con seaborn.
8. Predicción sobre imágenes aleatorias del test, mostrando cada imagen con su predicción.
9. Preguntas del assignment: learning rate, batch size, epochs, filtros, tamaño de imagen, normalización, kernel, accuracy y **"¿cómo mejorar la accuracy?"**.

**Lo que el maestro NO hace y nosotros sí agregamos** (y explicamos por qué):
- Usar la **GPU** (`device = "cuda"`). El código del maestro corre en CPU.
- Separar un **conjunto de validación** a partir de train.
- Graficar **curvas de loss y accuracy** (train vs validación) para detectar overfitting.
- **Guardar el mejor modelo** según la accuracy de validación.
- Un `classification_report` (precision, recall y F1 por clase).
- **(Opcional)** Ver filtros y feature maps para entender qué "ve" la CNN.
- **Transfer learning con ResNet18** como comparación, que responde a "¿cómo mejorar la accuracy?".
- Exportar el modelo y una **app** para usarlo.

---

## 3. Dataset

- **Nombre:** *Fruits fresh and rotten for classification* (Kaggle)
- **URL:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- **Tamaño aproximado:** ~13.6k imágenes (~10.9k train y ~2.7k test), unos 3–4 GB descomprimido.
- **Clases esperadas** (verificar los nombres reales de las carpetas después de descargar):
  `freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges`

**Verificado en la Fase 1 (2026-10-27):** los nombres de carpeta coinciden. Venía duplicado (`dataset/train|test` y `dataset/dataset/train|test`, idénticos); se dejó una sola copia en `data/train` y `data/test`. Todas son `.png` y ninguna está corrupta (se abrió y decodificó cada una con PIL).

| clase | train | test | total |
|---|---:|---:|---:|
| freshapples | 1693 | 395 | 2088 |
| freshbanana | 1581 | 381 | 1962 |
| freshoranges | 1466 | 388 | 1854 |
| rottenapples | 2342 | 601 | 2943 |
| rottenbanana | 2224 | 530 | 2754 |
| rottenoranges | 1595 | 403 | 1998 |
| **TOTAL** | **10901** | **2698** | **13599** |

**Data leakage (verificado):** solo ~11% de las imágenes son originales (1182 en train y 329 en test); el resto son copias aumentadas por el autor (prefijos `rotated_by_{15,30,45,60,75}_`, `translation_`, `vertical_flip_`, `saltandpepper_`). Las 2698 imágenes de test comparten nombre base con alguna de train, así que la accuracy en test/val es optimista. Esto se explica en el notebook 01 (sección 4); la prueba honesta son las fotos de `imagenes_prueba/`.

### Descarga

**Opción A, con la CLI de Kaggle:**
```powershell
pip install kaggle
# Descarga tu token desde kaggle.com -> Settings -> API -> "Create New Token"
# y guarda kaggle.json en  C:\Users\<tu_usuario>\.kaggle\kaggle.json
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p data --unzip
```

**Opción B, manual:** botón *Download* en Kaggle y descomprimir dentro de `data/`.

### Estructura final esperada
```
data/
  train/  freshapples/ freshbanana/ freshoranges/ rottenapples/ rottenbanana/ rottenoranges/
  test/   (las mismas 6 carpetas)
```
> **Claude Code:** después de descomprimir, revisa la estructura real con `dir` o `ls`. Este dataset suele traer carpetas anidadas (`dataset/dataset/train`) y a veces una copia duplicada. Normaliza todo a `data/train` y `data/test`, borra los duplicados y **cuenta las imágenes por clase** (imprime la tabla). Si hay archivos que no son imágenes o que están corruptos, repórtalos y exclúyelos.

---

## 4. Estructura del proyecto

```
frutas-cnn/
├── CLAUDE.md
├── README.md                  # cómo instalar, entrenar y usar (para el profe/GitHub)
├── requirements.txt
├── .gitignore                 # data/, models/*.pth, models/*.onnx, .venv/, __pycache__/, .ipynb_checkpoints/
├── referencia/                # notebooks del maestro (solo lectura)
├── data/                      # dataset (NO se sube a git)
├── notebooks/
│   ├── 01_frutas_cnn.ipynb            # CNN desde cero (proyecto principal)
│   └── 02_frutas_transfer_learning.ipynb  # ResNet18 (comparación)
├── src/
│   ├── modelo.py              # clase Net (la misma del notebook) para reutilizar en la app
│   └── predecir.py            # función predecir(imagen) -> (clase, probabilidades)
├── models/
│   ├── frutas_cnn.pth         # pesos (state_dict) del mejor modelo
│   ├── frutas_resnet18.pth
│   ├── frutas_cnn.onnx
│   └── config.json            # clases, nombres en español, img_size, mean, std
├── outputs/                   # figuras: curvas, matriz de confusión, feature maps
├── imagenes_prueba/           # fotos propias tomadas con el celular (mercado, casa)
├── app/
│   └── app.py                 # demo con Gradio
└── web/                       # (Opcional) demo en el navegador con onnxruntime-web
```

---

## 5. Entorno (Windows + RTX 4070)

```powershell
# Python 3.11, 3.12 o 3.13 (en esta PC se usó 3.13.4, la única instalada; torch 2.14.1+cu130 funciona)
py -3.13 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

# PyTorch con CUDA: confirmar el comando exacto en https://pytorch.org/get-started/locally/
# (Stable, Windows, Pip, Python, la versión de CUDA más nueva que salga). Ejemplo:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# Si torch.cuda.is_available() da False: actualizar el driver de NVIDIA (GeForce/Studio más reciente)
# o usar un índice de CUDA más viejo (p. ej. cu128). No hace falta instalar el CUDA Toolkit aparte.

pip install numpy matplotlib seaborn scikit-learn pillow jupyter ipykernel tqdm onnx onnxscript onnxruntime gradio kaggle
python -m ipykernel install --user --name frutas-cnn --display-name "Python (frutas-cnn)"
```

**Verificación de la GPU** (la Fase 1 no se cierra hasta que esto imprima `True` y el nombre de la GPU):
```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

`requirements.txt` (sin la línea de torch; esa se instala con el index de CUDA):
```
numpy
matplotlib
seaborn
scikit-learn
pillow
jupyter
ipykernel
tqdm
onnx
onnxscript
onnxruntime
gradio
```

**Notas de Windows:**
- `DataLoader` con `num_workers=4`, `pin_memory=True` y `persistent_workers=True`. Si se cuelga en Jupyter, usar `num_workers=0`. No usar `lambda` dentro de los transforms (Windows no las puede picklear).
- Si sale *CUDA out of memory*, bajar `batch_size` a 32.

---

## 6. Plan por fases

### Fase 1 — Entorno y datos
- [ ] Crear la estructura de carpetas, `.gitignore`, `requirements.txt` y `git init`.
- [ ] Crear el venv e instalar dependencias. Verificar CUDA.
- [ ] Descargar y normalizar el dataset en `data/train` y `data/test`.
- [ ] Imprimir el conteo de imágenes por clase y por split.
- **Entregable:** tabla de conteos + `torch.cuda.is_available() == True`.

### Fase 2 — Notebook `01_frutas_cnn.ipynb` (CNN desde cero)
Secciones (mismo orden que el maestro, con markdown de teoría antes de cada una):

1. **Título y descripción**: problema, dataset, caso de uso.
2. **Importar librerías** + `device` + semillas (`torch.manual_seed(42)`, `random`, `numpy`).
3. **Hiperparámetros en una sola celda**: `IMG_SIZE=128`, `BATCH_SIZE=64`, `LR=0.001`, `N_EPOCHS=20`, `VAL_SPLIT=0.15`, `QUICK_RUN=False`.
   - `QUICK_RUN=True` usa ~500 imágenes y 1 epoch, para probar que todo corre en segundos.
4. **Transforms**: los del maestro. En train agregar **(opcional, con comentario)** `ColorJitter(brightness=0.2, contrast=0.2)`.
5. **Datasets y DataLoaders**: train / validación / test (ver plantilla 7.1).
6. **Visualizar un batch** con `imshow` (desnormalizando) y los nombres de las clases.
7. **Modelo `Net`**: la arquitectura del maestro con `n_clases=6`. Usar `%%writefile ../src/modelo.py` para que la misma celda cree el archivo y luego `from src.modelo import Net` (ver plantilla 7.2). Imprimir el modelo y el **número de parámetros**, y comentar que la mayoría están en `fc1`.
8. **Loss y optimizador**: `CrossEntropyLoss` + `Adam(lr=0.001)`.
9. **Entrenamiento**: el loop del maestro + validación por epoch + guardar el mejor modelo + historial (plantilla 7.3). Mostrar el tiempo por epoch.
10. **Curvas** de loss y accuracy (train vs val). Guardar en `outputs/curvas_cnn.png`. Explicar cómo se ve el overfitting.
11. **Evaluación en test** con el mejor modelo: accuracy, `classification_report` y matriz de confusión (seaborn) guardada en `outputs/`.
12. **Predicciones visuales**: 10 imágenes aleatorias del test con predicción, real y % de confianza (verde si acierta, rojo si falla).
13. **Errores**: mostrar las imágenes mal clasificadas y comentar por qué pudo fallar.
14. **(Opcional) ¿Qué ve la CNN?**: filtros de `conv1` y feature maps de `conv1`, `conv2` y `conv3` para una imagen (con hooks o llamando a las capas paso a paso).
15. **Predicción con fotos propias** de `imagenes_prueba/`.
16. **Guardar**: `models/frutas_cnn.pth` + `models/config.json`.
17. **Respuestas a las preguntas del assignment** (las 10 preguntas del maestro, respondidas con los valores de ESTE proyecto).
18. **Conclusiones**.

- **Entregable:** notebook ejecutado de punta a punta. Accuracy en test esperada: **~90–97%**. Tiempo esperado: ~10–25 min en la 4070. **Real (2026-10-27): 99.07% en test, 3.0 min (20 epochs, batch 64, RTX 4070 Laptop).**
- **Claude Code:** prueba primero con `QUICK_RUN=True` (puedes ejecutarlo con `jupyter nbconvert --to notebook --execute`). El entrenamiento completo lo corre Mirko desde VS Code/Jupyter, o tú en background si él lo pide.

### Fase 3 — Notebook `02_frutas_transfer_learning.ipynb` (ResNet18)
- `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- Transforms: `Resize((224,224))` y normalización de ImageNet `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`. **Explicar por qué cambian** (el modelo fue preentrenado así).
- Etapa 1: congelar todo, reemplazar `model.fc = nn.Linear(512, 6)` y entrenar 3 epochs solo la cabeza.
- Etapa 2 **(opcional)**: descongelar `layer4` y entrenar 3–5 epochs con `lr=1e-4`.
- Mismas métricas y gráficos que la Fase 2.
- **Tabla comparativa final:** CNN propia vs ResNet18 (accuracy, parámetros, tamaño en MB, tiempo de entrenamiento, tiempo de inferencia por imagen).
- Guardar `models/frutas_resnet18.pth`.
- **Entregable:** tabla comparativa + conclusión de cuál conviene y por qué.

### Fase 4 — Exportar
- `models/config.json` (plantilla 7.4).
- Exportar a ONNX el modelo elegido (plantilla 7.5) y **verificar** con `onnxruntime` que las salidas coinciden con PyTorch (`np.allclose(..., atol=1e-4)`).
- `src/predecir.py`: `cargar_modelo()` y `predecir(img_pil) -> dict{clase: prob}` usando el config. **Mismos transforms que test.**
- Probar `predecir` desde la terminal con una foto de `imagenes_prueba/`.

### Fase 5 — App con Gradio
- `app/app.py` (plantilla 7.6): subir una foto o usar la webcam y mostrar el top 3 con %, más un veredicto grande: **"✅ Apta para venta"** o **"❌ Descartar (podrida)"**.
- Incluir 4–6 imágenes de ejemplo (`examples=`).
- Correr con `python app/app.py` → http://127.0.0.1:7860
- **(Opcional) Deploy gratis en Hugging Face Spaces** (SDK Gradio): subir `app.py`, `src/`, `models/frutas_cnn.pth` (o el `.onnx`), `config.json` y un `requirements.txt` con torch CPU. Usar `hf upload` o git con LFS (el `.pth` pesa ~68 MB).

### Fase 6 — (Opcional) Demo web sin servidor
- `web/index.html` + `onnxruntime-web` (desde CDN): carga `frutas_cnn.onnx`, preprocesa la imagen en un `<canvas>` (resize 128, `(x/255 - 0.5)/0.5`, layout NCHW), aplica softmax y muestra el resultado.
- Deploy en Vercel como sitio estático.

### Fase 7 — Cierre
- `README.md` con: problema, dataset, arquitectura (diagrama simple en texto), resultados (tabla + imágenes de `outputs/`), cómo instalar, entrenar y correr la app.
- Revisar que los notebooks queden **ejecutados y con outputs visibles**.
- Commit final. **(Opcional)** subir a GitHub (`mirkovedia/frutas-cnn`) **sin** `data/` ni archivos pesados.

---

## 7. Plantillas de referencia

> Úsalas como base. Mantén el estilo de comentarios del maestro (en español, casi línea por línea).
> Las plantillas 7.1–7.6 se probaron de punta a punta (split, modelo, entrenamiento, ONNX y app) con torch 2.14, onnxruntime 1.29 y gradio 6.30. Datos del modelo: **16.9M parámetros, de los cuales 16.8M están en `fc1`**, y el `.pth` pesa ~68 MB.

### 7.1 Split train / validación / test
```python
from torch.utils.data import Subset, DataLoader

# Dos "vistas" de la MISMA carpeta train: una con augmentation (para entrenar) y otra sin augmentation (para validar)
full_train_aug  = torchvision.datasets.ImageFolder('../data/train', transform=train_transform)
full_train_eval = torchvision.datasets.ImageFolder('../data/train', transform=test_transform)
test_dataset    = torchvision.datasets.ImageFolder('../data/test',  transform=test_transform)

clases = full_train_aug.classes  # nombres de clase, ordenados alfabéticamente por ImageFolder

# Mezclar índices con una semilla fija para que el split sea reproducible
g = torch.Generator().manual_seed(42)
indices = torch.randperm(len(full_train_aug), generator=g).tolist()
n_val = int(len(indices) * VAL_SPLIT)
val_idx, train_idx = indices[:n_val], indices[n_val:]

if QUICK_RUN:  # modo prueba rápida
    train_idx, val_idx = train_idx[:400], val_idx[:100]

train_dataset = Subset(full_train_aug,  train_idx)
val_dataset   = Subset(full_train_eval, val_idx)

loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=True)
train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)
```

### 7.2 Modelo (`src/modelo.py`, generado desde el notebook con `%%writefile`)
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """CNN del maestro (Assignment 4) adaptada a n_clases. Entrada: imágenes RGB de 128x128."""
    def __init__(self, n_clases=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # 128 -> 64 -> 32 -> 16 después de 3 poolings
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, n_clases)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)          # aplanar: (batch, 128*16*16)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)                 # logits (sin softmax; CrossEntropyLoss lo aplica internamente)
```
> En el notebook, la celda va con `%%writefile ../src/modelo.py` y comentarios línea por línea. Para que funcione `from src.modelo import Net`, agregar la raíz del proyecto al path: `sys.path.append('..')`.

### 7.3 Loop de entrenamiento con validación
```python
import time, copy

net = Net(n_clases=len(clases)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
mejor_acc, mejores_pesos = 0.0, None

for epoch in range(N_EPOCHS):
    t0 = time.time()
    # ---- ENTRENAMIENTO ----
    net.train()
    running_loss, correctos, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correctos += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_loss, train_acc = running_loss / total, correctos / total

    # ---- VALIDACIÓN ----
    net.eval()
    running_loss, correctos, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            running_loss += criterion(outputs, labels).item() * inputs.size(0)
            correctos += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_loss, val_acc = running_loss / total, correctos / total

    for k, v in zip(historial, [train_loss, train_acc, val_loss, val_acc]):
        historial[k].append(v)

    # Guardar el mejor modelo según la accuracy de validación
    if val_acc > mejor_acc:
        mejor_acc = val_acc
        mejores_pesos = copy.deepcopy(net.state_dict())

    print(f"Epoch [{epoch+1}/{N_EPOCHS}] train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
          f"| val_loss={val_loss:.3f} val_acc={val_acc:.3f} | {time.time()-t0:.1f}s")

net.load_state_dict(mejores_pesos)  # quedarse con el mejor modelo
```

### 7.4 `models/config.json`
```json
{
  "modelo": "Net",
  "img_size": 128,
  "mean": [0.5, 0.5, 0.5],
  "std": [0.5, 0.5, 0.5],
  "clases": ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"],
  "nombres_es": {
    "freshapples": "Manzana fresca", "freshbanana": "Banana fresca", "freshoranges": "Naranja fresca",
    "rottenapples": "Manzana podrida", "rottenbanana": "Banana podrida", "rottenoranges": "Naranja podrida"
  }
}
```
> `clases` debe salir de `ImageFolder.classes`, **no escribirse a mano**, porque el orden define qué índice es cada clase. Si los nombres reales de las carpetas difieren, ajustar `nombres_es`.

### 7.5 Exportar a ONNX y verificar
```python
import onnxruntime as ort, numpy as np

net.eval().cpu()
dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    net, (dummy,), "../models/frutas_cnn.onnx",
    input_names=["imagen"], output_names=["logits"],
    dynamic_shapes=({0: torch.export.Dim("batch")},),  # tamaño de batch variable
    external_data=False,   # un solo archivo .onnx (sin .onnx.data aparte), necesario para la app y la web
)
# Probado con torch 2.14 (exportador nuevo, el default desde 2.9). Requiere onnx + onnxscript instalados.

sess = ort.InferenceSession("../models/frutas_cnn.onnx")
x = torch.randn(4, 3, 128, 128)
salida_onnx = sess.run(None, {"imagen": x.numpy()})[0]
salida_torch = net(x).detach().numpy()
print("¿Coinciden PyTorch y ONNX?", np.allclose(salida_onnx, salida_torch, atol=1e-4))
```

### 7.6 App Gradio (`app/app.py`)
```python
import json, sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
import gradio as gr

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))
from src.modelo import Net

config = json.loads((RAIZ / "models" / "config.json").read_text(encoding="utf-8"))
clases, nombres = config["clases"], config["nombres_es"]

modelo = Net(n_clases=len(clases))
modelo.load_state_dict(torch.load(RAIZ / "models" / "frutas_cnn.pth", map_location="cpu"))
modelo.eval()

transform = transforms.Compose([
    transforms.Resize((config["img_size"], config["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(config["mean"], config["std"]),
])

def predecir(img):
    x = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0]
    resultado = {nombres[c]: float(p) for c, p in zip(clases, probs)}
    top = clases[int(probs.argmax())]
    veredicto = "❌ Descartar (podrida)" if top.startswith("rotten") else "✅ Apta para venta"
    return resultado, f"## {veredicto}"

demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="pil", label="Foto de la fruta", sources=["upload", "webcam"]),
    outputs=[gr.Label(num_top_classes=3, label="Predicción"), gr.Markdown()],
    title="🍎 Clasificador de Frutas Frescas vs Podridas",
    description="CNN entrenada con PyTorch · Proyecto de IA — UTEPSA",
)

if __name__ == "__main__":
    demo.launch()
```

---

## 8. Puntos para la defensa (Claude Code: asegúrate de que el notebook los explique)

- **Qué es una convolución**: un filtro 3x3 que recorre la imagen y detecta patrones (bordes, manchas, texturas). Las primeras capas detectan cosas simples y las profundas, más complejas.
- **Por qué `padding=1`**: mantiene el tamaño espacial después de la convolución.
- **MaxPool 2x2**: reduce a la mitad el ancho y el alto (128 → 64 → 32 → 16), con menos cómputo y algo de invarianza a la posición.
- **BatchNorm**: normaliza las activaciones por batch, entrena más estable y más rápido.
- **Dropout**: apaga neuronas al azar al entrenar para reducir el overfitting.
- **Por qué `128*16*16`** en `fc1`: 128 filtros × 16 × 16 después de 3 poolings. Ahí están ~16.8M de los ~17M de parámetros del modelo.
- **Data augmentation**: flips y rotaciones generan "nuevas" imágenes, así el modelo generaliza mejor.
- **Train vs validación vs test**: validación para elegir el mejor epoch, test solo al final para la nota honesta.
- **CrossEntropyLoss y softmax**: el modelo devuelve logits; softmax los convierte en probabilidades.
- **Transfer learning**: reutiliza filtros aprendidos con millones de imágenes (ImageNet). Da más accuracy con menos datos y menos epochs.
- **¿Cómo mejorar la accuracy?** Más augmentation, más epochs con scheduler, `AdaptiveAvgPool2d` en lugar de un `fc1` gigante (menos parámetros y menos overfitting), imágenes más grandes, o transfer learning.

---

## 9. Reglas para Claude Code

- Trabajar **una fase a la vez** y al final de cada una resumir en 3–5 líneas qué se hizo y qué debe correr o revisar Mirko.
- **No inventar resultados**: las métricas que se escriban en el README y en las conclusiones deben salir de ejecuciones reales.
- **No subir a git** `data/`, `.venv/` ni modelos pesados (salvo que Mirko lo pida).
- Ante un error, explicar la causa en una línea y corregir. No reescribir notebooks enteros por un error puntual.
- Si algo del dataset no coincide con este archivo (nombres de carpetas, conteos), **actualizar este CLAUDE.md** para que quede como fuente de verdad.
- Mantener los notebooks **ejecutables de arriba hacia abajo** (Restart & Run All) sin errores.
