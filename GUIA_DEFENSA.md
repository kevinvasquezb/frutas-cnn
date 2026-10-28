# Guía completa del proyecto: Clasificador de Frutas Frescas vs Podridas (CNN con PyTorch)

---

## 0. INSTRUCCIONES PARA EL ASISTENTE (léelas primero)

Eres mi **tutor de Deep Learning** y mi **jurado de práctica**. Soy Kevin y **mañana defiendo este proyecto** ante mi profesor de Inteligencia Artificial (UTEPSA, Santa Cruz, Bolivia). Necesito **entenderlo de verdad** y poder explicarlo celda por celda, no solo memorizarlo.

Todo lo que necesitas saber del proyecto está en este documento. Los números son **resultados reales** de las ejecuciones; no inventes otros.

**Cómo quiero que trabajes conmigo (en español):**
1. **Primero, enséñame la teoría en orden** (secciones 5 a 7), un tema a la vez. En cada tema: explícalo simple, con una analogía, conéctalo con **mi código y mis números**, y termina con **1 o 2 preguntas cortas** para comprobar que entendí. **Espera mi respuesta** antes de seguir; si me equivoco, corrígeme con paciencia.
2. Cuando terminemos la teoría, **recorremos los notebooks** (sección 4) como si yo los presentara: me preguntas qué hace cada sección y por qué.
3. Al final, hacemos el **simulacro de defensa** (sección 9). Tú eres el profesor: preguntas **una por una** y esperas mi respuesta. Después de cada respuesta me das una nota del 1 al 10, me dices qué faltó y cómo lo diría mejor. Mezcla preguntas fáciles, difíciles y "trampa".
4. Si te pido "modo rápido", resume con viñetas. Si te pido "más detalle", profundiza con fórmulas y ejemplos numéricos.
5. Si algo no está en este documento, dilo en vez de suponer.

---

## 1. Resumen en 30 segundos (para abrir la defensa)

> "Entrené una red neuronal convolucional que mira una foto de una manzana, banana o naranja y decide si está **fresca o podrida** (6 clases). El caso de uso es el **control de calidad** en mercados: una cámara decide si la fruta se vende o se descarta. Seguí el pipeline del Assignment 4 del profesor y le agregué GPU, validación, curvas y métricas por clase. Después lo comparé con **transfer learning (ResNet18)**. Ambos sacaron ~99% en el test de Kaggle, pero al probar con mi webcam **fallaba**. Descubrí que el dataset tiene **data leakage** y que había **domain shift**. Lo corregí reentrenando con **fotos propias**: en mis fotos de prueba pasó de **66.7% a 100%**. Al final hice una **página web que detecta la fruta en vivo con la cámara**, corriendo el modelo dentro del navegador."

---

## 2. El proyecto en números (todo real)

| Qué | Valor |
|---|---|
| Hardware | Laptop con RTX 4070 (8 GB VRAM), 32 GB de RAM, i7 de 14.ª generación, Windows 11 |
| Software | Python 3.13.4, PyTorch 2.14.1 + CUDA 13.0, torchvision, scikit-learn, ONNX, Gradio, OpenCV |
| Dataset | *Fruits fresh and rotten for classification* (Kaggle): 13,599 imágenes .png, 6 clases |
| Split de Kaggle | 10,901 de train y 2,698 de test |
| Nuestro split | De train sacamos 15% de validación: **9,266 de train, 1,635 de validación** y 2,698 de test |
| **CNN propia (notebook 01)** | **99.07%** en test (2,673/2,698, 25 errores) · 16,875,526 parámetros · 67.5 MB · 3.0 min (20 epochs) |
| **ResNet18 (notebook 02)** | **99.85%** en test (2,694/2,698, 4 errores) · 11,179,590 parámetros · 44.8 MB · 1.4 min (7 epochs) |
| **ResNet18 + fotos propias (notebook 03)** | Fotos propias de test: **66.7% → 100%** (9 fotos) · sigue con 99.85% en Kaggle · 7 clases (se agrega "nada") |

### Conteo del dataset por clase

| Clase (carpeta) | En español | Train | Test | Total |
|---|---|---:|---:|---:|
| freshapples | Manzana fresca | 1,693 | 395 | 2,088 |
| freshbanana | Banana fresca | 1,581 | 381 | 1,962 |
| freshoranges | Naranja fresca | 1,466 | 388 | 1,854 |
| rottenapples | Manzana podrida | 2,342 | 601 | 2,943 |
| rottenbanana | Banana podrida | 2,224 | 530 | 2,754 |
| rottenoranges | Naranja podrida | 1,595 | 403 | 1,998 |
| **Total** | | **10,901** | **2,698** | **13,599** |

---

## 3. Historia del proyecto (qué hicimos y en qué orden)

1. **Fase 1, entorno y datos.**
   - Creé un entorno virtual, instalé PyTorch con CUDA y verifiqué que use la GPU (`torch.cuda.is_available() == True`).
   - El dataset venía **duplicado** (`dataset/` y `dataset/dataset/`): dejé una sola copia en `data/train` y `data/test`.
   - Abrí y decodifiqué cada imagen: **ninguna corrupta**.
2. **Hallazgo importante del dataset (data leakage).** Por los nombres de archivo:
   - Solo **~11% son fotos originales** (1,182 en train y 329 en test).
   - El resto son **copias aumentadas por el autor**: `rotated_by_15_`, `rotated_by_30_`, `rotated_by_45_`, `rotated_by_60_`, `rotated_by_75_`, `translation_`, `vertical_flip_` y `saltandpepper_` (ruido).
   - **Las 2,698 imágenes de test tienen el mismo nombre base que alguna de train**: el test tiene versiones rotadas o volteadas de fotos que la red ya vio al entrenar. **Por eso el ~99% es optimista.**
3. **Fase 2, notebook 01:** la CNN del profesor adaptada a 6 clases → **99.07%**.
4. **Fase 3, notebook 02:** transfer learning con ResNet18 → **99.85%**, con menos parámetros y en menos tiempo.
5. **Fase 4:** elegí ResNet18. La exporté a **ONNX** (formato estándar) y verifiqué que da lo mismo que PyTorch (diferencia máxima ≈ 0.0000019). Hice `src/predecir.py` para predecir desde la terminal.
6. **Fase 5:** una app web con **Gradio** (subir foto o usar la webcam).
7. **El problema real.** Al probar con mi webcam, **predecía mal**. Hubo tres causas:
   - **Domain shift:** las fotos de Kaggle son de estudio, con **fondo blanco**, y mi webcam muestra mi cuarto, mi mano y mi cara. Además, **las naranjas locales son verde-amarillas** y las del dataset son naranja intenso.
   - **Leakage:** el test de Kaggle no medía la generalización real.
   - **No existía la opción "no hay fruta":** con softmax, el modelo siempre elige una clase aunque le muestres una pared.
8. **La solución, notebook 03:**
   - Hice `src/capturar.py`, que toma fotos con la webcam y las guarda por clase (presionas un número y graba 20 segundos).
   - Agregué la clase **"nada"**.
   - Reentrené con un **augmentation más fuerte** y medí el **ANTES/DESPUÉS** en fotos mías que el modelo no vio: **66.7% → 100%**.
9. **Demo final:** `web/index.html`, una **página que abre la cámara y detecta en vivo** (sin presionar botones), o permite **subir una foto**. El modelo corre **dentro del navegador** con onnxruntime-web, sin servidor de IA.

---

## 4. Los archivos y los 3 notebooks, en detalle

### Estructura del proyecto
```
notebooks/
  01_frutas_cnn.ipynb                 CNN desde cero (el proyecto principal, sigue al profesor)
  02_frutas_transfer_learning.ipynb   ResNet18 + tabla comparativa
  03_frutas_fotos_propias.ipynb       Reentrenamiento con fotos propias + clase "nada" + antes/después
src/
  modelo.py     Clase Net (la CNN); la genera el notebook 01 con %%writefile
  datos.py      Dataset propio ListaImagenes; lo genera el notebook 03
  predecir.py   cargar_modelo() y predecir(imagen) -> {clase: probabilidad}; se usa también desde la terminal
  exportar.py   Exporta el modelo a ONNX y verifica que coincide con PyTorch
  capturar.py   Toma fotos con la webcam para crear el dataset propio
app/app.py      App Gradio (subir foto / webcam, top 3 y veredicto)
web/index.html  Cámara en vivo en el navegador + subir foto (la demo principal)
web/servir.py   Servidor local para abrir la página (http://localhost:8000/web/)
models/         config.json (+ los .pth y .onnx, que pesan ~45 MB y NO están en GitHub)
outputs/        Gráficos y métricas reales (curvas, matrices de confusión, errores, JSON)
```

### Notebook 01: CNN desde cero (`01_frutas_cnn.ipynb`)
Sigue **las mismas secciones que el Assignment 4 del profesor** (perros vs gatos) y le agrega mejoras.

| # | Sección | Qué hace | Qué agregamos frente al profesor |
|---|---|---|---|
| 1 | Librerías | torch, torchvision, numpy, matplotlib, seaborn, sklearn | `device = "cuda"` (**GPU**) y **semillas** (42) para reproducibilidad |
| 2 | Hiperparámetros | IMG_SIZE=128, BATCH_SIZE=64, LR=0.001, N_EPOCHS=20, VAL_SPLIT=0.15, NUM_WORKERS=4, QUICK_RUN | Todo en una celda; `QUICK_RUN` sirve para probar en segundos |
| 3 | Transforms | Resize 128, flip horizontal, rotación 10°, ToTensor, Normalize(0.5, 0.5) | `ColorJitter` (brillo y contraste ±20%, **sin tocar el tono**, porque el color delata lo podrido) |
| 4 | Datasets y DataLoaders | `ImageFolder` + `DataLoader` | **Split de validación del 15%** con dos "vistas" de train: con augmentation para entrenar y sin augmentation para validar |
| 5 | Visualizar un batch | `imshow` con desnormalización (`x/2 + 0.5`) y `make_grid` | Etiquetas en español |
| 6 | Modelo `Net` | 3 bloques Conv-BN-ReLU-Pool + FC | Se guarda en `src/modelo.py` con `%%writefile`; se imprimen los parámetros y el recorrido de formas |
| 7 | Loss y optimizador | `CrossEntropyLoss` + `Adam(lr=0.001)` | Igual que el profesor |
| 8 | Entrenamiento | `zero_grad → forward → loss → backward → step` | **Validación por epoch**, historial, **guardar el mejor epoch**, tiempo por epoch |
| 9 | Curvas | — | Loss y accuracy de **train vs validación** (para detectar overfitting) |
| 10 | Evaluación en test | accuracy | `classification_report` (precision, recall y F1 por clase) + **matriz de confusión** |
| 11 | Predicciones visuales | imágenes al azar con predicción | Real vs predicho, % de confianza, verde/rojo |
| 12 | Análisis de errores | — | Imágenes mal clasificadas + confusiones más frecuentes |
| 13 | (Opcional) ¿Qué ve la CNN? | — | **Filtros de conv1** y **feature maps** de conv1, conv2 y conv3 |
| 14 | Fotos propias | — | Predecir fotos de `imagenes_prueba/` (con corrección EXIF de celular) |
| 15 | Guardar | — | `state_dict` + `config.json` + métricas en JSON |
| 16 | Preguntas del assignment | las 10 preguntas | Respondidas con los valores de este proyecto |
| 17 | Conclusiones | — | Con los números reales |

**Arquitectura `Net`** (la del profesor con 6 salidas):
```python
class Net(nn.Module):
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
        self.fc1 = nn.Linear(128 * 16 * 16, 512)   # 32,768 -> 512
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, n_clases)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (3,128,128) -> (32,64,64)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (64,32,32)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (128,16,16)
        x = x.view(x.size(0), -1)                       # aplanar -> 32,768
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)                              # logits (sin softmax)
```
- Parámetros: **16,875,526** en total, de los cuales **16,777,728 están en `fc1` (99.4%)**: 32,768 × 512 + 512 de bias.

**Loop de entrenamiento** (el corazón del proyecto):
```python
for epoch in range(N_EPOCHS):
    net.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()              # 1. borrar gradientes viejos
        outputs = net(inputs)              # 2. forward
        loss = criterion(outputs, labels)  # 3. pérdida
        loss.backward()                    # 4. backpropagation
        optimizer.step()                   # 5. actualizar pesos
    net.eval()
    with torch.no_grad():
        ...  # medir loss y accuracy en validación
    if val_acc > mejor_acc:
        mejores_pesos = copy.deepcopy(net.state_dict())   # guardar el mejor epoch
net.load_state_dict(mejores_pesos)
```

**Resultados reales del notebook 01:**
- **Tiempo:** ~7.5 s por epoch en GPU (el primero tardó 38 s porque arranca los procesos de carga). Total: **3.0 min**.
- **Validación:** subió de 0.870 (epoch 1) a **0.993 (epoch 20, el mejor)**.
- **Overfitting:** las curvas de train y validación **bajan juntas**, así que no se ve. Ojo: la validación también tiene leakage.
- **Test: 99.07%** (25 errores). El F1 por clase va de 0.984 a 0.999.
- **Confusiones más frecuentes:** naranja podrida → manzana podrida (7), manzana podrida → manzana fresca (6) y naranja fresca → naranja podrida (4).

### Notebook 02: transfer learning con ResNet18 (`02_frutas_transfer_learning.ipynb`)
- **Modelo:** `resnet18(weights=ResNet18_Weights.DEFAULT)`, con pesos preentrenados en **ImageNet** (1.2 millones de fotos y 1,000 clases).
- **Transforms:** `Resize(224)` y la normalización de ImageNet, `mean=[0.485, 0.456, 0.406]` y `std=[0.229, 0.224, 0.225]`. **Cambian porque el modelo fue preentrenado así.**
- **Mismo split** que el notebook 01 (misma semilla), para que la comparación sea justa.
- **Etapa 1 (feature extraction):** se congela todo y se cambia `model.fc = nn.Linear(512, 6)`. Se entrena **solo la cabeza: 3,078 parámetros** (512×6 + 6), 3 epochs con lr=0.001. La validación dio 0.946 → 0.964 → 0.972.
- **Etapa 2 (fine-tuning):** se descongela `layer4`. Los entrenables pasan a ser **8,396,806**. Son 4 epochs con **lr=0.0001**, 10 veces menor para no destruir lo aprendido. La validación dio **0.997** → 0.999 → **1.000** → 0.998.
- Se usa una **función `entrenar()`** para no repetir el loop en las dos etapas.

**Tabla comparativa (real):**

| Métrica | CNN propia | ResNet18 |
|---|---:|---:|
| Accuracy en test | 99.07% (25 errores) | **99.85%** (4 errores) |
| Parámetros | 16,875,526 | **11,179,590** |
| Tamaño del .pth | 67.5 MB | **44.8 MB** |
| Tamaño de imagen | 128×128 | 224×224 |
| Epochs | 20 | **7** (3 + 4) |
| Tiempo de entrenamiento | 3.0 min | **1.4 min** |
| Inferencia por imagen (GPU) | **0.63 ms** | 2.74 ms |
| Inferencia por imagen (CPU) | **2.63 ms** | 12.94 ms |

**Conclusión:** ResNet18 gana en casi todo; solo es más lenta, y 13 ms en CPU igual es instantáneo para una app. **¿Por qué tiene menos parámetros siendo más profunda?** Por el *global average pooling*: su capa final recibe 512 números, mientras que nuestro `fc1` recibe 32,768.

### Notebook 03: fotos propias (`03_frutas_fotos_propias.ipynb`)
- **Fotos propias** (webcam, con `src/capturar.py`), **35 en total**:

| Clase | Fotos | Train | Test |
|---|---:|---:|---:|
| Manzana fresca | 11 | 8 | 3 |
| Banana fresca | 7 | 5 | 2 |
| Naranja fresca | 7 | 5 | 2 |
| Naranja podrida | 7 | 5 | 2 |
| Nada (sin fruta) | 3 | 3 | 0 |
| **Total** | **35** | **26** | **9** |

- **Split por tiempo, no al azar:** las fotos de una ráfaga (tomadas con 0.3 s de diferencia) son casi idénticas. Si se reparten al azar, el test tiene "gemelas" de train y vuelve el leakage. Por eso, dentro de cada clase, el **último 30%** en el tiempo va a test.
- **Clases del modelo nuevo (7):** `freshapples, freshbanana, freshoranges, nada, rottenapples, rottenbanana, rottenoranges`. Es orden alfabético, y ese orden define el índice de salida.
- **Dataset propio `ListaImagenes`:** una clase con `__len__` y `__getitem__`, guardada en `src/datos.py`. **Detalle de Windows:** los procesos que cargan datos en paralelo (`num_workers`) no pueden ver clases definidas dentro del notebook. Por eso va en un archivo .py; la primera vez falló justo por esto.
- **Oversampling:** las fotos propias se repiten **8 veces** por epoch (11,109 imágenes por epoch = 10,901 de Kaggle + 26×8). Si no, serían el 0.2% de los datos y casi no influirían.
- **Augmentation más fuerte:** `RandomResizedCrop(224, scale=(0.5, 1))` (zoom y recorte aleatorio), flip, rotación 15° y `ColorJitter(0.3, 0.3, saturation=0.2)`, sin tocar el hue.
- **Entrenamiento:** se parte de nuevo de ResNet18 con pesos de ImageNet. Etapa 1: solo `fc`, 2 epochs, lr 1e-3. Etapa 2: se descongelan **layer3 y layer4**, 5 epochs, lr 1e-4. Tiempo: **1.5 min**. Se usa el modelo del **último epoch**, porque no hay validación propia (pocas fotos).

**Resultado real:**

| Prueba | ANTES (solo Kaggle) | DESPUÉS |
|---|---:|---:|
| Fotos propias con fruta (9 de test) | **66.7%** (6/9) | **100%** (9/9) |
| Test de Kaggle | 99.85% | 99.85% |

- **Honestidad:** son solo 9 fotos de test, no hay fotos de "nada" en el test ni manzanas o bananas podridas propias. La mejora es clara, pero la muestra es chica.
- **Frase clave:** "Cambiar de arquitectura mejoró 0.8 puntos en Kaggle; agregar 26 fotos del mundo real mejoró **33 puntos** en el uso real. **Los datos importan más que el modelo.**"

### La demo en vivo (`web/index.html`)
- Se abre con `python web/servir.py` → **http://localhost:8000/web/**. La cámara del navegador solo funciona en localhost o https; por eso hay un servidor y no basta con hacer doble clic.
- **onnxruntime-web** (desde CDN) carga `models/frutas_resnet18_propias.onnx` y **corre en el navegador**.
- **En cada cuadro de la cámara:**
  1. Se recorta el **cuadrado central (80%)**, el mismo recuadro que usa `capturar.py`.
  2. Se redimensiona a 224×224.
  3. Se normaliza con la media y la desviación de ImageNet, en formato NCHW.
  4. Se ejecuta el modelo y se aplica softmax.
- **Suavizado:** un promedio móvil de las probabilidades, para que el resultado no "parpadee".
- **Veredicto:**
  - Verde **"✅ Apta para venta"** o rojo **"❌ Descartar (podrida)"**.
  - Si la clase es "nada" o la confianza es **< 60%**: "Acerca una fruta al recuadro".
- Muestra el **top 3** con barras y los milisegundos por cuadro.
- El botón **"📁 Subir foto"** clasifica una imagen subida (por ejemplo, fotos de fruta podrida de Google).

---

## 5. Teoría I: cómo "ve" una CNN

### 5.1 La imagen como tensor
- Una foto RGB es un tensor de forma **(3, alto, ancho)**, con valores de 0 a 255.
- `ToTensor()` los pasa a [0, 1] y cambia el orden a (C, H, W). Matplotlib usa (H, W, C); por eso al mostrar se hace `np.transpose(img, (1, 2, 0))`.
- Un **batch** tiene forma (64, 3, 128, 128).
- **Normalizar:** `(x − mean) / std`. En la CNN propia es `(x − 0.5)/0.5`, que lleva los valores a [−1, 1]. Con datos centrados en 0, los gradientes son estables y se entrena mejor. Para mostrar la imagen hay que **desnormalizar**: `x*0.5 + 0.5` (el profesor lo escribe `x/2 + 0.5`).

### 5.2 La convolución
- Un **filtro (kernel) 3×3** es una matriz chica de pesos (en RGB son 3×3×3 = 27 pesos más 1 bias).
- **Se desliza** por la imagen; en cada posición multiplica elemento a elemento y suma, y sale un número.
- **Salida alta = "aquí está mi patrón"** (un borde, una mancha, una textura).
- La salida completa de un filtro es un **feature map**. `conv1` tiene 32 filtros, así que genera 32 mapas.
- **Los pesos se aprenden**, no se diseñan a mano.
- **Pesos compartidos:** el mismo filtro recorre toda la imagen. Por eso `conv1` tiene solo 3·3·3·32 + 32 = **896 parámetros**. Una capa densa sobre la imagen completa tendría millones.
- **Jerarquía:** conv1 detecta bordes y colores; conv2 y conv3 **combinan** eso en patrones complejos, como el moho, una mancha marrón o una cáscara arrugada.
- **`padding=1`:** un marco de ceros alrededor, para que la salida mida lo mismo que la entrada (128 → 128).
- **Fórmula del tamaño de salida:** `(N − K + 2P)/S + 1`. Con N=128, K=3, P=1 y S=1 da (128 − 3 + 2)/1 + 1 = **128**.

### 5.3 ReLU
- `ReLU(x) = max(0, x)`.
- Es la **no linealidad**: sin ella, 10 capas lineales equivalen a 1 sola, porque una composición de funciones lineales sigue siendo lineal.
- Es barata y no se satura para x > 0, a diferencia de la sigmoide, cuyos gradientes se vuelven casi 0.

### 5.4 MaxPool 2×2
- Toma el **máximo** de cada cuadrito de 2×2 con paso 2, así que el alto y el ancho quedan a la **mitad**: 128 → 64 → 32 → 16.
- **Ventajas:** 4 veces menos datos por capa (menos cómputo), algo de **invarianza** a pequeños desplazamientos y un **campo receptivo** más grande (cada neurona "ve" más área de la imagen original).
- **No tiene parámetros.**
- **Por qué los filtros se duplican (32 → 64 → 128):** se pierde resolución, y se compensa con más **variedad** de patrones.

### 5.5 BatchNorm
- Normaliza cada canal con la **media y la varianza del batch**, y luego aplica una escala (γ) y un desplazamiento (β) aprendidos.
- **Para qué:** cada capa recibe valores en una escala estable, así que se entrena **más rápido**, más estable y tolera un learning rate más alto.
- En **`train()`** usa las estadísticas del batch; en **`eval()`** usa promedios acumulados. Por eso **`net.eval()` es obligatorio al evaluar**.
- **Detalle:** `BatchNorm1d` en modo train falla con un batch de 1 imagen (no puede calcular la varianza). Por eso, para pasar una sola imagen, se usa `eval()`.

### 5.6 Dropout(0.25)
- Al entrenar, **apaga al azar el 25% de las neuronas** en cada paso.
- La red no puede depender de unas pocas: aprende representaciones redundantes y **sobreajusta menos**. Es como un equipo donde nadie es imprescindible.
- En `eval()` se desactiva.

### 5.7 Aplanar y la capa fully connected
- Después de 3 bloques quedan (128, 16, 16) = **32,768** valores. `view(x.size(0), -1)` los aplana a un vector por imagen.
- `fc1`: 32,768 × 512 = **16.8M de pesos**, el 99.4% del modelo. Esa es su **debilidad**: mucha memoria, más riesgo de overfitting, y depende de la posición exacta de las cosas.
- **Mejora típica:** `AdaptiveAvgPool2d(1)` promedia cada mapa a 1 número. Quedan 128 entradas en vez de 32,768. Es lo que hace ResNet.

### 5.8 Logits, softmax y CrossEntropyLoss
- `fc2` devuelve **6 logits**: puntajes crudos, que pueden ser negativos.
- **Softmax:** `p_i = e^{z_i} / Σ_j e^{z_j}` da probabilidades que suman 1.
  - Ejemplo: logits [2, 1, 0] → probabilidades ≈ [0.67, 0.24, 0.09].
- **CrossEntropyLoss** = `−log(p_correcta)`.
  - Si p = 0.9, la loss es 0.105; si p = 0.5, es 0.69; si p = 0.01, es **4.6**.
  - Castiga fuerte los errores con alta confianza.
- **No ponemos softmax en `forward`:** `CrossEntropyLoss` lo aplica por dentro (log-softmax, numéricamente estable). Softmax solo se usa para **mostrar** los %.

---

## 6. Teoría II: cómo aprende

### 6.1 Gradiente y backpropagation
- **Gradiente** de la loss respecto a un peso: cuánto y en qué dirección cambia la loss si mueves ese peso.
- **Backpropagation:** calcula los gradientes de **todos** los pesos aplicando la **regla de la cadena** desde la salida hacia atrás, capa por capa. PyTorch lo hace solo con **autograd**, al llamar a `loss.backward()`.
- **Descenso de gradiente:** `w ← w − lr · ∂loss/∂w`. Es como bajar una montaña con niebla, dando pasos en la dirección de mayor pendiente.

### 6.2 Los 5 pasos del loop (y por qué cada uno)
1. **`optimizer.zero_grad()`:** PyTorch **acumula** gradientes. Si no los borras, se suman con los del batch anterior y el paso sale mal.
2. **`outputs = net(inputs)`:** forward, la red predice.
3. **`loss = criterion(outputs, labels)`:** mide el error.
4. **`loss.backward()`:** calcula los gradientes.
5. **`optimizer.step()`:** actualiza los pesos.

### 6.3 Adam
- Es una versión mejorada del descenso de gradiente:
  - **Momentum** (β1 = 0.9): promedia los gradientes recientes, como una pelota que rueda; suaviza el ruido y acelera en direcciones consistentes.
  - **Learning rate adaptativo** por peso (β2 = 0.999): los pesos con gradientes grandes o ruidosos dan pasos más chicos.
- **lr = 0.001** es el valor estándar.
- **Pregunta del assignment:** "¿qué momentum usa el optimizador?". El enunciado del profesor dice SGD, pero **su código usa Adam**, y Adam no tiene un parámetro `momentum`. Su equivalente es `betas = (0.9, 0.999)`.

### 6.4 Batch, epoch e iteración
- **Batch** = 64 imágenes, es decir, 1 paso del optimizador.
- **Epoch** = una pasada por todo train: 9,266 / 64 ≈ **145 pasos**.
- **Batch más grande:** gradiente más estable y más rápido en GPU, pero usa más memoria.

### 6.5 Learning rate
- **Muy grande:** "salta" y no converge, o diverge.
- **Muy chico:** aprende lentísimo.
- **En fine-tuning se usa más chico** (1e-4) para no destruir los pesos preentrenados.

---

## 7. Teoría III: evaluar bien y mejorar

### 7.1 Train / validación / test
- **Train:** la red aprende de estas imágenes.
- **Validación:** sirve para **decidir** (elegir el mejor epoch, ajustar hiperparámetros). La red no aprende de ellas.
- **Test:** se usa **una sola vez, al final**, para la nota honesta. Si eliges usando test, el número queda optimista.
- **Truco del notebook:** dos "vistas" de la carpeta train, una con augmentation para entrenar y otra sin augmentation para validar, y se reparten los índices con `Subset`.

### 7.2 Overfitting y underfitting
- **Overfitting:** la loss de train baja y la de validación **sube o se estanca**. La red **memoriza**.
- **Underfitting:** ambas quedan altas; el modelo es muy simple o le faltan epochs.
- **En este proyecto:** las curvas bajan juntas y el mejor epoch fue el último, así que **no hubo overfitting visible**.
- **¿Por qué la validación a veces supera a train?** En train hay augmentation y dropout, así que la tarea es más difícil.

### 7.3 Data augmentation
- Se transforma cada imagen al azar en **cada epoch** (flip, rotación, brillo, recorte). Es como tener más datos gratis y ayuda a generalizar.
- **No se aplica en test.**
- **No tocamos el hue** porque el color es la señal de "podrido".

### 7.4 Métricas
- **Accuracy:** % de aciertos. Puede engañar con clases desbalanceadas.
- **Precision (clase X):** de las que dije X, cuántas eran X.
- **Recall (clase X):** de las X reales, cuántas encontré.
  - **En control de calidad importa el recall de las clases podridas:** una fruta podrida que pasa como fresca es el error más caro.
- **F1:** media armónica de precision y recall.
- **Matriz de confusión:** filas = real, columnas = predicho. La diagonal son los aciertos.

### 7.5 Transfer learning
- Se **reutiliza** una red entrenada con ImageNet: sus filtros ya saben "ver" bordes, texturas y formas.
- **Feature extraction:** se congela todo (`requires_grad = False`) y se entrena solo la última capa.
- **Fine-tuning:** se descongelan las últimas capas (las más específicas) con un lr chico.
- **Ventajas:** más accuracy con menos datos y menos epochs.
- **Catastrophic forgetting:** si el lr es muy grande, se "olvida" lo aprendido.
- **ResNet y las conexiones residuales:** cada bloque calcula `salida = x + F(x)`. El gradiente fluye por el atajo y se pueden entrenar redes muy profundas sin que el gradiente **desaparezca** (*vanishing gradient*).

### 7.6 Data leakage y domain shift (la lección central)
- **Data leakage:** información del test "se filtra" en train. En este dataset, el test tiene copias rotadas o volteadas de fotos de train, así que el 99% **no mide la generalización real**.
- **Domain shift:** los datos de uso son distintos a los de entrenamiento (fondo blanco de estudio contra la webcam del cuarto, y naranjas verdes locales). El modelo bajaba a **66.7%**.
- **Soluciones aplicadas:** datos del dominio real, oversampling, augmentation fuerte, la clase "nada" y una **evaluación honesta con split por tiempo**.

### 7.7 Despliegue
- **`state_dict`:** solo los pesos; la arquitectura está en el código (`src/modelo.py` o `resnet18()`).
- **`config.json`:** orden de las clases (sale de `ImageFolder.classes`, **nunca a mano**), img_size, mean y std. Al predecir, el preprocesamiento debe ser **idéntico** al de test; si no, el modelo falla.
- **ONNX:** formato estándar que corre sin PyTorch (en el navegador, en el celular o en otros lenguajes). Se verificó con `np.allclose(..., atol=1e-4)` que da lo mismo que PyTorch.
- **EXIF:** las fotos de celular traen la orientación en metadatos; `ImageOps.exif_transpose` las endereza.

---

## 8. Las 10 preguntas del Assignment del profesor (respondidas para este proyecto)

1. **¿Qué learning rate usa el optimizador?** 0.001. El enunciado dice SGD, pero el código usa **Adam** (lr=0.001).
2. **¿Qué momentum?** Adam no tiene un parámetro `momentum`; su equivalente es `betas=(0.9, 0.999)`, donde 0.9 actúa como momentum.
3. **¿Qué batch size?** 64 (el profesor usa 32). La GPU de 8 GB lo soporta y cada epoch tiene la mitad de pasos.
4. **¿Cuántos epochs?** 20, y se guarda el mejor según la validación.
5. **¿Cómo se definen los filtros?** 32 → 64 → 128: se duplican en cada bloque para compensar la reducción espacial del pooling.
6. **¿Tamaño de imagen?** 128×128 (224×224 en ResNet18).
7. **¿Normalización?** mean = std = (0.5, 0.5, 0.5), que lleva los valores a [−1, 1]. En ResNet18 se usa la de ImageNet.
8. **¿Kernel?** 3×3 con padding=1 en las 3 capas convolucionales.
9. **¿Accuracy?** CNN propia **99.07%**; ResNet18 **99.85%** en el test de Kaggle. Es optimista por el leakage; en fotos propias, ResNet18 pasó de 66.7% a 100% tras reentrenar.
10. **¿Cómo mejorar la accuracy?**
    - Transfer learning (hecho: +0.8 puntos).
    - **Datos del dominio real** (hecho: +33 puntos en fotos propias).
    - `AdaptiveAvgPool2d` en lugar del `fc1` gigante.
    - Más augmentation, más epochs con un scheduler e imágenes más grandes.

---

## 9. SIMULACRO DE DEFENSA

**Instrucciones para el asistente:** haz las preguntas **de a una**, en orden aleatorio, como un profesor exigente pero justo. Después de cada respuesta da una nota del 1 al 10, qué faltó y una respuesta modelo corta. Al final da un resumen con mis puntos fuertes y débiles. Las respuestas modelo de abajo son tu guía: **no me las muestres antes de que yo responda**.

### Básicas
1. **Explica tu proyecto en 1 minuto.** → Sección 1.
2. **¿Qué es una convolución?** → Un filtro 3×3 de pesos aprendidos que se desliza, multiplica y suma; detecta patrones; con pesos compartidos.
3. **¿Para qué sirve el MaxPool?** → Reduce a la mitad el alto y el ancho: menos cómputo, invarianza a desplazamientos y mayor campo receptivo. No tiene parámetros.
4. **¿Qué hace ReLU y por qué es necesaria?** → max(0, x). Es la no linealidad; sin ella, varias capas equivalen a una lineal.
5. **¿Por qué `padding=1`?** → Mantiene el tamaño espacial: (128 − 3 + 2)/1 + 1 = 128.
6. **¿Por qué normalizas las imágenes?** → Valores centrados en 0 y en escala similar: gradientes estables y entrenamiento más rápido.
7. **¿Qué es un epoch y qué es un batch?** → Epoch: una pasada completa por train. Batch: 64 imágenes y un paso del optimizador (~145 pasos por epoch).

### Del código
8. **¿Por qué `128*16*16` en `fc1`?** → 128 filtros × (128/2/2/2 = 16) × 16.
9. **Explica las 5 líneas del loop de entrenamiento.** → zero_grad, forward, loss, backward, step (sección 6.2).
10. **¿Qué pasa si olvidas `optimizer.zero_grad()`?** → Los gradientes se acumulan entre batches y las actualizaciones salen mal.
11. **¿Diferencia entre `net.train()` y `net.eval()`?** → Dropout activo o apagado; BatchNorm con estadísticas del batch o con las acumuladas.
12. **¿Para qué `torch.no_grad()`?** → No construye el grafo de gradientes: más rápido y usa menos memoria al evaluar.
13. **¿Por qué no hay softmax en `forward`?** → `CrossEntropyLoss` ya aplica log-softmax internamente, de forma estable; softmax solo se usa para mostrar %.
14. **¿Qué es `%%writefile` y por qué lo usaste?** → Guarda la celda en `src/modelo.py` para que la app y los scripts usen la misma arquitectura.
15. **¿Dónde están la mayoría de los parámetros de tu red y por qué es un problema?** → En `fc1`, el 99.4% (16.8M): memoria, overfitting y dependencia de la posición.

### De evaluación
16. **¿Por qué separaste validación si el profesor no lo hizo?** → Para elegir el mejor epoch sin "espiar" el test; el test queda para la nota honesta.
17. **¿Tu modelo tiene overfitting? ¿Cómo lo sabes?** → No se ve: las curvas de train y validación bajan juntas y el mejor epoch fue el último. Pero la validación también tiene leakage.
18. **¿Qué es precision y recall? ¿Cuál importa más en tu caso?** → Sección 7.4; el recall de las podridas (que no se cuele fruta podrida).
19. **¿Qué clases confunde más tu modelo?** → Naranja podrida ↔ manzana podrida, y manzana podrida ↔ manzana fresca.
20. **¿Por qué la accuracy de validación a veces supera a la de train?** → En train hay augmentation y dropout.

### Transfer learning
21. **¿Qué es transfer learning y por qué funciona?** → Reutiliza filtros aprendidos con 1.2M de imágenes de ImageNet; los patrones básicos sirven para cualquier imagen.
22. **¿Por qué cambiaste la normalización y el tamaño para ResNet18?** → Fue preentrenada con 224 y con la media y desviación de ImageNet; hay que darle datos en la misma escala.
23. **¿Por qué dos etapas, y por qué un lr más chico en la segunda?** → Primero se entrena la cabeza aleatoria; después se afinan las capas preentrenadas sin destruirlas (*catastrophic forgetting*).
24. **¿Cómo tiene ResNet18 menos parámetros siendo más profunda?** → Global average pooling: 512 entradas a la capa final en vez de 32,768.
25. **¿Qué es una conexión residual?** → `x + F(x)`: el gradiente fluye por el atajo y se evita el *vanishing gradient*.
26. **¿Cuál modelo elegirías para producción y por qué?** → ResNet18: más accuracy, menos MB y menos entrenamiento; 13 ms en CPU es suficiente.

### Preguntas "trampa" (las más importantes)
27. **"Sacaste 99%: ¿tu modelo es perfecto?"** → No. El test de Kaggle tiene leakage (copias rotadas de fotos de train). Con mi webcam bajaba a 66.7%. El 99% mide reconocer variantes de fotos conocidas, no generalizar.
28. **"¿Cómo descubriste el leakage?"** → Por los nombres de archivo: prefijos como `rotated_by_15_`. Las 2,698 imágenes de test comparten el nombre base con alguna de train.
29. **"¿Por qué fallaba con la webcam?"** → Domain shift: fotos de estudio con fondo blanco contra mi cuarto, mi mano y otra luz; además, las naranjas locales son verdes.
30. **"¿Cómo lo arreglaste?"** → Fotos propias + clase "nada" + oversampling + augmentation más fuerte; se reentrenó y pasó de 66.7% a 100% en mis fotos de test.
31. **"9 fotos de test es muy poco, ¿no?"** → Sí, lo reconozco. La mejora es clara pero la muestra es chica; el siguiente paso es capturar cientos de fotos variadas y armar una validación propia.
32. **"¿Por qué separaste tus fotos por tiempo y no al azar?"** → Las fotos de una ráfaga son casi idénticas; al azar habría "gemelas" en train y test, que es leakage otra vez.
33. **"¿Qué pasa si le muestras una pera?"** → Antes elegía una de las 6 clases a la fuerza (softmax siempre elige). Ahora existe la clase "nada" y, con confianza < 60%, la página dice "acerca una fruta". Aun así, una pera podría confundirse.
34. **"¿Qué es lo más importante que aprendiste?"** → Que los datos importan más que el modelo: cambiar de arquitectura dio +0.8 puntos y agregar datos reales dio +33.

### Demo
35. **"Muéstrame que funciona."** → Abrir la página en vivo, mostrar una fruta (veredicto verde), la mesa vacía ("acerca una fruta") y subir una foto de fruta podrida de Google (rojo).
36. **"¿Cómo corre el modelo en el navegador?"** → Exporté a ONNX; onnxruntime-web lo ejecuta en JavaScript. En cada cuadro: recorte central → 224 → normalización de ImageNet → modelo → softmax → suavizado.

---

## 10. Guion para la demo de mañana

**Antes de la defensa, en la PC del proyecto.** Los modelos `.pth` y `.onnx` NO están en GitHub, así que la demo debe hacerse **desde esta laptop**.
1. Cierra todo lo que use la cámara: Zoom, Teams, otras pestañas o la app Cámara de Windows. Si no, la imagen sale negra.
2. Abre una terminal en `C:\dev\frutas-cnn` y ejecuta: `.\.venv\Scripts\python.exe web\servir.py`. Se abre `http://localhost:8000/web/`.
3. Presiona **"Iniciar cámara"** y acepta el permiso.
4. Ten listas en el escritorio 2 o 3 fotos de **fruta podrida descargadas de Google**, para "📁 Subir foto".

**Orden sugerido (unos 10 minutos):**
1. **Problema y caso de uso** (30 s).
2. **Notebook 01:** muestra el modelo (`print(net)`, parámetros), el loop, las curvas, la matriz de confusión y las imágenes mal clasificadas. Si preguntan, los filtros y feature maps.
3. **Notebook 02:** la tabla comparativa.
4. **El giro:** "pero con la webcam fallaba". Explica el leakage y el domain shift.
5. **Notebook 03:** el ANTES/DESPUÉS (66.7% → 100%).
6. **Demo en vivo:** fruta fresca, mesa vacía y foto podrida subida.
7. **Cierre:** "los datos importan más que el modelo" y cómo lo mejorarías.

**Otras formas de usar el modelo:** `python app/app.py` (Gradio, http://127.0.0.1:7860) y `python src/predecir.py foto.jpg` (terminal).

---

## 11. Limitaciones honestas (decirlas suma puntos)
- El test de Kaggle tiene **data leakage**, así que su ~99% es optimista.
- **Pocas fotos propias** (35 en total, 9 de test); no hay manzanas ni bananas podridas propias, ni "nada" en el test.
- Solo conoce **3 frutas**; otras cosas pueden confundirlo.
- **Fruta a medio pudrir** es el caso más difícil: el dataset tiene sobre todo frutas claramente frescas o claramente podridas.
- **Una fruta por foto**: no detecta varias frutas a la vez. Eso sería *detección de objetos* (por ejemplo, YOLO), no clasificación.
- En el notebook 03 no hay validación propia, así que se usó el último epoch.

**Trabajo futuro:** cientos de fotos reales variadas, otro dataset con fondos reales (por ejemplo FruitNet), validación propia, detección de varias frutas y una app en el celular.

---

## 12. Glosario rápido
- **Tensor:** arreglo multidimensional de números (lo que procesa PyTorch).
- **Kernel/filtro:** matriz de pesos que se desliza en la convolución.
- **Feature map:** salida de un filtro; indica dónde aparece su patrón.
- **Logits:** puntajes crudos de salida, antes de softmax.
- **Softmax:** convierte los logits en probabilidades que suman 1.
- **Loss:** número que mide qué tan mal predice la red.
- **Gradiente:** derivada de la loss respecto a un peso.
- **Backpropagation:** cálculo de todos los gradientes con la regla de la cadena.
- **Optimizador (Adam):** regla para actualizar los pesos con los gradientes.
- **Learning rate:** tamaño del paso de actualización.
- **Epoch / batch:** pasada completa por los datos / grupo de imágenes por paso.
- **Overfitting:** memorizar train y fallar en datos nuevos.
- **Data augmentation:** transformaciones aleatorias para "crear" más datos.
- **BatchNorm:** normaliza las activaciones por batch para un entrenamiento estable.
- **Dropout:** apaga neuronas al azar para reducir el overfitting.
- **Transfer learning:** reutilizar una red preentrenada.
- **Fine-tuning:** reentrenar parte de una red preentrenada con un lr chico.
- **Data leakage:** información del test que se filtra en train y da resultados falsamente altos.
- **Domain shift:** los datos reales difieren de los de entrenamiento.
- **Oversampling:** repetir los ejemplos escasos para que pesen más.
- **ONNX:** formato estándar para ejecutar modelos fuera de PyTorch.
- **state_dict:** diccionario con los pesos aprendidos de un modelo.
