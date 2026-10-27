# =====================================================================
# src/modelo.py — Arquitectura de la CNN.
# Este archivo se GENERA desde notebooks/01_frutas_cnn.ipynb (celda con %%writefile).
# Si quieres cambiar la red, cámbiala en el notebook y vuelve a ejecutar esa celda.
# =====================================================================
import torch.nn as nn  # Capas de redes neuronales
import torch.nn.functional as F  # Funciones sin parámetros (relu)


class Net(nn.Module):
    """CNN del maestro (Assignment 4) adaptada a n_clases. Entrada: imágenes RGB de 128x128."""

    def __init__(self, n_clases=6):
        super().__init__()  # Inicializa la clase padre nn.Module (obligatorio)

        # ---- Bloques convolucionales con BatchNorm ----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 canales de entrada (RGB) -> 32 filtros de 3x3; padding=1 mantiene 128x128
        self.bn1 = nn.BatchNorm2d(32)  # Normaliza los 32 mapas que salen de conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 filtros
        self.bn2 = nn.BatchNorm2d(64)  # Normaliza los 64 mapas de conv2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128 filtros
        self.bn3 = nn.BatchNorm2d(128)  # Normaliza los 128 mapas de conv3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling 2x2 con paso 2: reduce alto y ancho a la mitad
        self.dropout = nn.Dropout(0.25)  # Apaga al azar el 25% de las neuronas (solo en entrenamiento)

        # ---- Parte fully connected ----
        # Con entrada de 128x128 y 3 poolings: 128 -> 64 -> 32 -> 16. Salen 128 mapas de 16x16.
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # 32,768 entradas -> 512 neuronas (¡aquí está la mayoría de los parámetros!)
        self.bn_fc = nn.BatchNorm1d(512)  # Normaliza las 512 salidas de fc1
        self.fc2 = nn.Linear(512, n_clases)  # Capa de salida: un puntaje (logit) por clase

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Bloque 1: conv -> BN -> ReLU -> pool   (3,128,128) -> (32,64,64)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Bloque 2                               -> (64,32,32)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Bloque 3                               -> (128,16,16)
        x = x.view(x.size(0), -1)  # Aplanar: (batch, 128, 16, 16) -> (batch, 32768)
        x = self.dropout(x)  # Dropout antes de la capa densa
        x = F.relu(self.bn_fc(self.fc1(x)))  # fc1 -> BN -> ReLU
        x = self.dropout(x)  # Dropout otra vez
        return self.fc2(x)  # Logits (SIN softmax: CrossEntropyLoss ya lo aplica internamente)
