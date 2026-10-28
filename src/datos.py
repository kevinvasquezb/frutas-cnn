# src/datos.py — Dataset a partir de una lista de (ruta, etiqueta). Generado desde notebooks/03_frutas_fotos_propias.ipynb
from PIL import Image  # Para abrir las imágenes
from torch.utils.data import Dataset  # Clase base de los datasets de PyTorch


class ListaImagenes(Dataset):
    """Dataset a partir de una lista de (ruta, etiqueta)."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform  # Guardamos la lista y el transform

    def __len__(self):
        return len(self.items)  # Cantidad de imágenes

    def __getitem__(self, i):
        ruta, etiqueta = self.items[i]
        img = Image.open(ruta).convert("RGB")  # Abre y asegura 3 canales
        return self.transform(img), etiqueta  # (tensor, índice de clase)
