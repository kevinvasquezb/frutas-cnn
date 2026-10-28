# =====================================================================
# src/capturar.py — Toma fotos propias con la webcam para reentrenar el modelo.
# Guarda el recorte del recuadro central en data/propias/<clase>/.
# Uso:  python src/capturar.py
#
# Teclas (con la ventana de la cámara seleccionada):
#   1 manzana fresca   2 banana fresca   3 naranja fresca
#   4 manzana podrida  5 banana podrida  6 naranja podrida
#   0 nada (sin fruta: mesa vacía, tu mano, otros objetos)
#   Al presionar un número empieza a GRABAR esa clase: ~3 fotos/s durante 20 s.
#   Cualquier tecla detiene la grabación antes. q = salir.
# =====================================================================
import time  # Para nombrar los archivos y controlar la grabación
from pathlib import Path  # Rutas

import cv2  # OpenCV: acceso a la webcam y ventanas

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "propias"  # data/ no se sube a git
TECLAS = {  # tecla -> nombre de carpeta (mismos nombres que el dataset de Kaggle + "nada")
    "1": "freshapples", "2": "freshbanana", "3": "freshoranges",
    "4": "rottenapples", "5": "rottenbanana", "6": "rottenoranges",
    "0": "nada",
}
INTERVALO = 0.33  # Segundos entre fotos mientras graba
DURACION = 20  # Segundos que dura cada grabación
CUENTA_ATRAS = 1.5  # Segundos de espera antes de empezar (para acomodar la fruta)


def contar():
    """Cuántas fotos hay ya guardadas por clase."""
    return {c: len(list((DESTINO / c).glob("*.jpg"))) for c in TECLAS.values()}


def abrir_camara():
    """Prueba dos métodos de Windows (DirectShow y Media Foundation) y devuelve el primero que dé imagen no negra."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cam = cv2.VideoCapture(0, backend)
        if not cam.isOpened():
            continue
        for _ in range(30):  # Los primeros cuadros pueden salir negros mientras la cámara ajusta la exposición
            ok, frame = cam.read()
            if ok and frame is not None and frame.mean() > 8:  # Brillo promedio > 8 = hay imagen real
                return cam
            time.sleep(0.05)
        cam.release()
    return None


def main():
    for c in TECLAS.values():
        (DESTINO / c).mkdir(parents=True, exist_ok=True)  # Crea las 7 carpetas
    cam = abrir_camara()
    if cam is None:
        raise SystemExit("La cámara no da imagen (sale negra). Cierra el navegador/Zoom/Teams que la esté usando y reintenta.")

    conteo = contar()
    grabando, inicio, ultima, guardadas = None, 0.0, 0.0, 0  # Clase que se está grabando y tiempos
    while True:
        ok, frame = cam.read()  # Lee un cuadro de la cámara
        if not ok:
            break
        h, w = frame.shape[:2]
        lado = int(min(h, w) * 0.8)  # Recuadro central cuadrado (80% del lado menor)
        y0, x0 = (h - lado) // 2, (w - lado) // 2
        recorte = frame[y0:y0 + lado, x0:x0 + lado]  # Esto es lo que se guarda (y lo que verá el modelo en vivo)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla != 255:  # Se presionó alguna tecla
            if grabando:  # Si estaba grabando, cualquier tecla detiene
                grabando = None
            elif chr(tecla) in TECLAS:  # Si no, un número empieza a grabar esa clase
                grabando, inicio, guardadas = TECLAS[chr(tecla)], time.time(), 0

        ahora = time.time()
        if grabando and ahora - inicio > CUENTA_ATRAS + DURACION:  # Se acabó el tiempo
            grabando = None
        if grabando and ahora - inicio >= CUENTA_ATRAS and ahora - ultima >= INTERVALO:
            ruta = DESTINO / grabando / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(ahora*1000) % 1000:03d}.jpg"
            cv2.imwrite(str(ruta), recorte)  # Guarda el recorte en JPG
            conteo[grabando] += 1
            guardadas += 1
            ultima = ahora

        # ---- Dibujar la interfaz ----
        vista = frame.copy()
        color = (0, 0, 255) if grabando else (0, 255, 0)  # Rojo = grabando, verde = esperando
        cv2.rectangle(vista, (x0, y0), (x0 + lado, y0 + lado), color, 3)
        for i, (k, c) in enumerate(TECLAS.items()):  # Conteo por clase arriba a la izquierda
            marca = ">" if c == grabando else " "
            cv2.putText(vista, f"{marca}{k}: {c:<14} {conteo[c]:3d}", (10, 25 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if grabando:
            t = ahora - inicio
            if t < CUENTA_ATRAS:
                estado = f"Preparate... {grabando}"
            else:
                estado = f"GRABANDO {grabando}: {guardadas} fotos, quedan {int(CUENTA_ATRAS + DURACION - t)}s (tecla = parar)"
        else:
            estado = "Presiona 0-6 para grabar 20 s de esa clase | q = salir"
        cv2.putText(vista, estado, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Captura de fotos propias - frutas", vista)

    cam.release()
    cv2.destroyAllWindows()
    print("Fotos por clase:", contar())


if __name__ == "__main__":
    main()
