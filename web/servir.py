# =====================================================================
# web/servir.py — Sirve el proyecto en http://localhost:8000 y abre la página de cámara en vivo.
# La cámara del navegador solo funciona en "localhost" o https, por eso no basta con abrir el .html con doble clic.
# Uso:  python web/servir.py
# =====================================================================
import functools  # Para fijar la carpeta que se sirve
import http.server  # Servidor web simple de la librería estándar
import webbrowser  # Para abrir el navegador
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent  # Se sirve la raíz para que la página lea ../models/
PUERTO = 8000

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(RAIZ))
with http.server.ThreadingHTTPServer(("127.0.0.1", PUERTO), Handler) as servidor:
    url = f"http://localhost:{PUERTO}/web/"
    print(f"Página de cámara en vivo: {url}   (Ctrl+C para cerrar)")
    webbrowser.open(url)
    servidor.serve_forever()
