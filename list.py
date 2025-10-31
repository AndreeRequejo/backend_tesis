"""Script simple que calcula el SHA-256 de una imagen y lo añade a `blacklist.json`.

Edite la constante IMAGE_PATH abajo para indicar la imagen que desea bloquear.
El script evita sobrescribir datos existentes y solo añade el hash si no está presente.
"""
from pathlib import Path
import hashlib
import json

# --- CONFIGURACIÓN SIMPLE -------------------------------------------------
# Ruta de la imagen a añadir a la blacklist. Editar según necesidad.
IMAGE_PATH = Path("moana3.webp")

# Archivo blacklist en la raíz del proyecto
BLACKLIST_PATH = Path(__file__).resolve().parent / "blacklist.json"
# -------------------------------------------------------------------------


def compute_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_or_create_blacklist(p: Path) -> dict:
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        # Asegurar estructura mínima
        if isinstance(data, dict) and isinstance(data.get("hashes"), list):
            return data
        return {"hashes": []}
    # Crear archivo si no existe
    p.write_text(json.dumps({"hashes": []}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"hashes": []}


def save_blacklist(p: Path, data: dict):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    img = IMAGE_PATH
    if not img.exists():
        print(f"Error: archivo no encontrado: {img}")
        return

    h = compute_sha256(img)

    data = load_or_create_blacklist(BLACKLIST_PATH)
    hashes = data.get("hashes")

    if h in hashes:
        print(f"Hash ya existente en blacklist.json: {h}")
        return

    hashes.append(h)
    data["hashes"] = hashes
    save_blacklist(BLACKLIST_PATH, data)

    print(f"Hash agregado a blacklist.json: {h}")


if __name__ == "__main__":
    main()
