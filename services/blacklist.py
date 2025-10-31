"""
Módulo simple para manejar la lista negra de imágenes (hashes SHA-256).
Archivo JSON esperado: `blacklist.json` en la raíz del proyecto.
Formato:
{
  "hashes": ["abc123...", "def456..."]
}

Funcionalidad mínima requerida por el usuario:
- Cargar la lista de hashes al importar
- is_image_blacklisted(image_bytes) -> bool

No se añade ninguna funcionalidad adicional ni endpoints de administración.
"""
import json
import hashlib
from pathlib import Path
from typing import Set
import logging

logger = logging.getLogger(__name__)

# Archivo por defecto con lista negra (puede editarse manualmente)
BLACKLIST_FILE = Path(__file__).resolve().parents[1] / "blacklist.json"


def _load_blacklist() -> Set[str]:
    """Cargar hashes desde el archivo JSON. Si no existe, crear uno vacío."""
    try:
        if not BLACKLIST_FILE.exists():
            # Crear archivo vacío con estructura básica
            BLACKLIST_FILE.write_text(json.dumps({"hashes": []}, indent=2, ensure_ascii=False))
            return set()

        data = json.loads(BLACKLIST_FILE.read_text(encoding="utf-8"))
        hashes = data.get("hashes", []) if isinstance(data, dict) else []
        return set(hashes)
    except Exception as e:
        logger.error(f"Error cargando blacklist desde {BLACKLIST_FILE}: {e}")
        return set()


# Cargar al importar el módulo
_BLACKLIST_HASHES = _load_blacklist()


def is_image_blacklisted(image_bytes: bytes) -> bool:
    """Calcular SHA-256 de los bytes de la imagen y comprobar si está en la lista negra."""
    try:
        h = hashlib.sha256(image_bytes).hexdigest()
        return h in _BLACKLIST_HASHES
    except Exception as e:
        logger.error(f"Error calculando hash para verificación de blacklist: {e}")
        # En caso de error, no bloquear por seguridad (comportamiento conservador)
        return False
