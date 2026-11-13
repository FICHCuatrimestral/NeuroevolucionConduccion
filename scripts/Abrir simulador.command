#!/bin/bash

# Abre el menú del simulador con doble clic, en macOS.
# Usa el Python de .venv si existe; si no, python3.

cd "$(dirname "$0")/.." || exit 1

python_proyecto=".venv/bin/python"
if [ ! -x "$python_proyecto" ]; then
    python_proyecto="python3"
fi

if ! command -v "$python_proyecto" >/dev/null 2>&1; then
    echo "No se encontró python3. Instalá Python 3.11 o posterior."
    read -r -p "Presioná Enter para cerrar..." _
    exit 1
fi

"$python_proyecto" -B scripts/simulador.py
resultado=$?
if [ "$resultado" -ne 0 ]; then
    echo
    echo "Si falta alguna biblioteca, instalá las dependencias con:"
    echo "\"$python_proyecto\" -m pip install -r requirements.txt"
    read -r -p "Presioná Enter para cerrar..." _
fi
exit "$resultado"
