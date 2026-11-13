@echo off
rem Abre el menú del simulador con doble clic, en Windows.
rem Usa el Python de .venv si existe; si no, el que esté instalado.

cd /d "%~dp0\.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" -B scripts\simulador.py
if errorlevel 1 (
    echo.
    echo Si falta alguna biblioteca, instala las dependencias con:
    echo "%PYTHON%" -m pip install -r requirements.txt
    pause
)
