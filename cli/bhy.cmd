@echo off
rem Launch bhy from anywhere. Resolves the project relative to this script, so the
rem folder can be moved or added to PATH without editing anything here.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%ROOT%\bhy.py" %*
