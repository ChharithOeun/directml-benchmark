@echo off
REM install.bat — Windows setup for directml-benchmark
REM Requires Python 3.11 for DirectML (hard ABI ceiling).
REM For Linux/macOS: use install.sh instead.
REM
REM Usage:
REM   install.bat              — install DirectML (default, Python 3.11 required)
REM   install.bat --cpu        — CPU-only install (any Python version)
REM   install.bat --cuda       — CUDA install (NVIDIA GPU)
REM   install.bat --check      — check environment only

setlocal EnableDelayedExpansion
set MODE=directml
set VENV=.venv311

for %%A in (%*) do (
    if "%%A"=="--cpu"   set MODE=cpu
    if "%%A"=="--cuda"  set MODE=cuda
    if "%%A"=="--check" set MODE=check
)

echo.
echo ════════════════════════════════════════════════════
echo   directml-benchmark installer ^(Windows^)
echo ════════════════════════════════════════════════════
echo.

if "%MODE%"=="check" (
    python benchmark.py --check
    goto :end
)

if "%MODE%"=="directml" (
    echo [INFO] DirectML mode — requires Python 3.11
    echo.

    REM Find Python 3.11
    where py >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python Launcher ^(py^) not found.
        echo         Install Python 3.11 from https://python.org
        echo         Make sure to check "Add to PATH" during install.
        goto :fail
    )

    py -3.11 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3.11 not found.
        echo         Download from: https://python.org/downloads/release/python-3119/
        echo         torch-directml will NOT work on Python 3.12 or newer.
        goto :fail
    )

    for /f "tokens=*" %%V in ('py -3.11 --version 2^>^&1') do (
        echo [OK]   Found %%V
    )

    echo [1/3] Creating venv (.venv311^)...
    if not exist "%VENV%" (
        py -3.11 -m venv %VENV%
    ) else (
        echo       Already exists — skipping creation
    )

    echo [2/3] Installing torch-directml into %VENV%...
    echo       ^(Do NOT pre-install torch — let directml pull torch 2.4.1 automatically^)
    call "%VENV%\Scripts\activate.bat"
    pip install torch-directml

    echo [3/3] Verifying install...
    python benchmark.py --check

    echo.
    echo ════════════════════════════════════════════════════
    echo   Setup complete! To run the benchmark:
    echo.
    echo   %VENV%\Scripts\activate
    echo   python benchmark.py --cpu-only    ^(verify math^)
    echo   python benchmark.py               ^(full GPU benchmark^)
    echo.
    echo   Results saved to: results\DirectML_YYYYMMDD.json
    echo ════════════════════════════════════════════════════
    goto :end
)

if "%MODE%"=="cpu" (
    echo [INFO] CPU-only mode — any Python version supported
    echo.
    echo [1/2] Installing torch ^(CPU^)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    echo [2/2] Verifying install...
    python benchmark.py --check
    goto :end
)

if "%MODE%"=="cuda" (
    echo [INFO] CUDA mode — NVIDIA GPU
    echo.
    echo [1/2] Installing torch+CUDA 12.1...
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    echo [2/2] Verifying install...
    python benchmark.py --check
    goto :end
)

:fail
echo.
echo [FAILED] See error above.
pause
exit /b 1

:end
echo.
pause
