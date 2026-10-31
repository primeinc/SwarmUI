@echo off
setlocal
set "CAPTURE=%~dp0launchtools\qwen_segment_capture.py"
set "UVPY=%APPDATA%\uv\python\cpython-3.13-windows-x86_64-none\python.exe"
if exist "%UVPY%" goto uvpython
where py >nul 2>nul
if not errorlevel 1 goto pylauncher
where python >nul 2>nul
if not errorlevel 1 goto pythonpath
echo No Python interpreter found. Run launchtools\qwen_segment_capture.py with your ComfyUI Python.
exit /b 2
:uvpython
"%UVPY%" "%CAPTURE%" %*
exit /b %errorlevel%
:pylauncher
py -3 "%CAPTURE%" %*
exit /b %errorlevel%
:pythonpath
python "%CAPTURE%" %*
exit /b %errorlevel%
