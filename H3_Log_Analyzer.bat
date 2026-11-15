@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" 2>nul || (
    echo ERROR: Cannot open the directory containing this script.
    exit /b 1
)
title SWARM - H3 Performance Studio
color 0B

set "ANALYZER=%CD%\tests\h3_log_console.py"
set "REPORT_DIR=%CD%\H3-Reports"
set "PYRUN="
python -c "import sys; sys.exit(0 if sys.version_info.major == 3 else 1)" >nul 2>&1
if not errorlevel 1 set "PYRUN=python"
if not defined PYRUN (
    py -3 -c "import sys; sys.exit(0 if sys.version_info.major == 3 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYRUN=py -3"
)
if not defined PYRUN (
    echo.
    echo ERROR: Python 3 is required. Install Python or make it available on PATH.
    echo This is the same Python runtime used by the existing H3 comparison utility.
    popd
    exit /b 1
)
if not exist "%ANALYZER%" (
    echo.
    echo ERROR: tests\h3_log_console.py is missing.
    echo Update primeinc/SwarmUI dev before running the analyzer.
    popd
    exit /b 1
)

if /I "%~1"=="--self-test" goto SELFTEST
if /I "%~1"=="--scan" goto ARG_SCAN
if /I "%~1"=="--extract" goto ARG_EXTRACT
if not "%~1"=="" goto ARG_DROPPED

:MENU
cls
echo.
echo    +========================================================================+
echo    ^|                                                                        ^|
echo    ^|                S W A R M  /  H 3   P E R F   S T U D I O             ^|
echo    ^|                                                                        ^|
echo    +========================================================================+
echo    ^|  DIAGNOSTICS                                                           ^|
echo    ^|                                                                        ^|
echo    ^|   [1]  QUICK SCAN              Legacy logs + detailed sampler traces     ^|
echo    ^|   [2]  TWO RUNS / ONE LOG      I2V first, main video second             ^|
echo    ^|   [3]  THREE RUNS / ONE LOG    I2V, main, one-frame follow-up          ^|
echo    ^|                                                                        ^|
echo    ^|  COMPARISONS                                                           ^|
echo    ^|                                                                        ^|
echo    ^|   [4]  COMPARE TWO FILES      Two-stage I2V vs main video               ^|
echo    ^|   [5]  COMPARE THREE FILES    Include the one-frame follow-up         ^|
echo    ^|                                                                        ^|
echo    ^|  UTILITIES                                                             ^|
echo    ^|                                                                        ^|
echo    ^|   [6]  EXPORT H3 TRACE        Save timestamped JSONL events           ^|
echo    ^|   [7]  OPEN REPORTS           Browse the H3-Reports folder             ^|
echo    ^|   [8]  CAPTURE GUIDE          How to enable and record tracing        ^|
echo    ^|   [9]  EXIT                                                          ^|
echo    ^|                                                                        ^|
echo    +------------------------------------------------------------------------+
echo.
set "MENU_CHOICE="
set /p "MENU_CHOICE=    CHOOSE  [1-9] : "
if "%MENU_CHOICE%"=="1" goto PICK_SCAN
if "%MENU_CHOICE%"=="2" goto PICK_TWO_ONE
if "%MENU_CHOICE%"=="3" goto PICK_THREE_ONE
if "%MENU_CHOICE%"=="4" goto PICK_TWO_FILES
if "%MENU_CHOICE%"=="5" goto PICK_THREE_FILES
if "%MENU_CHOICE%"=="6" goto PICK_EXTRACT
if "%MENU_CHOICE%"=="7" goto OPEN_REPORTS
if "%MENU_CHOICE%"=="8" goto HELP
if "%MENU_CHOICE%"=="9" goto END
if /I "%MENU_CHOICE%"=="Q" goto END
echo.
echo   That option does not exist.
call :WAIT
goto MENU

:PICK_SCAN
set "MODE=scan"
call :PICKLOG LOG1 "Select the SwarmUI log to scan"
if errorlevel 1 goto MENU
goto RUN

:PICK_TWO_ONE
set "MODE=two-in-one"
call :PICKLOG LOG1 "Select log containing I2V, then main video"
if errorlevel 1 goto MENU
goto RUN

:PICK_THREE_ONE
set "MODE=three-in-one"
call :PICKLOG LOG1 "Select log containing I2V, main, then one-frame follow-up"
if errorlevel 1 goto MENU
goto RUN

:PICK_TWO_FILES
set "MODE=two-files"
call :PICKLOG LOG1 "First: two-video I2V run"
if errorlevel 1 goto MENU
call :PICKLOG LOG2 "Second: one-video main H3 run"
if errorlevel 1 goto MENU
goto RUN

:PICK_THREE_FILES
set "MODE=three-files"
call :PICKLOG LOG1 "First: two-video I2V run"
if errorlevel 1 goto MENU
call :PICKLOG LOG2 "Second: one-video main H3 run"
if errorlevel 1 goto MENU
call :PICKLOG LOG3 "Third: one-frame first stage plus I2V follow-up"
if errorlevel 1 goto MENU
goto RUN

:PICK_EXTRACT
set "MODE=extract"
call :PICKLOG LOG1 "Select log with [H3PERF] lines to export"
if errorlevel 1 goto MENU
goto RUN

:PICKLOG
set "PICKED="
echo.
echo    %~2
rem Use the native Windows file picker. If unavailable/cancelled, accept a path.
for /f "usebackq delims=" %%F in (`powershell.exe -NoLogo -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Title='Swarm H3 Performance Studio'; $d.Filter='Logs (*.log;*.txt)|*.log;*.txt|All files (*.*)|*.*'; $d.InitialDirectory=[Environment]::CurrentDirectory; if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {[Console]::WriteLine($d.FileName)}" 2^>nul`) do set "PICKED=%%F"
if not defined PICKED (
    echo    File picker closed. Enter a log path, or leave blank to cancel.
    set /p "PICKED=    FILE : "
)
if not defined PICKED exit /b 1
rem Windows Explorer drag-and-drop may wrap paths in quotes.
set "PICKED=%PICKED:"=%"
if not exist "%PICKED%" (
    echo.
    echo    File not found: "%PICKED%"
    call :WAIT
    exit /b 1
)
set "%~1=%PICKED%"
exit /b 0

:RUN
cls
echo.
echo   +------------------------------------------------------------------------+
echo   ^|  RUNNING H3 ANALYSIS                                                   ^|
echo   +------------------------------------------------------------------------+
echo   MODE: %MODE%
echo.
if /I "%MODE%"=="two-files" goto RUN_TWO_FILES
if /I "%MODE%"=="three-files" goto RUN_THREE_FILES
:RUN_ONE
%PYRUN% "%ANALYZER%" --mode "%MODE%" --log "%LOG1%" --outdir "%REPORT_DIR%"
goto RUN_DONE
:RUN_TWO_FILES
%PYRUN% "%ANALYZER%" --mode "%MODE%" --log "%LOG1%" --log "%LOG2%" --outdir "%REPORT_DIR%"
goto RUN_DONE
:RUN_THREE_FILES
%PYRUN% "%ANALYZER%" --mode "%MODE%" --log "%LOG1%" --log "%LOG2%" --log "%LOG3%" --outdir "%REPORT_DIR%"
:RUN_DONE
set "RUN_RC=%ERRORLEVEL%"
if not "%RUN_RC%"=="0" echo   ANALYSIS FAILED. No source log was modified.
if defined NONINTERACTIVE goto END_WITH_RC
echo.
call :WAIT
goto MENU

:OPEN_REPORTS
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"
start "" "%REPORT_DIR%"
goto MENU

:HELP
cls
echo.
echo   +========================================================================+
echo   ^|  H3 / CAPTURE GUIDE                                                    ^|
echo   +========================================================================+
echo.
echo   1. Update your primeinc/SwarmUI dev checkout.
echo   2. Close existing SwarmUI and ComfyUI processes.
echo   3. Launch with the dev Windows updater: it sets SWARM_H3_TRACE=1.
echo   4. Record all three comparison runs in the same warm backend session:
echo        A: 324-frame I2V follow-up, with preliminary H3 stage.
echo        B: 324-frame main H3 generation.
echo        C: 1-frame main H3, then 324-frame I2V follow-up.
echo   5. Save the full SwarmUI log to .log or .txt.
echo.
echo   For a persistent console log, launch from PowerShell:
echo     $env:SWARM_H3_TRACE = '1'
echo     .\launch-windows.bat --port 7861 *^>^&1 ^| Tee-Object -FilePath h3-session.log
echo.
echo   NOTE: Old logs without [H3PERF] can be Quick Scanned, but cannot
echo   prove actual sampler conditioning or model-weight residency.
echo.
echo   Reports include both human-readable TXT and machine-readable JSON.
echo   Logs are never altered. H3-Reports is gitignored.
echo.
call :WAIT
goto MENU

:WAIT
echo    Press any key to continue...
pause >nul
exit /b 0

:ARG_SCAN
set "NONINTERACTIVE=1"
set "MODE=scan"
set "LOG1=%~2"
if not defined LOG1 goto BAD_ARGS
goto RUN
:ARG_EXTRACT
set "NONINTERACTIVE=1"
set "MODE=extract"
set "LOG1=%~2"
if not defined LOG1 goto BAD_ARGS
goto RUN
:ARG_DROPPED
set "NONINTERACTIVE=1"
set "MODE=scan"
set "LOG1=%~1"
goto RUN

:SELFTEST
%PYRUN% -m unittest discover -s "%CD%\tests" -p "test_h3_log_console.py" -v
if errorlevel 1 (
    popd
    exit /b 1
)
echo H3_LOG_MENU_SELFTEST_OK
popd
exit /b 0

:BAD_ARGS
echo   ERROR: Missing input log path.
popd
exit /b 2
:END_WITH_RC
popd
exit /b %RUN_RC%
:END
popd
exit /b 0
