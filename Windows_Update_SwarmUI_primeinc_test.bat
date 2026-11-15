@echo off
setlocal EnableExtensions

rem Enable opt-in H3 sampler diagnostics for SwarmUI and its ComfyUI child process.
rem Only this updater opts in; ordinary SwarmUI launches remain unchanged.
set "SWARM_H3_TRACE=1"

rem This updater may be downloaded next to SwarmUI or run from the repo root.
rem Normalize the working directory to the parent of the SwarmUI checkout.
if exist "%~dp0SwarmUI\.git" (
    pushd "%~dp0" || exit /b 1
) else if exist "%~dp0.git" (
    pushd "%~dp0.." || exit /b 1
) else (
    echo ERROR: Put this updater next to the SwarmUI folder or in its repository root.
    exit /b 1
)

rem Fast Windows/CI smoke test: no network access, updates, builds, or launch.
if /I "%~1"=="--verify-h3-trace" (
    cmd /d /c set SWARM_H3_TRACE | findstr /X /C:"SWARM_H3_TRACE=1" >nul
    if errorlevel 1 (
        echo ERROR: Trace variable did not reach the child process.
        popd
        exit /b 1
    )
    echo H3_TRACE_CHILD_INHERITANCE_OK
    popd
    exit /b 0
)

rem Remove any duplicate nested src directory that may have been created by failed builds
if exist "SwarmUI\src\src" (
    echo Removing duplicate src directory...
    rmdir /S /Q "SwarmUI\src\src"
)

if not exist "SwarmUI\src\BuiltinExtensions\ComfyUIBackend\DLNodes\ComfyUI-Frame-Interpolation" (
    git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-Frame-Interpolation
)

if not exist "SwarmUI\src\BuiltinExtensions\ComfyUIBackend\DLNodes\ComfyUI-TeaCache" (
    git clone --depth 1 https://github.com/FurkanGozukara/ComfyUI-TeaCache SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-TeaCache
)

if not exist "SwarmUI\src\BuiltinExtensions\ComfyUIBackend\DLNodes\comfyui_controlnet_aux" (
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/comfyui_controlnet_aux
)

cd /d SwarmUI || exit /b 1

git config --get-all remote.origin.fetch | findstr /X /L /C:"+refs/heads/dev:refs/remotes/origin/dev" >nul
if errorlevel 1 git config --add remote.origin.fetch "+refs/heads/dev:refs/remotes/origin/dev"

git fetch origin --tags || exit /b 1

git show-ref --verify --quiet refs/heads/dev
if errorlevel 1 (
    git switch -c dev --track origin/dev || exit /b 1
) else (
    git switch dev || exit /b 1
)

git branch --set-upstream-to=origin/dev dev || exit /b 1
git pull --ff-only || exit /b 1

git status -sb
git log -1 --oneline

cd src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-Frame-Interpolation
git reset --hard

git pull

cd .. 
cd ComfyUI-TeaCache

git remote set-url origin https://github.com/FurkanGozukara/ComfyUI-TeaCache

git reset --hard

git pull

cd .. 
cd comfyui_controlnet_aux

git reset --hard

git pull

cd ..
cd ..
cd ..
cd ..
cd ..
cd ..

if exist "install_premium_extensions.py" (
    python install_premium_extensions.py
) else (
    echo Premium extensions installer not present in parent folder, skipping.
)

echo SwarmUI dev checkout updated.

cd SwarmUI || exit /b 1

rem Force rebuild so extensions compile
echo. 2>src\bin\must_rebuild

rem Child processes inherit SWARM_H3_TRACE=1 from this CMD session.
rem [H3PERF] logs identify samplers by ComfyUI prompt and node IDs.
echo H3 sampler trace enabled: SWARM_H3_TRACE=%SWARM_H3_TRACE%
echo Find [H3PERF] records in the SwarmUI ComfyUI backend logs.
echo Compare three sequential traced generations using:
echo   python tests\h3_perf_compare.py h3-session.log --three-in-one-log
call launch-windows.bat --port 7861
set "SWARM_LAUNCH_EXIT=%ERRORLEVEL%"
pause
popd
exit /b %SWARM_LAUNCH_EXIT%
