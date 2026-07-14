@echo off
REM ============================================================================
REM  build.bat - compile the overlay (overlay.cpp + Dear ImGui) with MSVC.
REM
REM  Usage:
REM     build.bat            Release build  -> overlay.exe
REM     build.bat debug      Debug build (symbols, no optimisation)
REM     build.bat clean      Remove build artifacts
REM
REM  Requires Visual Studio 2019/2022 (any edition incl. Build Tools) with the
REM  "Desktop development with C++" workload and the Windows 10/11 SDK.
REM ============================================================================
setlocal enableextensions

cd /d "%~dp0"

if /i "%~1"=="clean" (
    echo Cleaning build artifacts...
    del /q overlay.exe overlay.obj *.obj 2>nul
    echo Done.
    goto :eof
)

set BUILD=release
if /i "%~1"=="debug" set BUILD=debug

REM ---------------------------------------------------------------------------
REM 1. Locate and initialise the MSVC toolchain (skip if already in a VS shell).
REM ---------------------------------------------------------------------------
if defined VCINSTALLDIR goto :have_env

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

set "VSDIR="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
)

if not defined VSDIR (
    echo [ERROR] Could not locate a Visual Studio C++ toolchain via vswhere.
    echo         Install "Desktop development with C++", or run this from a
    echo         "x64 Native Tools Command Prompt for VS".
    exit /b 1
)

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [ERROR] Failed to initialise the MSVC x64 environment.
    exit /b 1
)

:have_env

where cl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cl.exe not found on PATH after environment setup.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 2. Compiler flags.
REM ---------------------------------------------------------------------------
REM  /I imgui       - so #include "imgui/..." and imgui-internal includes resolve
REM  /EHsc          - standard C++ exception model
REM  /std:c++17     - matches std::mutex / algorithm usage in overlay.cpp
REM  /DUNICODE ...  - Win32 wide-char APIs (CreateWindowW, DefWindowProcW, etc.)
set "COMMON=/nologo /EHsc /std:c++17 /I imgui /DUNICODE /D_UNICODE /DWIN32_LEAN_AND_MEAN"

if /i "%BUILD%"=="debug" (
    echo Building DEBUG...
    set "CFLAGS=%COMMON% /Zi /Od /MDd /Fe:overlay.exe /Fd:overlay.pdb"
    set "LFLAGS=/link /SUBSYSTEM:WINDOWS /DEBUG"
) else (
    echo Building RELEASE...
    set "CFLAGS=%COMMON% /O2 /MD /Fe:overlay.exe"
    set "LFLAGS=/link /SUBSYSTEM:WINDOWS"
)

REM ---------------------------------------------------------------------------
REM 3. Translation units. overlay.cpp pulls in d3d11/dxgi/dwmapi via #pragma
REM    comment(lib, ...) already, so no explicit libs are needed on the link
REM    line. WinMain lives in overlay.cpp -> /SUBSYSTEM:WINDOWS.
REM ---------------------------------------------------------------------------
set "SOURCES=overlay.cpp imgui\imgui.cpp imgui\imgui_draw.cpp imgui\imgui_tables.cpp imgui\imgui_widgets.cpp imgui\imgui_impl_dx11.cpp imgui\imgui_impl_win32.cpp"

for %%f in (overlay.cpp imgui\imgui.cpp imgui\imgui_impl_dx11.cpp imgui\imgui_impl_win32.cpp) do (
    if not exist "%%f" (
        echo [ERROR] Missing source file: %%f
        exit /b 1
    )
)

cl %CFLAGS% %SOURCES% %LFLAGS%
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

REM Tidy intermediate object files, keep the exe (and pdb in debug).
del /q *.obj 2>nul

echo.
echo [OK] Build succeeded -^> overlay.exe
endlocal
