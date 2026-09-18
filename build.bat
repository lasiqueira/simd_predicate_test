@echo off
setlocal enabledelayedexpansion

rem Usage: build.bat [msvc|clang]
rem Default compiler is MSVC cl; pass "clang" to use clang-cl instead.
set "COMPILER=msvc"
if /i "%~1"=="msvc" set "COMPILER=msvc"
if /i "%~1"=="clang" set "COMPILER=clang"

rem Locate and load the MSVC developer environment (needed for both cl and clang-cl on Windows).
rem Note: %ProgramFiles(x86)% is set outside any ( ) block, since its parentheses
rem confuse cmd's block parser if expanded with %...% from inside a block.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not defined VCINSTALLDIR (
    if not exist "!VSWHERE!" (
        echo Could not find vswhere.exe, cannot locate Visual Studio installation.
        exit /b 1
    )

    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSINSTALLPATH=%%i"
    )

    if not defined VSINSTALLPATH (
        echo Could not find a Visual Studio installation with the VC++ toolset.
        exit /b 1
    )

    call "!VSINSTALLPATH!\VC\Auxiliary\Build\vcvarsall.bat" x64
    if errorlevel 1 exit /b 1
)

set "SRC=src\main.cpp src\platform_metrics.cpp src\repetition_tester.cpp"
set "BUILDDIR=build"
set "OUT=%BUILDDIR%\simd_predicate_test.exe"

rem no incremental build (every invocation recompiles all sources from scratch), so nothing to clean
if not exist "%BUILDDIR%" mkdir "%BUILDDIR%"

if /i "%COMPILER%"=="clang" (
    echo Building with clang-cl...
    clang-cl /EHsc /O2 /arch:AVX512 /std:c++17 %SRC% /Fo"%BUILDDIR%\\" /Fe:"%OUT%"
) else (
    echo Building with MSVC cl...
    cl /EHsc /O2 /arch:AVX512 /std:c++17 %SRC% /Fo"%BUILDDIR%\\" /Fe:"%OUT%"
)

if errorlevel 1 exit /b 1

echo Build succeeded: %OUT%
