param(
    [switch]$Diagnostic,
    [string]$Python = "$PSScriptRoot\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Build this executable on Windows.' }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$variant = if ($Diagnostic) { 'diagnostic' } else { 'release' }
$previous = $env:SNAKE_BUILD_DIAGNOSTIC
try {
    $env:SNAKE_BUILD_DIAGNOSTIC = if ($Diagnostic) { '1' } else { '0' }
    & $Python -c "import pygame, av, PyInstaller; print('Build dependencies ready')"
    if ($LASTEXITCODE -ne 0) { throw 'Install requirements-build.txt in the build environment first.' }
    & $Python -m PyInstaller --noconfirm --distpath "$PSScriptRoot\dist\$variant" --workpath "$PSScriptRoot\build\$variant" "$PSScriptRoot\SnakeInfinite.spec"
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    Write-Output "Built: $PSScriptRoot\dist\$variant\SnakeInfinite\SnakeInfinite.exe"
} finally {
    $env:SNAKE_BUILD_DIAGNOSTIC = $previous
}
