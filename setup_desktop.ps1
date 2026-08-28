# Rebuild this project's environment from scratch on another machine.
#
#   powershell -ExecutionPolicy Bypass -File setup_desktop.ps1
#
# Safe to re-run. Verifies the torch transcription against magic.py at the end;
# if that check fails, stop -- nothing downstream is trustworthy.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== python ===" -ForegroundColor Cyan
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "python not found on PATH. Install Python 3.12 or newer." }
python -c "import sys; print(sys.executable); print(sys.version)"
$ver = python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "detected Python $ver"

Write-Host "`n=== venv ===" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv } else { Write-Host ".venv exists" }
$vpy = ".\.venv\Scripts\python.exe"

Write-Host "`n=== dependencies ===" -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nExact pins failed on this Python. Retrying unpinned --" -ForegroundColor Yellow
    Write-Host "results may differ from the recorded baseline. See DECISIONS.md." -ForegroundColor Yellow
    & $vpy -m pip install numpy scipy pandas pyarrow matplotlib torch
}

Write-Host "`n=== installed ===" -ForegroundColor Cyan
& $vpy -c "import numpy,scipy,pandas,pyarrow,torch;print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('pandas',pandas.__version__);print('pyarrow',pyarrow.__version__);print('torch',torch.__version__)"

Write-Host "`n=== data present? ===" -ForegroundColor Cyan
$n = (Get-ChildItem -Path "data" -Recurse -Filter *.mat -ErrorAction SilentlyContinue).Count
Write-Host "$n .mat files found (expect 31)"
if ($n -lt 31) { Write-Host "WARNING: raw data missing. git pull / copy data\ before running." -ForegroundColor Yellow }

Write-Host "`n=== GATE: torch transcription vs magic.py ===" -ForegroundColor Cyan
& $vpy mf_torch.py
if ($LASTEXITCODE -ne 0) { throw "mf_torch does not match magic.py. STOP." }

Write-Host "`nSetup OK. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -u sweep.py --quick --workers 4    # ~2 min smoke test"
Write-Host "  .\.venv\Scripts\python.exe -u sweep.py --workers 8            # the real run"
