$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

$pythonExe = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        & py -3.11 --version *> $null
        $pythonExe = "py -3.11"
    } catch {
        $pythonExe = "python"
    }
}

if ($pythonExe -eq "py -3.11") {
    & py -3.11 -m venv .venv
} else {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Setup complete."
Write-Host "Run: .\\.venv\\Scripts\\Activate.ps1 ; python src\\run_camera.py"
