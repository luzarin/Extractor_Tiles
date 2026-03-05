param(
    [int]$Port = 8090
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $Root
try {
    uvicorn api.workflow_ui_api:app --app-dir $Root --host 127.0.0.1 --port $Port --reload
}
finally {
    Pop-Location
}
