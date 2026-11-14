# 모든 수집기를 병렬로 실행 (서로 다른 서버라 동시에 돌려도 한도 충돌 없음).
# 백필과 매일 갱신 모두 이 스크립트 하나로 처리한다. 이미 받은 구간은 건너뛴다.
#   사용: powershell -ExecutionPolicy Bypass -File scripts\collect_all.ps1
#   일부만: ... -File scripts\collect_all.ps1 -Sources binance,kr

param([string[]]$Sources = @("binance", "upbit", "kr", "us", "macro", "kr-public"))

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

foreach ($s in $Sources) {
    # 같은 수집기가 이미 돌고 있으면 건너뜀 (중복 실행 방지)
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "autotrader.data.cli collect $s(\s|$)" }
    if ($running) { Write-Output "$s 이미 실행 중 - 건너뜀"; continue }
    Start-Process -FilePath "uv" -ArgumentList "run", "python", "-m", "autotrader.data.cli", "collect", $s `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$root\logs\run-$s-$stamp.out" -RedirectStandardError "$root\logs\run-$s-$stamp.err"
    Write-Output "$s 시작"
}
