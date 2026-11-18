# KIS를 쓰는 수집기를 하나씩 차례로 실행한다 (collect_all.ps1이 백그라운드로 부른다).
# 같은 KIS 키를 여러 프로세스가 동시에 쓰면 프로세스마다 따로 속도 제한(15회/초)을 걸어 합쳐서 한도(20회/초)를 넘는다.
#   사용: powershell -ExecutionPolicy Bypass -File scripts\collect_kis_serial.ps1 -Sources kr,dividends -Stamp 20261118-1700

param([string[]]$Sources = @("kr", "dividends"), [string]$Stamp = (Get-Date -Format "yyyyMMdd-HHmm"))

# -File로 부르면 "kr,dividends"가 한 문자열로 들어온다 → 쉼표로 나눈다
$Sources = @($Sources | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$kisPattern = "autotrader.data.cli collect (kr|kr-daily|kr-minute|dividends)(\s|$)"
$maxWaitMin = 360  # 앞 회차의 KIS 수집이 아직 돌고 있으면 최대 6시간 기다린다

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

function Get-KisRunning {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $kisPattern }
}

foreach ($s in $Sources) {
    # 같은 수집기가 이미 돌고 있으면 건너뜀 (중복 실행 방지)
    $same = Get-KisRunning | Where-Object { $_.CommandLine -match "autotrader.data.cli collect $s(\s|$)" }
    if ($same) { Write-Output "$(Get-Date -Format s) $s 이미 실행 중 - 건너뜀"; continue }
    # 다른 KIS 수집기(앞 회차가 남은 것 등)가 돌고 있으면 끝날 때까지 기다린다
    $waited = 0
    while ((Get-KisRunning) -and $waited -lt $maxWaitMin) { Start-Sleep -Seconds 60; $waited++ }
    if (Get-KisRunning) { Write-Output "$(Get-Date -Format s) $s 다른 KIS 수집이 $maxWaitMin분 넘게 실행 중 - 건너뜀"; continue }
    Write-Output "$(Get-Date -Format s) $s 시작"
    $p = Start-Process -FilePath "uv" -ArgumentList "run", "python", "-m", "autotrader.data.cli", "collect", $s `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput "$root\logs\run-$s-$Stamp.out" -RedirectStandardError "$root\logs\run-$s-$Stamp.err"
    Write-Output "$(Get-Date -Format s) $s 종료 (exit $($p.ExitCode))"
}
