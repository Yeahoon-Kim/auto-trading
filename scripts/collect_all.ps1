# 수집기를 병렬로 실행 (서로 다른 서버라 동시에 돌려도 한도 충돌 없음).
# 단, KIS를 쓰는 수집기(kr·kr-daily·kr-minute·dividends)는 같은 키를 쓰므로 collect_kis_serial.ps1이
# 백그라운드에서 -Sources에 적힌 순서대로 하나씩 돌린다 (그 묶음 전체는 다른 소스와 병렬).
# 백필과 매일 갱신 모두 이 스크립트 하나로 처리한다. 이미 받은 구간은 건너뛴다.
#   사용: powershell -ExecutionPolicy Bypass -File scripts\collect_all.ps1
#   일부만: ... -File scripts\collect_all.ps1 -Sources binance,kr

param([string[]]$Sources = @("binance", "upbit", "kr", "us", "us-sip", "news", "macro", "kr-public", "dividends", "edgar", "us-splits", "naver-trend", "naver-news"))

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

# -File로 부르면 "binance,kr"가 한 문자열로 들어온다 → 쉼표로 나눈다
$Sources = @($Sources | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$kisSources = @("kr", "kr-daily", "kr-minute", "dividends")
$kis = @($Sources | Where-Object { $kisSources -contains $_ })
if ($kis.Count -gt 0) {
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", `
        "`"$PSScriptRoot\collect_kis_serial.ps1`"", "-Sources", ($kis -join ","), "-Stamp", $stamp `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$root\logs\run-kis-serial-$stamp.out" -RedirectStandardError "$root\logs\run-kis-serial-$stamp.err"
    Write-Output "KIS 순차 실행 시작: $($kis -join ' → ')"
}

foreach ($s in ($Sources | Where-Object { $kisSources -notcontains $_ })) {
    # 같은 수집기가 이미 돌고 있으면 건너뜀 (중복 실행 방지)
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "autotrader.data.cli collect $s(\s|$)" }
    if ($running) { Write-Output "$s 이미 실행 중 - 건너뜀"; continue }
    Start-Process -FilePath "uv" -ArgumentList "run", "python", "-m", "autotrader.data.cli", "collect", $s `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$root\logs\run-$s-$stamp.out" -RedirectStandardError "$root\logs\run-$s-$stamp.err"
    Write-Output "$s 시작"
}
