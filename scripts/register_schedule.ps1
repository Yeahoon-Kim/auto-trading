# Windows 작업 스케줄러에 매일 수집 작업 등록.
#   07:30 (미국장 마감 후), 17:00 (국내장 마감 후), 로그인할 때. 코인은 매번 갱신.
#   PC가 꺼져 있어 놓친 실행은 켜진 직후 실행된다 (StartWhenAvailable).
#   모든 소스가 과거분을 보관하고 있어서 며칠 꺼져 있어도 다음 실행 때 빈 구간을 채운다.
#   단, 미국 1분봉은 Yahoo가 최근 30일만 주므로 30일 넘게 꺼져 있으면 그 구간은 영구 손실.
#   해제: Unregister-ScheduledTask -TaskName "autotrader-collect" -Confirm:$false

$root = Split-Path -Parent $PSScriptRoot
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\scripts\collect_all.ps1`"" `
    -WorkingDirectory $root
$triggers = @(
    (New-ScheduledTaskTrigger -Daily -At "07:30"),
    (New-ScheduledTaskTrigger -Daily -At "17:00"),
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME)  # 재부팅·재로그인 후 바로 이어받기
)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 20)
Register-ScheduledTask -TaskName "autotrader-collect" -Action $action -Trigger $triggers -Settings $settings `
    -Description "auto-trading 데이터 수집 (Binance, Upbit, KIS, Yahoo)" -Force
