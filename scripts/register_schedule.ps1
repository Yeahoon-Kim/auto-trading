# Windows 작업 스케줄러 등록 (다시 실행하면 덮어씀).
#
#   autotrader-collect : 07:30 (미국장 마감 후), 17:00 (국내장 마감 후), 로그인할 때 — 전체 수집
#   autotrader-brief   : 08:00 — 국면 지표 재계산 + 장 전 브리핑 텔레그램 전송
#   autotrader-health  : 12:00, 20:00 — 수집 상태 점검, 문제 있을 때만 텔레그램 알림
#   autotrader-ops     : 23:00 — 운영 마감 보고 (사실 표 + ops-reporter 요약) 텔레그램 전송
#   autotrader-weekly  : 토요일 10:00 (미국 금요일 장 마감 후) — 주간 회의 자료 (reports/weekly/) 텔레그램 전송
#
#   PC가 꺼져 있어 놓친 실행은 켜진 직후 실행된다 (StartWhenAvailable).
#   모든 소스가 과거분을 보관하고 있어서 며칠 꺼져 있어도 다음 실행 때 빈 구간을 채운다.
#   단, 미국 1분봉·코인 미결제약정은 최근 30일만 제공되므로 30일 넘게 꺼져 있으면 그 구간은 영구 손실.
#   해제: Get-ScheduledTask autotrader-* | Unregister-ScheduledTask -Confirm:$false

$root = Split-Path -Parent $PSScriptRoot
$ps = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 20)

function Register($name, $script, $argsLine, $triggers, $desc) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "$ps `"$root\scripts\$script`" $argsLine" -WorkingDirectory $root
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings `
        -Description $desc -Force | Out-Null
    Write-Output "등록: $name"
}

Register "autotrader-collect" "collect_all.ps1" "" @(
    (New-ScheduledTaskTrigger -Daily -At "07:30"),
    (New-ScheduledTaskTrigger -Daily -At "17:00"),
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME)
) "auto-trading 데이터 수집 (Binance, Upbit, KIS, Yahoo, FRED)"

Register "autotrader-brief" "run_module.ps1" "autotrader.intel.market --notify" @(
    (New-ScheduledTaskTrigger -Daily -At "08:00")
) "auto-trading 장 전 브리핑"

Register "autotrader-health" "run_module.ps1" "autotrader.data.cli health --notify" @(
    (New-ScheduledTaskTrigger -Daily -At "12:00"),
    (New-ScheduledTaskTrigger -Daily -At "20:00")
) "auto-trading 수집 상태 점검"

Register "autotrader-ops" "run_module.ps1" "autotrader.monitoring.daily_report --notify" @(
    (New-ScheduledTaskTrigger -Daily -At "23:00")
) "auto-trading 운영 마감 보고"

Register "autotrader-weekly" "run_module.ps1" "autotrader.monitoring.weekly_report --notify" @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "10:00")
) "auto-trading 주간 회의 자료"
