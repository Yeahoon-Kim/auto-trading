# 작업 스케줄러용: 파이썬 모듈 하나를 실행하고 출력을 logs/ 에 남긴다.
#   예: powershell -File scripts\run_module.ps1 autotrader.intel.market --notify
param([Parameter(Mandatory = $true)][string]$Module, [Parameter(ValueFromRemainingArguments = $true)]$Rest)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"   # 이모지·한글 출력 시 cp949 인코딩 오류 방지
[Console]::OutputEncoding = [Text.Encoding]::UTF8   # 파이썬 출력(UTF-8)을 올바르게 읽기
$name = $Module.Split(".")[-1]
& uv run python -m $Module @Rest 2>&1 | Out-File -Append -Encoding utf8 "$root\logs\task-$name-$(Get-Date -Format yyyyMMdd).log"
