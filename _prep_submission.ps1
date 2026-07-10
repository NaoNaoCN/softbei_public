# 临时提交准备脚本：把 DATABASE_URL 指向 softbei_submission
# 用法: 传入子命令 alembic | seed | index-check
param([string]$step = "alembic")

$line = (Get-Content .env | Where-Object { $_ -match '^DATABASE_URL=' } | Select-Object -First 1)
$url = ($line -replace '^DATABASE_URL=', '').Trim()
$newUrl = $url -replace '/softbei$', '/softbei_submission'
if ($newUrl -eq $url) { Write-Host 'ERROR: db name replace failed (url did not end with /softbei)'; exit 1 }
$env:DATABASE_URL = $newUrl
Write-Host ('target db = ' + ($newUrl -replace '://[^@]*@', '://***@'))

$py = "D:/Anaconda3/envs/softbei/python.exe"

switch ($step) {
    "alembic" { & $py -m alembic upgrade head }
    "seed"    { & $py seed_demo.py }
    default   { Write-Host "unknown step: $step"; exit 1 }
}
