$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\PC\anaconda3\envs\paper\python.exe"
$logDir = Join-Path $repoRoot "smacv2_poc\results"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$jobs = @(
    @{
        Seed = 3
        OutLog = Join-Path $logDir "log_cond5_seed3.out.txt"
        ErrLog = Join-Path $logDir "log_cond5_seed3.err.txt"
    },
    @{
        Seed = 4
        OutLog = Join-Path $logDir "log_cond5_seed4.out.txt"
        ErrLog = Join-Path $logDir "log_cond5_seed4.err.txt"
    }
)

foreach ($job in $jobs) {
    $args = @(
        "-u"
        (Join-Path $repoRoot "train.py")
        "--condition"
        "5"
        "--seed"
        [string]$job.Seed
    )

    Start-Process -FilePath $python `
        -ArgumentList $args `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $job.OutLog `
        -RedirectStandardError $job.ErrLog
}

Write-Host "Started condition 5 runs for seeds 3 and 4."
Write-Host "Logs:"
foreach ($job in $jobs) {
    Write-Host "  stdout: $($job.OutLog)"
    Write-Host "  stderr: $($job.ErrLog)"
}
