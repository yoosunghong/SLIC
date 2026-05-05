$ErrorActionPreference = "Continue"

$repo = "D:\Github\SLIC"
$python = "C:\ProgramData\anaconda3\envs\paper\python.exe"
$train = Join-Path $repo "smacv2_poc\training\train_scaling.py"
$resultsDir = Join-Path $repo "smacv2_poc\results_scaling"
$logDir = Join-Path $resultsDir "logs"
$workDir = Join-Path $repo "smacv2_poc"
$blockingPid = 15592
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Write-RunnerLog($message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message"
    Add-Content -LiteralPath (Join-Path $logDir "resume_cond1_seed0_2_after_seed1_$stamp.runner.log") -Value $line
}

function Get-LatestCheckpoint($seed) {
    $tag = "3v3_balanced_cond1_seed$seed"
    $ckptDir = Join-Path $resultsDir "checkpoints\$tag"
    $latest = Get-ChildItem -LiteralPath $ckptDir -Filter "step*.pt" |
        Sort-Object { [int64](($_.BaseName -replace '^step', '')) } |
        Select-Object -Last 1
    if ($null -eq $latest) {
        throw "No checkpoint found in $ckptDir"
    }
    return $latest.FullName
}

function Wait-ForProcessExit($targetPid) {
    while (Get-Process -Id $targetPid -ErrorAction SilentlyContinue) {
        Write-RunnerLog "Waiting for existing seed1 python pid=$targetPid to finish..."
        Start-Sleep -Seconds 60
    }
}

function Wait-ForSc2Exit() {
    $deadline = (Get-Date).AddMinutes(10)
    while ((Get-Process -Name "SC2_x64" -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) {
        Write-RunnerLog "Waiting for SC2_x64 to exit cleanly..."
        Start-Sleep -Seconds 30
    }
}

function Run-Seed($seed) {
    $tag = "3v3_balanced_cond1_seed$seed"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $ckpt = Get-LatestCheckpoint $seed
        $outLog = Join-Path $logDir "$tag.resume_after_seed1_$stamp.attempt$attempt.stdout.log"
        $errLog = Join-Path $logDir "$tag.resume_after_seed1_$stamp.attempt$attempt.stderr.log"
        Write-RunnerLog "Starting $tag attempt=$attempt from $ckpt"

        Push-Location $workDir
        & $python -u $train `
            --config_name 3v3_balanced `
            --condition 1 `
            --seed $seed `
            --results_dir results_scaling `
            --resume $ckpt `
            1> $outLog `
            2> $errLog
        $exitCode = $LASTEXITCODE
        Pop-Location

        Write-RunnerLog "$tag attempt=$attempt exited code=$exitCode"
        if ($exitCode -eq 0) {
            return
        }

        Wait-ForSc2Exit
        Start-Sleep -Seconds 120
    }
    Write-RunnerLog "$tag failed after 3 attempts"
}

Write-RunnerLog "Runner started. Blocking pid=$blockingPid"
Wait-ForProcessExit $blockingPid
Wait-ForSc2Exit
Run-Seed 0
Wait-ForSc2Exit
Run-Seed 2
Write-RunnerLog "Runner finished."
