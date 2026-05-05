<#
Launch scaling-study training runs from PowerShell.

Default: 3v3_balanced x conditions {1,3} x seeds {0,1,2}.
Each condition launches all requested seeds in parallel, waits for them,
then moves to the next condition.

Examples:
  .\run_scaling_batch.ps1
  .\run_scaling_batch.ps1 -Conditions "1"
  .\run_scaling_batch.ps1 -Batch 6v6_balanced -Conditions "1 3" -Seeds "0 1 2"
  .\run_scaling_batch.ps1 -Python C:\ProgramData\anaconda3\envs\paper\python.exe
#>

param(
    [string]$Batch = "3v3_balanced",
    [string]$Conditions = "1 3",
    [string]$Seeds = "0 1 2",
    [string]$Python = "C:\ProgramData\anaconda3\envs\paper\python.exe",
    [switch]$SequentialSeeds
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsDir = Join-Path $RepoRoot "results_scaling"
$LogDir = Join-Path $ResultsDir "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$ConditionValues = @(
    $Conditions -split "[,\s]+" |
        Where-Object { $_ -ne "" } |
        ForEach-Object { [int]$_ }
)
$SeedValues = @(
    $Seeds -split "[,\s]+" |
        Where-Object { $_ -ne "" } |
        ForEach-Object { [int]$_ }
)

function Quote-CmdArg {
    param([string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Start-ScalingRun {
    param(
        [string]$RepoRoot,
        [string]$Python,
        [string]$Batch,
        [int]$Condition,
        [int]$Seed,
        [string]$LogDir
    )

    $tag = "${Batch}_cond${Condition}_seed${Seed}"
    $stdout = Join-Path $LogDir "${tag}.stdout.log"
    $stderr = Join-Path $LogDir "${tag}.stderr.log"

    $trainScript = Join-Path $RepoRoot "training\train_scaling.py"
    $command = @(
        Quote-CmdArg $Python
        "-u"
        Quote-CmdArg $trainScript
        "--config_name"
        Quote-CmdArg $Batch
        "--condition"
        $Condition
        "--seed"
        $Seed
        ">"
        Quote-CmdArg $stdout
        "2>"
        Quote-CmdArg $stderr
    ) -join " "

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = "C:\Windows\System32\cmd.exe"
    $psi.Arguments = '/d /s /c "' + $command + '"'
    $psi.WorkingDirectory = $RepoRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false

    $proc = [System.Diagnostics.Process]::Start($psi)
    return [PSCustomObject]@{
        Tag = $tag
        Condition = $Condition
        Seed = $Seed
        Pid = $proc.Id
        Stdout = $stdout
        Stderr = $stderr
        Process = $proc
    }
}

$conditionList = $ConditionValues -join ","
$seedList = $SeedValues -join ","
$mode = if ($SequentialSeeds) { "sequential" } else { "parallel" }

Write-Host "[batch=$Batch] conditions=[$conditionList] seeds=[$seedList] mode=$mode"
Write-Host "[results] $ResultsDir"

$failedRuns = @()

foreach ($condition in $ConditionValues) {
    Write-Host "== Launching config=$Batch cond=$condition seeds=[$seedList] =="
    $runs = @()

    foreach ($seed in $SeedValues) {
        $run = Start-ScalingRun `
            -RepoRoot $RepoRoot `
            -Python $Python `
            -Batch $Batch `
            -Condition $condition `
            -Seed $seed `
            -LogDir $LogDir

        $runs += $run
        Write-Host ("  -> {0} pid={1}" -f $run.Tag, $run.Pid)
        Write-Host ("     stdout: {0}" -f $run.Stdout)
        Write-Host ("     stderr: {0}" -f $run.Stderr)

        if ($SequentialSeeds) {
            $run.Process.WaitForExit()
            Write-Host ("  <- {0} exit={1}" -f $run.Tag, $run.Process.ExitCode)
            if ($run.Process.ExitCode -ne 0) {
                $failedRuns += $run
            }
        }
    }

    if (-not $SequentialSeeds) {
        Write-Host "  waiting on $($runs.Count) seed runs for cond=$condition..."
        foreach ($run in $runs) {
            $run.Process.WaitForExit()
            Write-Host ("  <- {0} exit={1}" -f $run.Tag, $run.Process.ExitCode)
            if ($run.Process.ExitCode -ne 0) {
                $failedRuns += $run
            }
        }
    }

    Write-Host "== Finished config=$Batch cond=$condition =="
}

if ($failedRuns.Count -gt 0) {
    Write-Host "[batch=$Batch] failed runs:"
    foreach ($run in $failedRuns) {
        Write-Host ("  - {0}: stderr={1}" -f $run.Tag, $run.Stderr)
    }
    exit 1
}

Write-Host "[batch=$Batch] all runs complete."
