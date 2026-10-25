# Low-label experiment, run after scripts/run_queue.ps1 (waits for it): only 3 training speakers have transcripts.
#   LL_noKD : CTC on the 3 labeled speakers only (degraded crops), from scratch
#   LL_m2m3 : same CTC + teacher KD on all 28 training speakers (the other 25 contribute KD only)
# The teacher targets need audio but no transcripts, so KD can exploit unlabeled video.
# Then the GRID-Crowd evaluation for every finished run.
$ScratchMin = 60
$Workers    = 8
$Labeled    = @("s3", "s4", "s5")
if (-not $env:L2L_WORK) { $env:L2L_WORK = "E:\College\IE_Project\work" }
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location (Join-Path $PSScriptRoot "..")
$Runs = Join-Path $env:L2L_WORK "runs"
$Common = @("--backend", "conformer", "--bs", 16, "--accum", 1, "--workers", $Workers)

while (-not (Select-String -Path (Join-Path $Runs "queue.out") -Pattern "queue finished" -Quiet)) { Start-Sleep 60 }

function Train-Eval([string]$Name, [string[]]$TrainArgs) {
    Write-Host "=== $(Get-Date -Format HH:mm) $Name : $TrainArgs"
    python distill/train.py --name $Name @Common @TrainArgs 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
    $ckpt = Join-Path $Runs "$Name\best.pt"
    if ((Test-Path $ckpt) -and -not (Test-Path (Join-Path $Runs "$Name\eval_summary.json"))) {
        python eval/evaluate.py --ckpt $ckpt --workers $Workers 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
    }
}

$LL = @("--degrade", "--max_minutes", $ScratchMin, "--val_every", 1000)
Train-Eval "LL_noKD" ($LL + @("--kd", "none", "--train_spk") + $Labeled)
Train-Eval "LL_m2m3" ($LL + @("--kd", "m2m3", "--labeled_spk") + $Labeled)

$ckpts = Get-ChildItem $Runs -Directory | Where-Object { (Test-Path "$($_.FullName)\best.pt") -and -not (Test-Path "$($_.FullName)\eval_crowd.json") -and $_.Name -ne "smoke_conformer" } | ForEach-Object { "$($_.FullName)\best.pt" }
if ($ckpts) { python -m eval.eval_crowd --ckpt @ckpts --workers $Workers 2>&1 | Tee-Object -FilePath (Join-Path $Runs "crowd.out") -Append }
Write-Host "=== lowlabel queue finished $(Get-Date -Format HH:mm)"
