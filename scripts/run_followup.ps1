# Follow-up experiments, run after scripts/run_queue.ps1 (waits for it). Budgets are optimizer steps, not minutes,
# so the comparisons do not depend on machine load: fine-tunes match R2 (5416 steps), scratch runs match R1 (12763).
#   R9_ft_codec       : R2 + H.264 codec degradation (low-bitrate video), no KD
#   R10_ft_codec_m2m3 : R9 + M2+M3 KD
#   LL_noKD / LL_m2m3 : low-label regime, only s3-s5 transcribed; KD run also uses the other 25 speakers unlabeled
# Then GRID-Crowd evaluation for every run that does not have it yet.
$Workers  = 8
$Labeled  = @("s3", "s4", "s5")
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

$R1 = Join-Path $Runs "R1_base_clean\best.pt"
$Ft = @("--init", $R1, "--degrade", "--max_minutes", 0, "--max_steps", 5416, "--val_every", 500)
$LL = @("--degrade", "--max_minutes", 0, "--max_steps", 12763, "--val_every", 1000)
Train-Eval "R9_ft_codec"       ($Ft + @("--codec", 0.5, "--kd", "none"))
Train-Eval "R10_ft_codec_m2m3" ($Ft + @("--codec", 0.5, "--kd", "m2m3"))
Train-Eval "LL_noKD"           ($LL + @("--kd", "none", "--train_spk") + $Labeled)
Train-Eval "LL_m2m3"           ($LL + @("--kd", "m2m3", "--labeled_spk") + $Labeled)

$ckpts = Get-ChildItem $Runs -Directory | Where-Object { (Test-Path "$($_.FullName)\best.pt") -and -not (Test-Path "$($_.FullName)\eval_crowd.json") -and $_.Name -ne "smoke_conformer" } | ForEach-Object { "$($_.FullName)\best.pt" }
if ($ckpts) { python -m eval.eval_crowd --ckpt @ckpts --workers 6 2>&1 | Tee-Object -FilePath (Join-Path $Runs "crowd.out") -Append }
Write-Host "=== followup queue finished $(Get-Date -Format HH:mm)"
