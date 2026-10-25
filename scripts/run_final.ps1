# Last experiments, run after scripts/run_followup.ps1 (waits for it). Step budgets as in the follow-up queue.
#   R11_ft_crowdsim : R9 recipe (degrade + codec) plus ground-truth-box mouth crops of 8,400 low-bitrate crowd
#                     composites built from the 28 TRAINING speakers only (data/crowd.py --out crowd_train, seed 7;
#                     data/crowd_crops.py), repeated x2 (~37% of samples). Same generator as the test set, different
#                     speakers/scenes: an upper bound for matched simulation.
#   R4b_m1_clean_steps : M1 from scratch with exactly R1's step count (R4's time budget gave it 9% more steps).
#   R9s1 / R10s1 : second seed of the codec pair (R10 vs R9 was the only KD gain, -1.2 pt at 16 px, ~1.7x seed noise).
if (-not $env:L2L_WORK) { $env:L2L_WORK = "E:\College\IE_Project\work" }
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location (Join-Path $PSScriptRoot "..")
$Runs = Join-Path $env:L2L_WORK "runs"
$Common = @("--backend", "conformer", "--bs", 16, "--accum", 1, "--workers", 8)
while (-not (Select-String -Path (Join-Path $Runs "followup.out") -Pattern "followup queue finished" -Quiet)) { Start-Sleep 60 }
if (-not (Test-Path (Join-Path $env:L2L_WORK "crowd_train\crops.json"))) { python -m data.crowd_crops --root crowd_train --workers 6 }

function Train-Eval([string]$Name, [string[]]$TrainArgs) {
    Write-Host "=== $(Get-Date -Format HH:mm) $Name : $TrainArgs"
    python distill/train.py --name $Name @Common @TrainArgs 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
    $ckpt = Join-Path $Runs "$Name\best.pt"
    if ((Test-Path $ckpt) -and -not (Test-Path (Join-Path $Runs "$Name\eval_summary.json"))) {
        python eval/evaluate.py --ckpt $ckpt --workers 8 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
    }
}
$R1 = Join-Path $Runs "R1_base_clean\best.pt"
Train-Eval "R11_ft_crowdsim" @("--init", $R1, "--degrade", "--codec", 0.5, "--crowd_train", "crowd_train", "--crowd_rep", 2,
                              "--kd", "none", "--max_minutes", 0, "--max_steps", 5416, "--val_every", 500)
python -m eval.eval_crowd --ckpt (Join-Path $Runs "R11_ft_crowdsim\best.pt") --workers 6 2>&1 | Tee-Object -FilePath (Join-Path $Runs "crowd.out") -Append
Train-Eval "R4b_m1_clean_steps" @("--m1", "--max_minutes", 0, "--max_steps", 12763, "--val_every", 1000)
$Codec = @("--init", $R1, "--degrade", "--codec", 0.5, "--max_minutes", 0, "--max_steps", 5416, "--val_every", 500, "--seed", 1)
Train-Eval "R9s1_ft_codec"       ($Codec + @("--kd", "none"))
Train-Eval "R10s1_ft_codec_m2m3" ($Codec + @("--kd", "m2m3"))
Write-Host "=== final queue finished $(Get-Date -Format HH:mm)"
