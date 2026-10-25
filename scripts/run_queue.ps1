# Full experiment queue (warm-start protocol). Idempotent: finished runs (DONE marker) are skipped,
# interrupted runs resume from last.pt. Run from anywhere:  pwsh scripts/run_queue.ps1
# Runs are sized by wall clock; the cosine LR schedule ends exactly at the budget.
# ---- parameters ----
$Backend      = "conformer"
$ScratchMin   = 60      # R1 / R4 (from scratch)
$FtMin        = 25      # each fine-tune from R1
$Bs           = 16
$Accum        = 1
$Workers      = 8
$ValEvery     = 1000    # optimizer steps between validations (fine-tunes use half)
if (-not $env:L2L_WORK) { $env:L2L_WORK = "E:\College\IE_Project\work" }
# --------------------
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location (Join-Path $PSScriptRoot "..")
$Runs = Join-Path $env:L2L_WORK "runs"
$Common = @("--backend", $Backend, "--bs", $Bs, "--accum", $Accum, "--workers", $Workers)

function Train-Eval([string]$Name, [string[]]$TrainArgs) {
    Write-Host "=== $(Get-Date -Format HH:mm) $Name : $TrainArgs"
    python distill/train.py --name $Name @Common @TrainArgs 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
    $ckpt = Join-Path $Runs "$Name\best.pt"
    if ((Test-Path $ckpt) -and -not (Test-Path (Join-Path $Runs "$Name\eval_summary.json"))) {
        python eval/evaluate.py --ckpt $ckpt --workers $Workers 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
    }
}

New-Item -ItemType Directory -Force $Runs | Out-Null
$R1 = Join-Path $Runs "R1_base_clean\best.pt"
$Scratch = @("--max_minutes", $ScratchMin, "--val_every", $ValEvery)
$Ft = @("--init", $R1, "--degrade", "--max_minutes", $FtMin, "--val_every", [int]($ValEvery / 2))

Train-Eval "R1_base_clean"   $Scratch
Train-Eval "R2_ft_noKD"      ($Ft + @("--kd", "none"))
Train-Eval "R3_ft_m2m3"      ($Ft + @("--kd", "m2m3"))
Train-Eval "R5_ft_m3"        ($Ft + @("--kd", "m3"))
Train-Eval "R6_ft_m2"        ($Ft + @("--kd", "m2"))
Train-Eval "R8_ft_m2m3nce"   ($Ft + @("--kd", "m2m3nce"))
Train-Eval "R4_m1_clean"     ($Scratch + @("--m1"))
Train-Eval "R2s1_ft_noKD"    ($Ft + @("--kd", "none", "--seed", "1"))
Train-Eval "R3s1_ft_m2m3"    ($Ft + @("--kd", "m2m3", "--seed", "1"))
Write-Host "=== queue finished $(Get-Date -Format HH:mm)"
