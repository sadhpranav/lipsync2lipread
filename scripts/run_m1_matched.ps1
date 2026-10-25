# M1 check at an equal step budget: R4 (time budget) got 13,913 steps vs R1's 12,763 in 60 min, which confounds
# the comparison. R4b reruns M1 from scratch with exactly R1's step count. Starts after run_followup.ps1.
if (-not $env:L2L_WORK) { $env:L2L_WORK = "E:\College\IE_Project\work" }
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location (Join-Path $PSScriptRoot "..")
$Runs = Join-Path $env:L2L_WORK "runs"
while (-not (Select-String -Path (Join-Path $Runs "followup.out") -Pattern "followup queue finished" -Quiet)) { Start-Sleep 60 }
$Name = "R4b_m1_clean_steps"
Write-Host "=== $(Get-Date -Format HH:mm) $Name"
python distill/train.py --name $Name --m1 --backend conformer --bs 16 --accum 1 --workers 8 --max_minutes 0 --max_steps 12763 --val_every 1000 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
$ckpt = Join-Path $Runs "$Name\best.pt"
if ((Test-Path $ckpt) -and -not (Test-Path (Join-Path $Runs "$Name\eval_summary.json"))) {
    python eval/evaluate.py --ckpt $ckpt --workers 8 2>&1 | Tee-Object -FilePath (Join-Path $Runs "$Name.out") -Append
}
Write-Host "=== m1 queue finished $(Get-Date -Format HH:mm)"
