param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [int]$PollSeconds = 60,
  [int]$StableChecks = 2,
  [switch]$ForceVal3
)

$ErrorActionPreference = "Stop"

$finalCkpt = Join-Path $RepoRoot "artifacts\checkpoints_conch_masklut_g4c_hierarchical\final_hier_g4c_combined_folds.pth"
$val3Ckpt = Join-Path $RepoRoot "artifacts\checkpoints_conch_masklut_g4c_hierarchical\best_Val3_hier_g4c.pth"
$pythonExe = Join-Path $RepoRoot "prostata_env\Scripts\python.exe"
$trainScript = Join-Path $RepoRoot "src\training_conch_g4c_hierarchical.py"
$baseVal3 = Join-Path $RepoRoot "artifacts\checkpoints_conch_masklut\best_Val3_0.8201.pth"
$logDir = Join-Path $RepoRoot "artifacts\logs"
$logFile = Join-Path $logDir "watch_hier_allfolds_then_val3.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Log([string]$Message) {
  $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Host $line
  Add-Content -Path $logFile -Value $line
}

Log "RepoRoot: $RepoRoot"
Log "Watching final checkpoint: $finalCkpt"

if (-not (Test-Path $pythonExe)) {
  throw "Python venv not found: $pythonExe"
}
if (-not (Test-Path $trainScript)) {
  throw "Training script not found: $trainScript"
}
if (-not (Test-Path $baseVal3)) {
  throw "Val3 base checkpoint not found: $baseVal3"
}

if ((Test-Path $val3Ckpt) -and -not $ForceVal3) {
  Log "Val3 checkpoint already exists. Use -ForceVal3 to launch anyway: $val3Ckpt"
  exit 0
}

$lastLength = -1
$stableCount = 0

while ($stableCount -lt $StableChecks) {
  if (-not (Test-Path $finalCkpt)) {
    Log "All-folds checkpoint not found yet. Sleeping ${PollSeconds}s..."
    Start-Sleep -Seconds $PollSeconds
    continue
  }

  $item = Get-Item $finalCkpt
  if ($item.Length -gt 0 -and $item.Length -eq $lastLength) {
    $stableCount += 1
    Log "Checkpoint stable check $stableCount/$StableChecks, size=$($item.Length)"
  } else {
    $stableCount = 0
    $lastLength = $item.Length
    Log "Checkpoint detected/changed, size=$($item.Length). Waiting for stability..."
  }

  if ($stableCount -lt $StableChecks) {
    Start-Sleep -Seconds $PollSeconds
  }
}

Log "All-folds checkpoint is stable. Launching Val3..."

Push-Location $RepoRoot
try {
  & $pythonExe $trainScript `
    --fold Val3 `
    --base-checkpoint $baseVal3 `
    --unfreeze-last 4 `
    --learning-rate 4e-5 `
    --batch-size 6 --grad-accum 2 `
    --ema --ema-decay 0.999 `
    --max-epochs 15 `
    --sampler-gg5 1.8 --sampler-gg4 1.3 --sampler-gg3 1.8 --sampler-gg4c 2.2 `
    --g4c-gray-min 125 `
    --checkpoint-name best_Val3_hier_g4c.pth

  $exitCode = $LASTEXITCODE
  Log "Val3 finished with exit code $exitCode"
  exit $exitCode
} finally {
  Pop-Location
}
