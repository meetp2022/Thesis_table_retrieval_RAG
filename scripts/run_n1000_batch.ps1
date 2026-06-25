#!/usr/bin/env pwsh
# ============================================================
# Batch script: scale headline evaluations from n=500 to n=1000
# ============================================================
# Runs 5 WikiTQ pipelines at n=1000 and 7 FinQA pipelines at n=883
# (full FinQA validation split). Handles the FinQA-GNN model swap
# automatically, verifies via hash checks, and swaps back at the end.
#
# Each pipeline's stdout/stderr is logged separately under logs/.
# The script continues even if one pipeline fails so a single error
# doesn't abort the whole batch.
#
# Total compute time: ~3.5 hours on CPU. Safe to walk away.
#
# Usage:
#   .\scripts\run_n1000_batch.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$ScriptStart = Get-Date

# ── Use the venv Python explicitly (PowerShell subprocesses do not ─
#    inherit the venv even if you activated it in the parent shell) ─
$VenvPy = "venv/Scripts/python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "ERROR: venv Python not found at $VenvPy" -ForegroundColor Red
    Write-Host "       Confirm the venv exists at .\venv and try again." -ForegroundColor Red
    exit 1
}
Write-Host "Using venv Python: $VenvPy" -ForegroundColor Green

# Quick import check so we catch dependency issues before burning time
& $VenvPy -c "import loguru, yaml, sentence_transformers, faiss, datasets, networkx, torch_geometric; print('Dependencies OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: required Python packages are missing in the venv." -ForegroundColor Red
    Write-Host "       Run: .\venv\Scripts\Activate.ps1; pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ── Set up directories ────────────────────────────────────────
$LogDir = "logs/n1000_batch"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

$ModelDir = "models/graph_encoder"
$Canonical = "$ModelDir/best_encoder.pt"
$WikiBackup = "$ModelDir/best_encoder_wikitq.pt"
$FinqaModel = "$ModelDir/best_encoder_finqa.pt"

# Sanity-check the model files exist before we start
foreach ($f in @($Canonical, $WikiBackup, $FinqaModel)) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: required model file not found: $f" -ForegroundColor Red
        exit 1
    }
}

# Confirm canonical and WikiTQ backup are identical (i.e., we are starting clean)
$canonicalHash = (Get-FileHash $Canonical).Hash
$wikiHash      = (Get-FileHash $WikiBackup).Hash
if ($canonicalHash -ne $wikiHash) {
    Write-Host "ERROR: best_encoder.pt does not match best_encoder_wikitq.pt." -ForegroundColor Red
    Write-Host "       Resolve before running this batch (the script assumes" -ForegroundColor Red
    Write-Host "       the canonical WikiTQ GNN is currently in place)." -ForegroundColor Red
    exit 1
}
Write-Host "Pre-flight: canonical model = WikiTQ GNN. OK." -ForegroundColor Green
Write-Host ""

# ── Helper that runs one command, logs it, and reports timing ───
function Run-Step {
    param([string]$Label, [string]$LogName, [scriptblock]$Command)
    $start = Get-Date
    Write-Host ("=" * 72)
    Write-Host "[$Label] starting at $($start.ToString('HH:mm:ss'))"
    Write-Host ("=" * 72)
    $log = "$LogDir/$LogName.log"
    & $Command *>&1 | Tee-Object -FilePath $log
    $duration = ((Get-Date) - $start).TotalMinutes
    Write-Host ""
    Write-Host "[$Label] finished — duration $([math]::Round($duration, 1)) min — log: $log"
    Write-Host ""
}

# ── PHASE A: WikiTQ pipelines at n=1000 (canonical WikiTQ GNN) ───
Write-Host ""
Write-Host "###########################################################"
Write-Host "#  PHASE A — WikiTQ at n = 1000 (canonical WikiTQ GNN)   #"
Write-Host "###########################################################"
Write-Host ""

Run-Step "WikiTQ 1/5 generic BGE"           "01_wikitq_text_n1000" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline text --dataset wikitq `
        --max-samples 1000 --split validation `
        --output data/results/text_wikitq_1000
}

Run-Step "WikiTQ 2/5 fine-tuned BGE"        "02_wikitq_text_finetuned_n1000" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline text_finetuned --dataset wikitq `
        --max-samples 1000 --split validation `
        --output data/results/text_finetuned_wikitq_1000
}

Run-Step "WikiTQ 3/5 graph"                 "03_wikitq_graph_n1000" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline graph --dataset wikitq `
        --max-samples 1000 --split validation `
        --output data/results/graph_wikitq_1000
}

Run-Step "WikiTQ 4/5 hybrid no-rerank"      "04_wikitq_hybrid_norerank_n1000" {
    & "$VenvPy" scripts/eval_hybrid_finetuned.py --max-samples 1000 --no-rerank `
        --output data/results/hybrid_finetuned_norerank_a0.7_wikitq_1000
}

Run-Step "WikiTQ 5/5 hybrid + rerank"       "05_wikitq_hybrid_rerank_n1000" {
    & "$VenvPy" scripts/eval_hybrid_finetuned.py --max-samples 1000 `
        --output data/results/hybrid_finetuned_rerank_a0.7_wikitq_1000
}

# ── PHASE B: FinQA cross-domain runs (canonical WikiTQ GNN) ──────
Write-Host ""
Write-Host "###########################################################"
Write-Host "#  PHASE B — FinQA at n = 883 (cross-domain stage)        #"
Write-Host "###########################################################"
Write-Host ""

Run-Step "FinQA 1/7 generic BGE"            "06_finqa_text_n883" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline text --dataset finqa `
        --max-samples 883 --split validation `
        --output data/results/text_finqa_883
}

Run-Step "FinQA 2/7 fine-tuned BGE"         "07_finqa_text_finetuned_n883" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline text_finetuned --dataset finqa `
        --max-samples 883 --split validation `
        --output data/results/text_finetuned_finqa_883
}

Run-Step "FinQA 3/7 graph (cross-domain)"   "08_finqa_graph_crossdomain_n883" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline graph --dataset finqa `
        --max-samples 883 --split validation `
        --output data/results/graph_finqa_crossdomain_883
}

# ── Swap to FinQA-trained GNN for the matched-hybrid runs ───────
Write-Host ""
Write-Host "Swapping in FinQA-trained GNN..."
Copy-Item $FinqaModel $Canonical -Force
$check = (Get-FileHash $Canonical).Hash
$expected = (Get-FileHash $FinqaModel).Hash
if ($check -ne $expected) {
    Write-Host "ERROR: swap to FinQA GNN failed (hash mismatch). Aborting." -ForegroundColor Red
    Copy-Item $WikiBackup $Canonical -Force
    exit 1
}
Write-Host "Swap verified. Canonical model = FinQA GNN." -ForegroundColor Green
Write-Host ""

# ── PHASE C: FinQA matched runs (FinQA GNN) ──────────────────────
Write-Host ""
Write-Host "###########################################################"
Write-Host "#  PHASE C — FinQA at n = 883 (in-domain FinQA GNN)       #"
Write-Host "###########################################################"
Write-Host ""

Run-Step "FinQA 4/7 graph (in-domain)"      "09_finqa_graph_indomain_n883" {
    & "$VenvPy" scripts/eval_retrieval.py --pipeline graph --dataset finqa `
        --max-samples 883 --split validation `
        --output data/results/graph_finqa_indomain_883
}

Run-Step "FinQA 5/7 hybrid no-rerank"       "10_finqa_hybrid_norerank_n883" {
    & "$VenvPy" scripts/eval_hybrid_finetuned.py --dataset finqa --max-samples 883 `
        --no-rerank `
        --output data/results/hybrid_finetuned_norerank_a0.7_finqa_883_with_finqa_gnn
}

Run-Step "FinQA 6/7 hybrid + rerank"        "11_finqa_hybrid_rerank_n883" {
    & "$VenvPy" scripts/eval_hybrid_finetuned.py --dataset finqa --max-samples 883 `
        --output data/results/hybrid_finetuned_rerank_a0.7_finqa_883_with_finqa_gnn
}

# ── Swap WikiTQ GNN back as canonical (CRITICAL) ─────────────────
Write-Host ""
Write-Host "Restoring canonical WikiTQ GNN..."
Copy-Item $WikiBackup $Canonical -Force
$check    = (Get-FileHash $Canonical).Hash
$expected = (Get-FileHash $WikiBackup).Hash
if ($check -ne $expected) {
    Write-Host "ERROR: restoring WikiTQ GNN failed. INSPECT MANUALLY." -ForegroundColor Red
    exit 1
}
Write-Host "Swap-back verified. Canonical model = WikiTQ GNN." -ForegroundColor Green
Write-Host ""

# ── Summary ──────────────────────────────────────────────────────
$totalDuration = ((Get-Date) - $ScriptStart).TotalMinutes
Write-Host ""
Write-Host ("=" * 72)
Write-Host "  Batch complete."
Write-Host "  Total wall-clock: $([math]::Round($totalDuration, 1)) min"
Write-Host "  Logs: $LogDir"
Write-Host ("=" * 72)
Write-Host ""
Write-Host "Next: run bootstrap CIs against the new n=1000 / n=883 results."
Write-Host "Example:"
Write-Host "  python scripts/bootstrap_ci.py \"
Write-Host "      data/results/text_finetuned_wikitq_1000 \"
Write-Host "      data/results/hybrid_finetuned_rerank_a0.7_wikitq_1000 \"
Write-Host "      --label-a 'FT-BGE' --label-b 'Hybrid+Rerank'"
Write-Host ""
Write-Host "Also: confirm verify_all_results.py shows all new folders:"
Write-Host "  python scripts/verify_all_results.py --filter 1000"
Write-Host "  python scripts/verify_all_results.py --filter 883"
