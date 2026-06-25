#!/usr/bin/env bash
# ============================================================
# Download the fine-tuned BGE model from GCP VM → local machine
# ============================================================
#
# Run this script locally (PowerShell / Git Bash / WSL) after
# the fine-tuning on the GCP VM finishes.
#
# Usage:
#   bash scripts/gcp_download_model.sh
#
# Requires: gcloud CLI configured with project + credentials

set -euo pipefail

PROJECT="your-gcp-project-id"   # ← change this
VM_NAME="rag-thesis-gpu"
ZONE="us-central1-a"
BUCKET="gs://rag-thesis-models"
LOCAL_MODEL_DIR="models/bge_finetuned"

echo "============================================================"
echo " Step 1: Check training status on VM"
echo "============================================================"
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT}" \
    --command "tail -50 /home/\$(whoami)/graph-table-rag/logs/bge_finetune.log 2>/dev/null || echo 'Log file not found — training may still be running or log is elsewhere'"

echo ""
echo "============================================================"
echo " Step 2: Check if best model directory exists on VM"
echo "============================================================"
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT}" \
    --command "ls -lh /home/\$(whoami)/graph-table-rag/models/bge_finetuned/best/ 2>/dev/null || echo 'Model not yet saved — training still in progress'"

echo ""
echo "============================================================"
echo " Step 3: Copy model from VM to GCS bucket"
echo "============================================================"
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT}" \
    --command "
        cd /home/\$(whoami)/graph-table-rag
        gsutil -m cp -r models/bge_finetuned/ ${BUCKET}/bge_finetuned/ && echo 'Uploaded to GCS OK'
    "

echo ""
echo "============================================================"
echo " Step 4: Download model from GCS to local machine"
echo "============================================================"
mkdir -p "${LOCAL_MODEL_DIR}"
gsutil -m cp -r "${BUCKET}/bge_finetuned/best"  "${LOCAL_MODEL_DIR}/"
gsutil -m cp -r "${BUCKET}/bge_finetuned/final" "${LOCAL_MODEL_DIR}/" || true
gsutil    cp    "${BUCKET}/bge_finetuned/training_summary.json" "${LOCAL_MODEL_DIR}/" || true

echo ""
echo "============================================================"
echo " Done! Model saved to: ${LOCAL_MODEL_DIR}/best"
echo ""
echo " Next steps:"
echo "   python scripts/eval_retrieval.py --pipeline text_finetuned --dataset wikitq --max-samples 500 --split validation"
echo "   python scripts/eval_retrieval.py --pipeline text_finetuned --dataset wikitq --max-samples 500 --split validation --output data/results/text_finetuned_wikitq_500"
echo "============================================================"
