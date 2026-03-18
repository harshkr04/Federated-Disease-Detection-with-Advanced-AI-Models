# ============================================================
# Run all ISIC training and evaluation scripts
# ============================================================
# This script trains all 5 models on the ISIC dataset
# using the SAME hyperparameters as HAM10000.
#
# Usage:    .\run_isic.ps1
# ============================================================

Write-Host "============================================================"
Write-Host "  ISIC DATASET — FULL TRAINING PIPELINE"
Write-Host "============================================================"
Write-Host ""

# --- Step 1: Centralized CNN ---
Write-Host ">>> Step 1/5: Training Centralized CNN on ISIC..."
python train_cnn_isic.py
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: CNN training failed!"; exit $LASTEXITCODE }

# --- Step 2: Centralized Hybrid ---
Write-Host ""
Write-Host ">>> Step 2/5: Training Centralized Hybrid on ISIC..."
python train_hybrid_isic.py
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Hybrid training failed!"; exit $LASTEXITCODE }

# --- Step 3: Federated FedAvg ---
Write-Host ""
Write-Host ">>> Step 3/5: Training FedAvg on ISIC..."
python federated_train_isic.py --algorithm fedavg --rounds 5 --local-epochs 2
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: FedAvg training failed!"; exit $LASTEXITCODE }

# --- Step 4: Federated FedProx ---
Write-Host ""
Write-Host ">>> Step 4/5: Training FedProx on ISIC..."
python federated_train_isic.py --algorithm fedprox --rounds 5 --local-epochs 2
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: FedProx training failed!"; exit $LASTEXITCODE }

# --- Step 5: Federated MOON ---
Write-Host ""
Write-Host ">>> Step 5/5: Training MOON on ISIC..."
python federated_train_isic.py --algorithm moon --rounds 5 --local-epochs 2
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: MOON training failed!"; exit $LASTEXITCODE }

# --- Step 6: Evaluate all ISIC models ---
Write-Host ""
Write-Host ">>> Evaluating all ISIC models on validation set..."
python evaluate_isic.py

Write-Host ""
Write-Host ">>> Evaluating all ISIC models on test set..."
python evaluate_isic.py --use-test-set

# --- Step 7: Cross-dataset comparison ---
Write-Host ""
Write-Host ">>> Running cross-dataset comparison (HAM10000 vs ISIC)..."
python cross_dataset_train.py --skip-training

Write-Host ""
Write-Host "============================================================"
Write-Host "  ALL DONE! Results saved in:"
Write-Host "    weights/isic/         — trained model weights"
Write-Host "    results/isic/         — ISIC evaluation results"
Write-Host "    results/cross_dataset/ — HAM10000 vs ISIC comparison"
Write-Host "============================================================"
