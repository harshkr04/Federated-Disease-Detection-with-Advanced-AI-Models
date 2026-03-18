python cross_dataset_train.py --model fedavg --epochs 10 --rounds 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python cross_dataset_train.py --model fedprox --epochs 10 --rounds 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python cross_dataset_train.py --model moon --epochs 10 --rounds 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python cross_dataset_train.py --skip-training
