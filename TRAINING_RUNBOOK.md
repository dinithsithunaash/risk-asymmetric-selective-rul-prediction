# Training Runbook (Provided Dataset)

This runbook documents the exact steps to train the project with your provided C-MAPSS-format dataset.

Use a generic project root in all commands:

```bash
PROJECT_ROOT=/path/to/risk-asymmetric-rul
```

## 1. Open the project directory

```bash
cd "$PROJECT_ROOT"
```

## 2. Activate Python 3.12 virtual environment

```bash
source "$PROJECT_ROOT/venv/bin/activate"
python --version
```

Expected Python version: `3.12.x`.

## 3. Confirm required files are available

Your dataset should be in standard C-MAPSS text format:

- `$PROJECT_ROOT/data/cmapss/train_FD001.txt` (required for training)
- `$PROJECT_ROOT/data/cmapss/test_FD001.txt` (optional, for evaluation)
- `$PROJECT_ROOT/data/cmapss/RUL_FD001.txt` (required only if test file is used)

If you do not have them yet, pull them with curl:

```bash
mkdir -p "$PROJECT_ROOT/data/cmapss"
curl -L -o "$PROJECT_ROOT/data/cmapss/train_FD001.txt" https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/train_FD001.txt
curl -L -o "$PROJECT_ROOT/data/cmapss/test_FD001.txt"  https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/test_FD001.txt
curl -L -o "$PROJECT_ROOT/data/cmapss/RUL_FD001.txt"   https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/RUL_FD001.txt
```

Verification:

```bash
ls -lh "$PROJECT_ROOT/data/cmapss/"
```

## 4. Optional: verify CUDA and dual GPU visibility

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"
```

If CUDA is available and `gpu_count > 1`, training will automatically use both GPUs through `DataParallel`.

## 5. Run project sanity check

```bash
python -m rul_selective.sanity_check
```

## 6. Train with your provided training dataset

```bash
python train.py \
  --train-file "$PROJECT_ROOT/data/cmapss/train_FD001.txt" \
  --output-dir "$PROJECT_ROOT/runs/fd001_provided" \
  --epochs 30 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --window-size 30 \
  --calibration-fraction 0.20 \
  --alpha 0.10 \
  --unsafe-miscoverage-fraction 0.20 \
  --overestimate-penalty 3.0 \
  --max-interval-width 35.0 \
  --seed 7
```

## 7. Train + evaluate (if test labels are available)

```bash
python train.py \
  --train-file "$PROJECT_ROOT/data/cmapss/train_FD001.txt" \
  --test-file "$PROJECT_ROOT/data/cmapss/test_FD001.txt" \
  --rul-file "$PROJECT_ROOT/data/cmapss/RUL_FD001.txt" \
  --output-dir "$PROJECT_ROOT/runs/fd001_provided_eval" \
  --epochs 30 \
  --batch-size 128
```

## 8. Outputs to review

After training, check:

- `runs/.../model.pt` — trained model checkpoint
- `runs/.../metadata.json` — device, GPU count, calibration info
- `runs/.../test_metrics.json` — evaluation metrics (only when test+rul are provided)
- Example path: `$PROJECT_ROOT/runs/fd001_provided_eval/test_metrics.json`

Quick review:

```bash
cat "$PROJECT_ROOT/runs/fd001_provided/metadata.json"
cat "$PROJECT_ROOT/runs/fd001_provided_eval/test_metrics.json"
```

## 9. Notes for reporting

For your project report, record:

1. Dataset variant used (e.g., FD001 / FD004 / converted N-MAPSS).
2. Final training command.
3. Hardware used (Dual NVIDIA RTX A2000 12GB).
4. Key metrics from `test_metrics.json` (coverage, selective risk, abstention rate).

---

This runbook focuses on training and evaluation only, so you can perform GitHub upload steps manually afterward.
