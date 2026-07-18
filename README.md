# Risk-Asymmetric Selective RUL Prediction

This project implements the proposed workflow for NASA C-MAPSS-style aircraft-engine
remaining-useful-life (RUL) data:

1. A temporal convolutional network (TCN) predicts low, median, and high RUL
   quantiles from sliding sensor windows.
2. A held-out, unit-disjoint calibration set supplies asymmetric conformalized
   quantile-regression (CQR) corrections. Misses where true RUL is *below* the
   lower prediction bound receive a smaller error budget, reflecting the greater
   danger of optimistic RUL estimates.
3. A selective router accepts only calibrated intervals below a configured width and
   escalates all other engines for human review.

The loader accepts standard whitespace-delimited C-MAPSS files (`train_FD001.txt`,
`test_FD001.txt`, and `RUL_FD001.txt`). FD001 can be used for baseline training and
FD004/N-MAPSS-formatted data for stress testing. N-MAPSS data must first be exported
to the documented C-MAPSS-compatible column layout.

## Quick start

```bash
PROJECT_ROOT=/path/to/risk-asymmetric-rul
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/venv/bin/activate"
mkdir -p "$PROJECT_ROOT/data/cmapss"
curl -L -o "$PROJECT_ROOT/data/cmapss/train_FD001.txt" https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/train_FD001.txt
curl -L -o "$PROJECT_ROOT/data/cmapss/test_FD001.txt"  https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/test_FD001.txt
curl -L -o "$PROJECT_ROOT/data/cmapss/RUL_FD001.txt"   https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/RUL_FD001.txt
python -m rul_selective.sanity_check
python "$PROJECT_ROOT/train.py" --train-file "$PROJECT_ROOT/data/cmapss/train_FD001.txt" --output-dir "$PROJECT_ROOT/runs/fd001"
```

For an external test set, add `--test-file` and `--rul-file`. Results include interval
coverage, unsafe lower-tail misses, selective risk, and abstention rate.

## Safety note

Conformal coverage assumes calibration and deployment examples are exchangeable. The
implementation uses one terminal window per calibration engine to avoid overlapping
windows artificially inflating calibration sample size. It is a research prototype, not
an aviation-certified maintenance system.
