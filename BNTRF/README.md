# BNTRF

This package implements a biased nonnegative tensor ring factorization model for third-order tensors with PSO-based adaptive selection of two regularization parameters.

## Data format

Each line in the train, validation, and test files should follow:

```text
aID::bID::cID::value
```

The code uses 1-based indices. Index 0 is reserved.

## Run

```bash
pip install -r requirements.txt
python -m bntrf_pso.main --train data/D1/tr.txt --valid data/D1/va.txt --test data/D1/te.txt
```

Common options:

```bash
python -m bntrf_pso.main \
  --train data/D1/tr.txt \
  --valid data/D1/va.txt \
  --test data/D1/te.txt \
  --rank 5 \
  --train-round 1000 \
  --population 5 \
  --min-lambda-reg 0.01 \
  --max-lambda-reg 0.10 \
  --min-lambda-b 0.01 \
  --max-lambda-b 0.10
```
