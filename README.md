# BNTRF

This repository provides the source code and datasets for the paper **"A Highly-Accurate Approach to Dynamic Networks Representation Learning based on Biased Nonnegative Tensor Ring Factorization"**.

The repository is organized as follows:

```text
BNTRF/
├── BNTRF/
│   ├── bntrf_pso/          # Python implementation of BNTRF with PSO-based hyperparameter adaptation
│   ├── requirements.txt    # Required Python packages
│   └── run.py              # Entry script
├── data/
│   ├── D1/
│   ├── D2/
│   ├── D3/
│   ├── D4/
│   ├── D5/
│   └── D6/                # Six dynamic network datasets and fixed data splits
│   └── 4th-order tensor/
│   └── UNSW-NB15/
└── README.md
```

## Datasets

The six datasets are collected from a real metropolitan area network (MAN) in a city in China. Due to privacy and security requirements, detailed device attributes, geographical sub-area labels, and wired/wireless access types are not released. Each terminal device is anonymized and represented by an integer ID.

Each record has the following format:

```text
source_device_id::destination_device_id::time_point::value
```

where `value` denotes the packet size transmitted from the source terminal device to the destination terminal device at the corresponding time point. One time point corresponds to a 10-minute interval. Therefore, each dataset describes a dynamic weighted network whose interaction weights change over time.

The six datasets correspond to different monitoring periods of the same MAN system. They have different numbers of nodes, time points, known entries, and densities.

| Dataset | Nodes | Time Points | Known Entries | Density |
|---|---:|---:|---:|---:|
| D1 | 352680 | 1080 | 1697654 | 1.26E-8 |
| D2 | 269028 | 865 | 1189151 | 1.90E-8 |
| D3 | 208634 | 748 | 720459 | 2.21E-8 |
| D4 | 148453 | 672 | 457273 | 3.08E-8 |
| D5 | 73281 | 556 | 48277 | 1.61E-8 |
| D6 | 37485 | 423 | 17739 | 2.98E-8 |

## Data Files

Each dataset folder contains the original data and the fixed train/validation/test split used in the experiments. For example, `data/D1/` contains:

```text
D1_original.txt       # Original unnormalized records
D1_train_70%.txt      # Training set used in the paper
D1_valid_10%.txt      # Validation set used in the paper
D1_test 20%.txt       # Test set used in the paper
```

The same naming rule is used for D2--D6. The `*_original.txt` files contain the unnormalized original packet-size records. The `*_train_70%.txt`, `*_valid_10%.txt`, and `*_test 20%.txt` files provide the fixed 70%/10%/20% split used for reproduction. No random re-splitting is required.

## Preprocessing

The preprocessing used for the released datasets is summarized below.

1. Terminal devices are anonymized as integer IDs.
2. Continuous monitoring records are organized into 10-minute time points.
3. Each record is represented as `source_device_id::destination_device_id::time_point::value`.
4. The original unnormalized records are provided in `*_original.txt`.
5. The fixed training, validation, and test files are provided for each dataset.
6. The released split files should be used directly to reproduce the reported experiments.

## Code

The code in `BNTRF/` implements the BNTRF model and the PSO-based adaptive selection of regularization coefficients. The main modules are located in `BNTRF/bntrf_pso/`, and the entry script is `BNTRF/run.py`.

The required packages are listed in:

```text
BNTRF/requirements.txt
```

## Reproducibility Notes

This repository releases all six datasets used in the paper, including the original records and the fixed train/validation/test splits. The data format, preprocessing procedure, dataset statistics, and code structure are described above to support reproducibility.

Because the raw MAN records contain sensitive network information, device identities and detailed device attributes are anonymized. The released data retain the source-device ID, destination-device ID, time point, and packet-size value required for reproducing the tensor-based experiments.
