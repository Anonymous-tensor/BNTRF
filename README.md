\# BNTRF: Biased Nonnegative Tensor Ring Factorization



This repository provides the source code and datasets for reproducing the experiments of \*\*Biased Nonnegative Tensor Ring Factorization (BNTRF)\*\* for representation learning on high-dimensional incomplete (HDI) dynamic networks.



BNTRF models a dynamic weighted network as a third-order HDI tensor. Each observed entry has the form



```text

i::j::k::value

```



where `i` is the source terminal-device ID, `j` is the destination terminal-device ID, `k` is the time point, and `value` is the packet-size interaction weight between the two terminal devices at the corresponding time point.



\## 1. Repository Structure



```text

BNTRF/

├── README.md

├── data/

│   ├── D1/

│   │   ├── D1\_original.txt

│   │   ├── D1\_train\_70%.txt

│   │   ├── D1\_valid\_10%.txt

│   │   └── D1\_test\_20%.txt

│   ├── D2/

│   │   ├── D2\_original.txt

│   │   ├── D2\_train\_70%.txt

│   │   ├── D2\_valid\_10%.txt

│   │   └── D2\_test\_20%.txt

│   ├── D3/

│   │   ├── D3\_original.txt

│   │   ├── D3\_train\_70%.txt

│   │   ├── D3\_valid\_10%.txt

│   │   └── D3\_test\_20%.txt

│   ├── D4/

│   │   ├── D4\_original.txt

│   │   ├── D4\_train\_70%.txt

│   │   ├── D4\_valid\_10%.txt

│   │   └── D4\_test\_20%.txt

│   ├── D5/

│   │   ├── D5\_original.txt

│   │   ├── D5\_train\_70%.txt

│   │   ├── D5\_valid\_10%.txt

│   │   └── D5\_test\_20%.txt

│   └── D6/

│       ├── D6\_original.txt

│       ├── D6\_train\_70%.txt

│       ├── D6\_Valid\_10%.txt

│       └── D6\_test\_20%.txt

└── BNTRF/

&#x20;   ├── requirements.txt

&#x20;   ├── run.py

&#x20;   └── bntrf\_pso/

&#x20;       ├── \_\_init\_\_.py

&#x20;       ├── config.py

&#x20;       ├── data\_utils.py

&#x20;       ├── initialize.py

&#x20;       ├── kernels.py

&#x20;       ├── main.py

&#x20;       ├── pso\_utils.py

&#x20;       └── trainer.py

```



The folder `data/` contains all six datasets used in the experiments. For each dataset, the original file and the fixed train/validation/test split files are provided. The folder `BNTRF/` contains the Python implementation of BNTRF with PSO-based hyperparameter adaptation.



\## 2. Dataset Description



The six datasets are collected from a real metropolitan area network (MAN) in a city in China. Due to privacy and security requirements, each terminal device is anonymized and encoded by an ID. Detailed device attributes, geographical sub-area labels, and wired/wireless access types are not provided.



Each record contains four fields:



```text

source\_device\_ID::destination\_device\_ID::time\_point::packet\_size\_value

```



One time point corresponds to a 10-minute interval. For each time point, a directed weighted network snapshot is constructed. The terminal devices are treated as nodes, and the packet size transmitted from one terminal device to another is treated as the directed interaction weight. The interaction weights have been normalized into the range `(0, 10)`.



By stacking the network snapshots along the time dimension, each dataset is represented as a third-order HDI tensor:



```text

Y ∈ R^{|I| × |J| × |K|}

```



where `|I|` and `|J|` denote the numbers of source and destination terminal devices, and `|K|` denotes the number of time points. In these datasets, the source-device and destination-device sets share the same anonymized node IDs, so the first two tensor modes have the same size.



The term \*\*dynamic\*\* means that the directed packet-transmission interaction weights among terminal devices vary across different 10-minute time points. It does not refer to physical node mobility.



\## 3. Dataset Statistics



| Dataset | Nodes | Time Points | Known Entries | Density |

|---|---:|---:|---:|---:|

| D1 | 352,680 | 1,080 | 1,697,654 | 1.26×10^-8 |

| D2 | 269,028 | 865 | 1,189,151 | 1.90×10^-8 |

| D3 | 208,634 | 748 | 720,459 | 2.21×10^-8 |

| D4 | 148,453 | 672 | 457,273 | 3.08×10^-8 |

| D5 | 73,281 | 556 | 48,277 | 1.61×10^-8 |

| D6 | 37,485 | 423 | 17,739 | 2.98×10^-8 |



The density is computed as:



```text

Density = Known Entries / (Nodes × Nodes × Time Points)

```



\## 4. Train/Validation/Test Splits



The fixed data splits used in the experiments are already released. Each dataset folder contains:



```text

D\*\_original.txt      full released dataset

D\*\_train\_70%.txt     training entries

D\*\_valid\_10%.txt     validation entries

D\*\_test\_20%.txt      test entries

```



The split ratio is:



```text

Training : Validation : Test = 70% : 10% : 20%

```



The training set is used to update model parameters. The validation set is used for PSO-based hyperparameter adaptation and early stopping. The test set is used only for final RMSE and MAE evaluation.



For D6, the validation file is currently named:



```text

D6\_Valid\_10%.txt

```



On case-sensitive operating systems, use this exact file name in the command line, or rename it to `D6\_valid\_10%.txt` for naming consistency.



\## 5. Preprocessing



The released files are already in the model-ready format. No additional preprocessing is required before running BNTRF.



The preprocessing applied before release is:



1\. anonymization of terminal devices into integer IDs;

2\. construction of records in the form `i::j::k::value`;

3\. normalization of packet-size interaction weights into `(0, 10)`;

4\. construction of fixed `70%/10%/20%` train/validation/test splits.



The code also provides an optional argument `--log10-value-plus-one`, but this option is disabled by default and is not required for the released normalized datasets.



\## 6. Environment



The implementation requires Python 3 and the packages listed in `BNTRF/requirements.txt`:



```text

numpy

numba

```



Install dependencies with:



```bash

cd BNTRF

python -m pip install -r requirements.txt

```



The computational kernels are implemented with Numba acceleration. The first run can be slower because Numba compiles the kernels. Subsequent runs are usually faster due to caching.



\## 7. Model and Hyperparameters



The implemented model contains two parts:



1\. a nonnegative tensor ring factorization term for modeling high-order source-destination-time interactions;

2\. a linear bias term for modeling source-specific, destination-specific, and time-specific fluctuations.



The prediction form is:



```text

ŷ\_ijk = TR\_term(i,j,k) + bias\_term(i,j,k)

```



The PSO algorithm adaptively searches two regularization hyperparameters:



```text

lambda\_reg  regularization coefficient for TR latent tensors

lambda\_b    regularization coefficient for bias factor matrices

```



Default settings:



| Parameter | Default value | Description |

|---|---:|---|

| rank | 5 | TR rank and bias rank |

| train\_round | 1000 | maximum training rounds |

| initscale | 0.25 | initialization scale for TR factors |

| initscale2 | 0.10 | initialization scale for bias factors |

| threshold | 2 | early stopping threshold |

| errorgap | 1e-5 | minimum improvement for early stopping |

| population | 5 | number of PSO particles |

| c1 | 2.0 | cognitive coefficient |

| c2 | 2.0 | social coefficient |

| w | 0.724 | inertia weight |

| alpha | 0.5 | RMSE/MAE balance in the validation fitness |

| min\_lambda\_reg | 0.01 | lower bound of `lambda\_reg` |

| max\_lambda\_reg | 0.10 | upper bound of `lambda\_reg` |

| min\_lambda\_b | 0.01 | lower bound of `lambda\_b` |

| max\_lambda\_b | 0.10 | upper bound of `lambda\_b` |

| velocity\_ratio | 0.2 | PSO velocity bound ratio |

| separator | `::` | file separator |



\## 8. Running BNTRF on One Dataset



From the repository root, run D1 as follows:



```bash

python BNTRF/run.py \\

&#x20; --train data/D1/D1\_train\_70%.txt \\

&#x20; --valid data/D1/D1\_valid\_10%.txt \\

&#x20; --test data/D1/D1\_test\_20%.txt \\

&#x20; --rank 5 \\

&#x20; --train-round 1000 \\

&#x20; --initscale 0.25 \\

&#x20; --initscale2 0.10 \\

&#x20; --threshold 2 \\

&#x20; --errorgap 1e-5 \\

&#x20; --population 5 \\

&#x20; --c1 2.0 \\

&#x20; --c2 2.0 \\

&#x20; --w 0.724 \\

&#x20; --alpha 0.5 \\

&#x20; --min-lambda-reg 0.01 \\

&#x20; --max-lambda-reg 0.10 \\

&#x20; --min-lambda-b 0.01 \\

&#x20; --max-lambda-b 0.10 \\

&#x20; --velocity-ratio 0.2

```



The program prints the elapsed time, stopping round, best PSO hyperparameters, and the test RMSE/MAE values at the selected rounds.



\## 9. Running All Six Datasets



Use the following commands from the repository root.



D1:



```bash

python BNTRF/run.py --train data/D1/D1\_train\_70%.txt --valid data/D1/D1\_valid\_10%.txt --test data/D1/D1\_test\_20%.txt

```



D2:



```bash

python BNTRF/run.py --train data/D2/D2\_train\_70%.txt --valid data/D2/D2\_valid\_10%.txt --test data/D2/D2\_test\_20%.txt

```



D3:



```bash

python BNTRF/run.py --train data/D3/D3\_train\_70%.txt --valid data/D3/D3\_valid\_10%.txt --test data/D3/D3\_test\_20%.txt

```



D4:



```bash

python BNTRF/run.py --train data/D4/D4\_train\_70%.txt --valid data/D4/D4\_valid\_10%.txt --test data/D4/D4\_test\_20%.txt

```



D5:



```bash

python BNTRF/run.py --train data/D5/D5\_train\_70%.txt --valid data/D5/D5\_valid\_10%.txt --test data/D5/D5\_test\_20%.txt

```



D6:



```bash

python BNTRF/run.py --train data/D6/D6\_train\_70%.txt --valid data/D6/D6\_Valid\_10%.txt --test data/D6/D6\_test\_20%.txt

```



For deterministic initialization, add fixed seeds:



```bash

\--seed 42 --pso-seed 42

```



\## 10. Main Command-Line Arguments



| Argument | Description |

|---|---|

| `--train` | path to the training file |

| `--valid` | path to the validation file |

| `--test` | path to the test file |

| `--separator` | separator used in data files, default `::` |

| `--rank` | TR rank and bias rank |

| `--train-round` | maximum number of training rounds |

| `--initscale` | initialization scale for TR factors |

| `--initscale2` | initialization scale for bias factors |

| `--threshold` | early stopping threshold |

| `--errorgap` | minimum validation improvement for early stopping |

| `--population` | PSO population size |

| `--c1` | PSO cognitive coefficient |

| `--c2` | PSO social coefficient |

| `--w` | PSO inertia weight |

| `--alpha` | validation fitness balance between RMSE and MAE |

| `--min-lambda-reg` | lower bound for `lambda\_reg` |

| `--max-lambda-reg` | upper bound for `lambda\_reg` |

| `--min-lambda-b` | lower bound for `lambda\_b` |

| `--max-lambda-b` | upper bound for `lambda\_b` |

| `--velocity-ratio` | velocity range ratio in PSO |

| `--seed` | random seed for model initialization |

| `--pso-seed` | random seed for PSO initialization |

| `--log10-value-plus-one` | optional log10(value+1) transformation |

| `--print-every` | printing interval |

| `--fixed-initial-fitness` | use fixed initial validation fitness instead of current validation fitness |



\## 11. Reproducibility Checklist



The repository provides the following materials for reproducibility:



\- all six datasets D1-D6;

\- the original released records for each dataset;

\- the fixed train/validation/test split files for each dataset;

\- the data format and preprocessing description;

\- the BNTRF implementation;

\- the PSO-based hyperparameter adaptation implementation;

\- default parameter settings;

\- running commands for each dataset.



Therefore, the reported experiments can be reproduced using the released split files and the command-line settings described above.



\## 12. Privacy Note



The datasets come from a real MAN system. For privacy and security protection, terminal devices are anonymized as integer IDs. The released data do not include device attributes, user information, geographical sub-area labels, or wired/wireless access labels.



