# EV Battery Capacity Prediction

Estimating electric-vehicle battery capacity from ordinary charging behaviour, using
real-world fleet telemetry — with a full production MLOps pipeline.

---

## The problem

Battery capacity is the clearest measure of an EV battery's health, but it is
expensive to observe. Measuring it requires a charging session to pass through a
specific voltage window (3.77 V → 4.05 V) at a steady current, and then integrating
the charge that flows in.

Real drivers rarely cooperate. They plug in at 60% charge, unplug early to run
errands, use whatever current the station provides, or the vehicle throttles current
as the pack warms. None of these are faults — they are normal usage.

**In this dataset, 44% of charging sessions (279,380 of 629,121) carry no usable
capacity measurement for exactly this reason.**

But charging *behaviour* is recorded on every session regardless: how voltage rises,
how individual cells drift apart, how the pack heats. That behaviour reflects the
battery's condition whether or not the measurement protocol was satisfied.

### Research question

> **Can charging behaviour alone — with no odometer reading — recover what mileage
> tells us for free?**

This framing was not the starting point. It emerged from the EDA (see
[Finding 5](#finding-5--a-one-feature-linear-model-beats-the-published-benchmark)),
where a single-feature linear regression on mileage was found to outperform the
published deep-learning benchmark. Once that was established, "which architecture gets
a lower RMSE" was no longer an honest question to ask.

### Why the distinction matters operationally

Mileage tells you a battery is *statistically* likely to be worn. Charging behaviour
tells you what *this specific* battery's state is.

At 150,000 km the per-vehicle spread in capacity is 1.33 Ah — two vehicles with
identical odometer readings can genuinely sit at 37 Ah and 40 Ah. An odometer cannot
separate them.

A fleet operator does not need *"vehicles at 150k km average 38.7 Ah."* They need
*"vehicle 47 needs attention now."*

---

## Dataset

**EVBattery** — He, Wang, Sun et al. (2023), *Real-World Electric Vehicle Battery
Dataset*, NeurIPS 2023 Datasets & Benchmarks Track. Released under CC-BY-NC-SA.

Real-world charging telemetry collected from public charging stations across a
production EV fleet.

| | |
|---|---|
| Total snippet files | 629,121 |
| Usable (capacity-labelled) | **349,741** |
| Unusable (`capacity == 0`) | 279,380 (44%) |
| Vehicles | 100 |
| Snippet shape | (128, 8) |
| Sampling interval | 10 s → ~21 min per snippet |
| Snippets per charging session | ~10 (sliding window) |
| Target range | 26.64 – 46.17 Ah |

### Record structure

Each `.pkl` is a two-element tuple. Filenames are arbitrary integers and encode
nothing — all identifying information lives inside the file, which is why an index
must be built before any subset can be selected.

```python
(
    array(shape=(128, 8)),            # charging sensor readings
    OrderedDict({
        'capacity':       40.50,      # ← prediction target (Ah)
        'car':            43,         # ← split key
        'mileage':        153177.1,   # odometer (km)
        'charge_segment': '365',      # charging-session ID
        'label':          '00',       # fault flag (not used)
    })
)
```

### Sensor channels

| Col | Feature | Notes |
|---|---|---|
| 0 | Average cell voltage | Rises predictably with fill level (~3.0 V empty → ~4.2 V full) |
| 1 | Charging current | Negative sign = charging |
| 2 | State of charge (SOC) | Software-derived fill percentage |
| 3 | Max cell voltage | Fullest individual cell |
| 4 | Min cell voltage | Emptiest individual cell |
| 5 | Max cell temperature | Hottest point in the pack |
| 6 | Min cell temperature | Coolest point |
| 7 | Timestamp | Seconds elapsed within the snippet |

**Why max and min are separate channels.** An EV pack is hundreds of cells wired
together. As it ages they drift apart in voltage, and that spread — *cell imbalance* —
is a recognised degradation indicator. `max_V − min_V` and `max_T − min_T` are
therefore physically meaningful derived features.

---

## Reference paper

van den Hoven & Ranković (2026), *Data-Driven Battery Capacity Estimation in Electric
Vehicles: Insights from Large-Scale Real-World Data*, **Energy Systems**.
<https://link.springer.com/article/10.1007/s12667-025-00775-y>

The study benchmarks ARIMA(X), XGBoost, LSTM and TCN on this data. Best result:

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **LSTM** | **1.42** | **1.09** | **2.70%** |

Those figures are the benchmark this project measures against.

### What this project adds

**1. A Transformer.** Neither the reference paper nor the dataset authors' own
implementation (which covers XGBoost, RandomForest, MLP, LSTMNet and GatedCNN) tests
an attention-based model on this task.

**2. A mileage-only baseline.** Neither work reports one. It turns out to matter
considerably — see Finding 5.

**3. Per-mileage-band and per-vehicle evaluation**, motivated by the variance
structure found during EDA rather than applied by convention.

---

## Key findings from EDA

Full detail in [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb).

### Finding 1 — the target is tightly clustered

Mean 40.43 Ah, **standard deviation 1.86 Ah** — a coefficient of variation of ~4.6%.

If a model always predicts the mean, its RMSE *equals* the standard deviation, by
definition. So **1.86 is the floor**, and the reference paper's LSTM at 1.42 is only a
24% improvement on doing nothing.

**Consequence:** RMSE alone is a misleading metric here. R² and an explicit mean
baseline appear in every results table. (The dataset authors' implementation includes
a `MEAN` baseline, suggesting they hit the same issue.)

### Finding 2 — strong, clean degradation signal

Pearson **r = −0.795** between mileage and capacity.

| Mileage (k km) | Mean capacity | Std |
|---|---|---|
| 0 – 50 | 43.34 | 0.77 |
| 50 – 100 | 41.49 | 1.17 |
| 100 – 150 | 40.32 | 1.21 |
| 150 – 200 | 38.72 | 1.33 |
| 200 – 300 | 37.99 | 1.37 |

Monotonic decline, ~5.3 Ah (≈12%) lost across the fleet's tracked life. The task is
viable.

### Finding 3 — variance grows with mileage

The std column nearly doubles (0.77 → 1.37). New batteries are uniform; aged ones
diverge as usage patterns compound.

**This is where the model's value should appear.** At low mileage every vehicle sits
at 43.3 ± 0.77 and the odometer is already near-optimal. At high mileage the odometer
can only report a fleet average over a wide spread.

**Testable hypothesis:** if the sequence models beat the mileage baseline anywhere, it
should be on high-mileage vehicles.

### Finding 4 — per-vehicle trajectories are distinct

Each vehicle traces its own degradation curve with a different starting capacity and
slope. Combined with ~10 overlapping snippets per charging session, this makes a
random snippet-level split badly optimistic — a model could memorise vehicles rather
than learn degradation.

**Consequence:** split by vehicle. (This also matches the dataset authors' own fold
construction, which operates on car numbers.)

### Finding 5 — a one-feature linear model beats the published benchmark

| Model | Inputs | RMSE | R² |
|---|---|---|---|
| Mean baseline | none | 1.865 | 0.000 |
| **Linear regression** | **mileage only** | **1.132** | **0.632** |
| Reference paper's LSTM | 128×8 sequence | 1.420 | not reported |

*(Linear figure is in-sample; recomputed out-of-sample on the held-out vehicle split
in notebook 03.)*

This is what reframed the project. Reporting "the LSTM achieved 1.40" would be
presenting a loss to a trivial model as a success.

---

## Method

### Models

| Model | Inputs | Role |
|---|---|---|
| Mean predictor | none | Floor. Establishes what knowing nothing scores. |
| Linear regression | mileage only | The bar the sequence models must clear. |
| **LSTM** | 128×8 sequence | Reproduces the reference paper's best architecture. |
| **Transformer** | 128×8 sequence | This project's addition. Not tested by prior work. |

The sequence models receive **only** the 128×8 charging window. Mileage is deliberately
withheld — supplying it would let the model shortcut to "old vehicle → low capacity"
without learning anything from charging behaviour.

**LSTM** follows the reference architecture: 1 LSTM layer → 2 fully-connected layers,
hidden size 128, dropout 0.3, LeakyReLU, Xavier initialisation, Adam,
ReduceLROnPlateau.

**Transformer** uses `nn.TransformerEncoder` with positional encoding over the 128
timesteps.

### Protocol

- **Split by vehicle**, never randomly (Finding 4)
- **Normalisation fitted on the training split only**, after splitting
- **No outlier removal on the target** — the low tail toward 26.64 Ah represents
  genuinely degraded batteries, the cases that matter most operationally
- **Metrics:** RMSE, MAE, MAPE, R² — matching the reference paper's choices plus R²
- **Evaluation:** aggregate, per mileage band (Finding 3), and per vehicle (guards
  against a single vehicle distorting results, as the reference paper observed)

### Deliberate deviations from the dataset authors' reference implementation

| Their approach | This project | Reason |
|---|---|---|
| Normalisation fitted on a 200-snippet subsample | Fitted on the full training split | 200 of ~350k is noisy and may miss distribution tails |
| `all_car_dict` with absolute paths from their machine | Index rebuilt locally | Their shipped file is a placeholder |
| `.cuda()` hardcoded, Python 3.6 / PyTorch 1.5.1 / CUDA 10.2 | Modern PyTorch with Apple Silicon MPS | Their code is read as a reference, not executed |
| 5-fold cross-validation | Single held-out vehicle split | Scope decision for project timeline; noted as a limitation |

---

## MLOps pipeline

| Stage | Tool |
|---|---|
| Code versioning | Git / GitHub |
| Data versioning | DVC (remote: Azure Blob Storage) |
| Experiment tracking | MLflow via Dagshub |
| Containerisation | Docker (separate training and serving images) |
| Serving | FastAPI |
| CI/CD | GitHub Actions |
| Orchestration | Azure Kubernetes Service |
| Managed training / endpoint | Azure ML |
| Monitoring | Prometheus + Grafana |

Cloud provider is **Azure**, architecturally equivalent to an AWS EC2/SageMaker
workflow.

---

## Repository layout

```
ev-battery-capacity-prediction/
├── data/
│   ├── raw/                     # dataset (gitignored, DVC-tracked)
│   └── processed/               # index + derived tables
├── notebooks/
│   ├── 01_data_ingestion.ipynb  # structure, channels, index build
│   ├── 02_eda.ipynb             # distribution, degradation, baselines
│   └── 03_split_and_baselines.ipynb
├── src/                         # refactored .py for DVC / CI
├── reports/figures/
├── docs/EDA_FINDINGS.md
├── requirements.txt
└── README.md
```

### Generated artefacts

| File | Produced by | Contents |
|---|---|---|
| `data/processed/dataset_index.json` | 01 | car → snippet file paths, per-car metadata |
| `data/processed/car_summary.csv` | 01 | per-vehicle summary |
| `data/processed/snippet_meta.csv` | 02 | per-snippet car / mileage / capacity |
| `reports/figures/*.png` | 02 | distribution and degradation plots |

---

## Setup

```bash
git clone https://github.com/<user>/ev-battery-capacity-prediction.git
cd ev-battery-capacity-prediction

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download the EVBattery dataset and extract into `data/raw/`, then run the notebooks in
order. Notebook 01 performs a one-time ~2–3 minute scan; subsequent notebooks load the
saved index in milliseconds.

**Apple Silicon:** PyTorch uses `device = "mps"` for GPU acceleration.

---

## Results

*(populated as models are trained)*

| Model | Inputs | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|---|
| Mean baseline | none | — | — | — | 0.000 |
| Linear regression | mileage | — | — | — | — |
| LSTM | 128×8 | — | — | — | — |
| Transformer | 128×8 | — | — | — | — |
| *Reference paper LSTM* | *128×8* | *1.42* | *1.09* | *2.70%* | *—* |

---

## Limitations

- **Single train/test split** rather than the reference paper's 5-fold CV. A scope
  decision; per-vehicle metrics partially mitigate the resulting variance.
- **Snippet overlap.** ~10 snippets per charging session means the effective sample
  size is smaller than the raw count. Splitting by vehicle prevents leakage but does
  not eliminate within-vehicle correlation.
- **Sparse high-capacity labels.** The reference paper noted all models struggled to
  predict high capacity values for this reason; the same limitation applies here.
- **Mileage as a wear proxy.** Correlated at r = −0.795, but it does not capture
  charging habits, climate exposure or driving style — which is precisely the gap the
  sequence models are being tested against.

---

## References

1. He, Wang, Sun et al. (2023). *Real-World Electric Vehicle Battery Dataset*.
   NeurIPS 2023 Datasets & Benchmarks Track. CC-BY-NC-SA.
2. van den Hoven & Ranković (2026). *Data-Driven Battery Capacity Estimation in
   Electric Vehicles: Insights from Large-Scale Real-World Data*. Energy Systems.
   <https://link.springer.com/article/10.1007/s12667-025-00775-y>
