# antares-timeseries-generation

Timeseries generation library aiming at creating input data
for Antares simulator studies.

## Install

```bash
pip install antares-timeseries-generation
```
Necessity to say that pandas~=2.2.3 requires python version 3.9 or newer versions

## Usage

The generation requires to define a few input data in a `ThermalCluster` object: 

```python
import numpy as np

days = 365
generation_params = OutageGenerationParameters(
    unit_count=10,
    fo_law=ProbabilityLaw.UNIFORM,
    fo_volatility=0,
    po_law=ProbabilityLaw.UNIFORM,
    po_volatility=0,
    fo_duration=10 * np.ones(dtype=int, shape=days),
    fo_rate=0.2 * np.ones(dtype=float, shape=days),
    po_duration=10 * np.ones(dtype=int, shape=days),
    po_rate=np.zeros(dtype=float, shape=days),
    npo_min=np.zeros(dtype=int, shape=days),
    npo_max=10 * np.ones(dtype=int, shape=days)
)
cluster = ThermalCluster(
    outage_gen_params=generation_params,
    nominal_power=100,
    modulation=np.ones(dtype=float, shape=24),
)
```

You then need to provide a random number generator: we provide `MersenneTwisterRNG` 
to ensure the same generation as in `antares-solver` tool.
```python
rng = MersenneTwisterRNG()
```

Then perform the timeseries generation:

```python
generator = TimeSeriesGenerator(rng=rng, days=days)
results = generator.generate_time_series_for_clusters(cluster, 1)
```

The actual timeseries for the total available power of the cluster are available in
the results object as a numpy 2D-array:
```python
print(results.available_power)
```