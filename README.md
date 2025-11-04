# Hospital Load Balancing Simulation

This repository contains a single script, `simulation.py`, that runs a weekly counterfactual simulation of proactive patient transfers to reduce hospital overload during the 2020 Q4 to 2021 Q1 COVID surge. The model infers weekly arrivals from observed occupancy, then uses an optimization step to redirect a portion of those arrivals to nearby hospitals with spare capacity and quantifies the reduction in near‑capacity hospitals.

## What the script does

1. Loads the HHS facility‑level capacity panel and keeps hospitals that report consistently over the study window.
2. Parses geocoded points, maps states to HHS regions, and filters out facilities with fewer than 20 adult inpatient beds or with impossible occupancy above 100 percent.
3. Computes weekly baseline surge counts with no transfers.
4. Builds a spatial index and a hospital‑to‑hospital distance map.
5. For each week and for each transfer radius in `DISTANCE_THRESHOLDS` (default 0 and 50 miles), solves a robust multi‑objective allocation that
   - minimizes unmet demand,
   - then minimizes the number of near‑capacity hospitals,
   - then minimizes total transfers.
6. Produces a regional bar chart comparing the average number of hospitals in surge under the baseline and the 50‑mile policy and prints a textual summary.

## Data

Download the input data from HealthData.gov:

https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u/about_data

Save the CSV locally and update the path near the top of `simulation.py`

```python
df = pd.read_csv(
    r"COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_--_RAW_20250728.csv",
    parse_dates=["collection_week"]
)
```

Tips

- To preserve leading zeros in `hospital_pk`, read it as a string by adding `dtype={'hospital_pk': str}` to `read_csv`.
- The script expects the 7‑day average fields used in the public file schema.

## Quick start

1. Python 3.10 or newer.
2. Install dependencies.

```
pip install pandas numpy scikit-learn matplotlib seaborn gurobipy
```

3. Ensure you have a working Gurobi installation and license for the optimization step. If you only want the baseline with no transfers you can set `DISTANCE_THRESHOLDS = [0]` and the script will run without Gurobi.

4. Run the script

```
python simulation.py
```

The script prints a summary block and opens a figure window. In notebooks the plot will display inline.

## Parameters you can tune

At the top of the file

```python
AVERAGE_LOS_DAYS = 7.0         # mean length of stay used to estimate weekly departures
DISTANCE_THRESHOLDS = [0, 50]  # transfer radii in miles to simulate
```

Inside the script

- Study window is set to 2020‑10‑01 through 2021‑03‑31.
- Near‑capacity definition is fixed at 90 percent of staffed adult inpatient beds.
- Minimum adult inpatient beds per facility is 20.
- Distance precomputation uses a 200‑mile neighborhood for speed, while the active policy radius is enforced by `DISTANCE_THRESHOLDS` during optimization.

## How it works

Inflow inference

- Weekly departures are modeled with an exponential length‑of‑stay: `depart = occ_t * (1 − exp(−7/LOS))`.
- Implied arrivals ensure the identity across weeks: `arrivals = occ_{t+1} − occ_t + depart`.
- Negative implied arrivals are interpreted as additional departures for feasibility.

Optimization

- Decision variables are patient flows `f_{i→j}` from origin `i` to destination `j` and binary surge flags `y_j`.
- Constraints enforce conservation of admissions, physical capacity at destinations, and a policy radius.
- A near‑capacity big‑M constraint activates `y_j` if occupancy exceeds 90 percent of capacity.
- Multi‑objective priorities minimize unmet demand, then the number of near‑capacity hospitals, then total transfers.
- A 30‑second time limit is set per weekly solve.

Outputs

- Text summary with average surge hospitals, total transfers, and net surge‑to‑non‑surge transitions.
- Bar chart of the average number of hospitals in surge by HHS region for 0 and 50 miles.
- Two in‑memory DataFrames
  - `opt_results`: per‑week totals by radius
  - `regional_opt_results`: per‑week regional transfer and surge statistics

To persist results add at the end of the script

```python
opt_results.to_csv("opt_results.csv", index=False)
regional_opt_results.to_csv("regional_opt_results.csv", index=False)
plt.savefig("hospitals_in_surge_by_region.png", dpi=200, bbox_inches="tight")
```

## Troubleshooting

- Gurobi output. The script currently sets `m.setParam("OutputFlag", 1)` with a comment that says suppress output. To silence the solver set this flag to `0`.
- Missing geocodes. Rows without lat‑lon are excluded from the spatial graph. If too many hospitals drop, check the `geocoded_hospital_address` field.
- Infeasibility. The model uses slack variables for unmet demand plus a conservative capacity constraint to maintain feasibility, and falls back to no‑transfer updates if the solver cannot return a solution within the time limit.
- Performance. The BallTree restricts candidate pairs. If memory is tight lower the 200‑mile precomputation radius or subset by region.

## Reproducibility notes

- No random seeds are used. Given the same CSV and solver settings the outputs are deterministic up to solver tolerances.
- If you change the study window, capacity thresholds, or completeness rule your counts will change.

## Citation

If you use the data please cite HealthData.gov as the source of the HHS facility‑level capacity dataset.
