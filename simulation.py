import pandas as pd
import numpy as np
import re
from datetime import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import Model, GRB
from sklearn.neighbors import BallTree
import gurobipy as gp

# --- PARAMETERS YOU CAN TUNE ---
AVERAGE_LOS_DAYS = 7.0        # assumed average length of stay
DISTANCE_THRESHOLDS = [0, 50]  # in miles

# HHS Region mapping
HHS_REGIONS = {
    'CT': 1, 'ME': 1, 'MA': 1, 'NH': 1, 'RI': 1, 'VT': 1,
    'NJ': 2, 'NY': 2, 'PR': 2, 'VI': 2,
    'DE': 3, 'DC': 3, 'MD': 3, 'PA': 3, 'VA': 3, 'WV': 3,
    'AL': 4, 'FL': 4, 'GA': 4, 'KY': 4, 'MS': 4, 'NC': 4, 'SC': 4, 'TN': 4,
    'IL': 5, 'IN': 5, 'MI': 5, 'MN': 5, 'OH': 5, 'WI': 5,
    'AR': 6, 'LA': 6, 'NM': 6, 'OK': 6, 'TX': 6,
    'IA': 7, 'KS': 7, 'MO': 7, 'NE': 7,
    'CO': 8, 'MT': 8, 'ND': 8, 'SD': 8, 'UT': 8, 'WY': 8,
    'AZ': 9, 'CA': 9, 'HI': 9, 'NV': 9, 'AS': 9, 'GU': 9, 'MP': 9,
    'AK': 10, 'ID': 10, 'OR': 10, 'WA': 10
}

# --- 1. LOAD & CLEAN DATA ---
df = pd.read_csv(
    r"C:\Users\shazn\Downloads\COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_--_RAW_20250728.csv",
    parse_dates=["collection_week"]
)

# Replace missing values with NaN
df.replace(-999999, np.nan, inplace=True)

# --- 2. PARSE GEOCODED POINTS TO LAT/LON ---
def parse_point(point):
    """Parse "POINT (lon lat)" into (lat, lon)."""
    if isinstance(point, str) and point.startswith("POINT"):
        lon, lat = map(float, re.findall(r"[-\d\.]+", point))
        return lat, lon
    return np.nan, np.nan

df[["lat", "lon"]] = df["geocoded_hospital_address"].apply(lambda p: pd.Series(parse_point(p)))

# Add state and HHS region mapping
df['state'] = df['state'].str.upper()
df['hhs_region'] = df['state'].map(HHS_REGIONS)

# --- 3. FILTER TO STUDY PERIOD ---
start_date = pd.to_datetime("2020-10-01")
end_date = pd.to_datetime("2021-03-31")
mask = (df["collection_week"] >= start_date) & (df["collection_week"] <= end_date)
df_period = df.loc[mask].copy()
df_period = df_period[df_period["hospital_pk"].str.fullmatch(r"\d{6}")]

weeks_in_period = df_period["collection_week"].nunique()

# Count how many weeks each hospital has non-null data on the target column
hospital_reporting_counts = (
    df_period
    .dropna(subset=["inpatient_beds_used_7_day_avg","all_adult_hospital_inpatient_beds_7_day_avg","inpatient_beds_used_7_day_avg"])
    .groupby("hospital_pk")["collection_week"]
    .nunique()
)

# Keep only hospitals that reported in ALL weeks
valid_hospitals = hospital_reporting_counts[hospital_reporting_counts == weeks_in_period].index
df_period = df_period[df_period["hospital_pk"].isin(valid_hospitals)]

# --- 4. COMPUTE OCCUPANCY RATIOS & NEAR-CAPACITY FLAG ---
df_period["capacity_ratio"] = (
    df_period["inpatient_beds_used_7_day_avg"] /
    df_period["all_adult_hospital_inpatient_beds_7_day_avg"]
)
df_period["over90"] = df_period["capacity_ratio"] > 0.9
overfull_hospitals = (
    df_period.loc[df_period["capacity_ratio"] > 1.0, "hospital_pk"]
    .unique()
)

# Keep only hospitals that never exceeded 100% occupancy
df_period = df_period[~df_period["hospital_pk"].isin(overfull_hospitals)]

min_beds_per_hosp = (
    df_period
    .groupby("hospital_pk")["all_adult_hospital_inpatient_beds_7_day_avg"]
    .min()
)

too_small = min_beds_per_hosp[min_beds_per_hosp < 20].index
df_period = df_period[~df_period["hospital_pk"].isin(too_small)]

# --- 5. WEEKLY BASELINE STATS (NO TRANSFERS) ---
weekly_baseline = (
    df_period
    .groupby("collection_week")
    .agg(
        n_hospitals=("hospital_pk", "nunique"),
        n_near=("over90", "sum")
    )
    .assign(pct_near=lambda d: d["n_near"] / d["n_hospitals"])
    .reset_index()
)

# Regional baseline stats
regional_baseline = (
    df_period
    .groupby(["collection_week", "hhs_region"])
    .agg(
        n_hospitals=("hospital_pk", "nunique"),
        n_near=("over90", "sum")
    )
    .assign(pct_near=lambda d: d["n_near"] / d["n_hospitals"])
    .reset_index()
)

# Pre-compute unique hospital locations
hosp_locs = (
    df_period[["hospital_pk", "lat", "lon", "hhs_region"]]
    .dropna(subset=["lat", "lon"])
    .drop_duplicates("hospital_pk")
    .set_index("hospital_pk")
)
coords_rad = np.radians(hosp_locs[["lat", "lon"]].values)
hosp_ids = hosp_locs.index.to_numpy()

# Build spatial tree
tree = BallTree(coords_rad, metric="haversine")
R_EARTH_MI = 3958.8
radius_rad = 200.0 / R_EARTH_MI
indices, dists = tree.query_radius(coords_rad, r=radius_rad, return_distance=True, sort_results=True)

# Pack into sparse dict: (i,j) → distance_mi
dist_within = {}
for idx_i, neigh in enumerate(indices):
    i = hosp_ids[idx_i]
    for idx_j, dist_rad in zip(neigh, dists[idx_i]):
        if idx_i == idx_j:
            continue
        j = hosp_ids[idx_j]
        dist_mi = dist_rad * R_EARTH_MI
        dist_within[(i, j)] = dist_mi

def weekly_departures(occ_t, los_days=AVERAGE_LOS_DAYS):
    """Expected departures in one week under an exponential LOS."""
    rate = 1.0 - np.exp(-7.0 / los_days)
    return occ_t * rate

def optimise_one_week_robust(occ_t, cap_t1, cap_constraint, arrivals, departs, dist_mat, max_miles):
    """
    Robust optimization that ensures feasibility by using slack variables.
    
    Returns:
        occ_t1          – end‑of‑week occupancies after optimal re‑allocation
        near_capacity   – 1/0 flag per hospital (occ ≥ 0.9*cap)
        transfers_count – total patients moved
        regional_stats  – detailed regional transfer statistics
    """
    demand_hosp = arrivals[arrivals > 0].index.tolist()
    all_hosp = occ_t.index.tolist()
    
    # Calculate available capacity more conservatively
    current_freed = np.maximum(0, cap_t1 - occ_t + departs)
    supply_hosp = current_freed[current_freed > 0.1].index.tolist()  # At least 0.1 bed available
    
    if max_miles == 0 or not demand_hosp or not supply_hosp:
        # No transfers case
        occ_t1 = np.minimum(occ_t - departs + arrivals, cap_t1)
        near_cap = (occ_t1 > 0.9 * cap_t1).astype(int)
        regional_stats = calculate_regional_stats(occ_t1, near_cap, {}, hosp_locs)
        return occ_t1, near_cap, 0, arrivals, regional_stats
    
    # Gurobi model with slack variables for robustness
    m = Model("hospital_transfers")
    m.setParam("OutputFlag", 1)  # Suppress output
    m.setParam("TimeLimit", 30)  # 30 second time limit
    
    # Decision variables
    f = {}  # flow variables f[i,j]: patients from i to j
    for i in demand_hosp:
        for j in supply_hosp:
            if (i, j) in dist_mat and dist_mat[(i, j)] <= max_miles:
                f[i, j] = m.addVar(lb=0.0, ub=min(arrivals[i], current_freed[j]),
                                   vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}")
    
    # Slack variables for unmet demand (patients that can't be placed)
    unmet = {i: m.addVar(lb=0.0, ub=arrivals[i], vtype=GRB.CONTINUOUS, 
                        name=f"unmet_{i}") for i in demand_hosp}
    
    # Binary flags for near-capacity hospitals
    y = {h: m.addVar(vtype=GRB.BINARY, name=f"y_{h}") for h in all_hosp}
    
    m.update()

    # 1. Physical capacity constraints with slack
    for h in all_hosp:
        in_flow = gp.quicksum(f.get((i, h), 0) for i in demand_hosp)
        out_flow = gp.quicksum(f.get((h, j), 0) for j in supply_hosp)
        m.addConstr(out_flow <= arrivals[h], name=f"demand_{h}")
        
        # Add unmet demand if this hospital has arrivals
        unmet_local = unmet.get(h, 0)
        
        const_part = occ_t[h] - departs[h] + arrivals[h]
        occ_end = const_part + in_flow - out_flow - unmet_local
        
        m.addConstr(occ_end <= cap_constraint[h], name=f"physcap_{h}")
        
        # Near-capacity constraint
        BIGM = cap_t1.max() * 2
        m.addConstr(occ_end <= 0.9 * cap_t1[h] + BIGM * y[h], name=f"nearcap_{h}")
    
    # Multi-objective optimization
    m.ModelSense = GRB.MINIMIZE

    # Primary: minimize unmet demand
    m.setObjectiveN(gp.quicksum(unmet.values()), index=0, priority=2, name="MinUnmet")
    
    # Secondary: minimize near-capacity hospitals
    m.setObjectiveN(gp.quicksum(y.values()), index=1, priority=1, name="MinNearCapacity")
    
    # Tertiary: minimize transfers
    m.setObjectiveN(gp.quicksum(f.values()), index=2, priority=0, name="MinTransfers")
    
    m.optimize()
    
    if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        # Extract solution
        transfer_flows = {}
        total_transfers = 0
        
        for (i, j), var in f.items():
            if var.X > 0.01:  # Small threshold to avoid numerical issues
                transfer_flows[(i, j)] = var.X
                total_transfers += var.X
        
        # Calculate final occupancies
        net_flow = pd.Series(0.0, index=all_hosp)
        for (i, j), flow in transfer_flows.items():
            net_flow[j] += flow
            net_flow[i] -= flow
        
        # change arrivals to match unmet demand
        total_unmet = 0
        for i, var in unmet.items():
            if var.X > 0.01:
                arrivals[i] -= var.X
                total_unmet += var.X
        print(f"Total unmet: {total_unmet}")
        
        occ_t1 = np.minimum(occ_t - departs + net_flow + arrivals, cap_t1)
        near_cap = (occ_t1 > 0.9 * cap_t1).astype(int)
        
        regional_stats = calculate_regional_stats(occ_t1, near_cap, transfer_flows, hosp_locs)
        
        return occ_t1, near_cap, total_transfers, arrivals, regional_stats
    
    else:
        # Fallback: no transfers
        occ_t1 = np.minimum(occ_t - departs + arrivals, cap_t1)
        near_cap = (occ_t1 > 0.9 * cap_t1).astype(int)
        regional_stats = calculate_regional_stats(occ_t1, near_cap, {}, hosp_locs)
        return occ_t1, near_cap, 0, arrivals, regional_stats

def calculate_regional_stats(occ_t1, near_cap, transfer_flows, hosp_locs):
    """Calculate regional statistics for transfers and capacity changes."""
    regional_stats = {}
    
    # Initialize regional counters
    for region in range(1, 11):
        regional_stats[region] = {
            'inbound_transfers': 0,
            'outbound_transfers': 0,
            'hospitals_in_surge': 0,
            'total_hospitals': 0
        }
    
    # Count transfers by region
    for (i, j), flow in transfer_flows.items():
        if i in hosp_locs.index and j in hosp_locs.index:
            region_i = hosp_locs.loc[i, 'hhs_region']
            region_j = hosp_locs.loc[j, 'hhs_region']
            
            if pd.notna(region_i) and pd.notna(region_j):
                regional_stats[region_i]['outbound_transfers'] += flow
                regional_stats[region_j]['inbound_transfers'] += flow
    
    # Count hospitals in surge by region
    for hosp in near_cap.index:
        if hosp in hosp_locs.index:
            region = hosp_locs.loc[hosp, 'hhs_region']
            if pd.notna(region):
                regional_stats[region]['hospitals_in_surge'] += near_cap[hosp]
                regional_stats[region]['total_hospitals'] += 1
    
    return regional_stats

# ------------------ MAIN DYNAMIC LOOP ------------------
all_results = []  # Enhanced results collection
regional_results = []  # Regional-level results

weeks = sorted(df_period["collection_week"].unique())

for max_miles in DISTANCE_THRESHOLDS:
    print(f"Processing transfer radius: {max_miles} miles")
    
    occ_start = (
        df_period[df_period["collection_week"] == weeks[0]]
        .set_index("hospital_pk")["inpatient_beds_used_7_day_avg"]
        .fillna(0)
    )
    hosp_index = occ_start.index
    occ_cur = occ_start.copy()
    
    # Track baseline surge status for comparison
    baseline_surge = {}
    
    for t in range(len(weeks) - 1):
        wk_t = weeks[t]
        wk_t1 = weeks[t + 1]
        df_t = df_period[df_period["collection_week"] == wk_t]
        df_t1 = df_period[df_period["collection_week"] == wk_t1]
        
        cap_true = (
            df_t.set_index("hospital_pk")["all_adult_hospital_inpatient_beds_7_day_avg"]
            .reindex(hosp_index, fill_value=0)
        )
        cap_true1 = (
            df_t1.set_index("hospital_pk")["all_adult_hospital_inpatient_beds_7_day_avg"]
            .reindex(hosp_index, fill_value=0)
        )
        occ_true = (
            df_t.set_index("hospital_pk")["inpatient_beds_used_7_day_avg"]
            .reindex(hosp_index, fill_value=0)
        )
        
        # Store baseline surge status
        baseline_surge[wk_t1] = (occ_true > 0.9 * cap_true).astype(int)
        
        # Calculate arrivals
        occ_next_true = (
            df_t1.set_index("hospital_pk")["inpatient_beds_used_7_day_avg"]
            .reindex(hosp_index, fill_value=0)
        )
        cap_constraint = np.maximum(cap_true1, occ_next_true)
        departs = weekly_departures(occ_true)
        arr_implied = occ_next_true - occ_true + departs
        
        arr_pos = arr_implied.clip(lower=0.0)
        extra_dep = (-arr_implied).clip(lower=0.0)
        departs_cor = departs + extra_dep
        
        # Optimize with robust method
        occ_next, near_flag, xfers, arr_true, regional_stats = optimise_one_week_robust(
            occ_cur, cap_true1, cap_constraint, arr_pos, departs_cor, dist_within, max_miles
        )
        n_near_inbounds = arr_true[near_flag == 1].sum()
        # Calculate hospitals and total inbounds in hospitals from surge to non-surge
        surge_to_nonsurge_net = 0
        surge_to_nonsurge_inbounds_net = 0
        if max_miles > 0 and wk_t1 in baseline_surge:
            baseline = baseline_surge[wk_t1]
            current = near_flag
            surge_to_nonsurge_net = ((baseline == 1) & (current == 0)).sum() - ((baseline == 0) & (current == 1)).sum()
            surge_to_nonsurge_inbounds_net = arr_true[(baseline == 1) & (current == 0)].sum() - arr_true[(baseline == 0) & (current == 1)].sum()
        
        # Store overall results
        all_results.append({
            "collection_week": wk_t1,
            "threshold": max_miles,
            "n_near": near_flag.sum(),
            "n_near_inbounds": n_near_inbounds,
            "transfers": xfers,
            "surge_to_nonsurge": surge_to_nonsurge_net,
            "surge_to_nonsurge_inbounds": surge_to_nonsurge_inbounds_net,
            "total_hospitals": len(near_flag)
        })
        
        # Store regional results
        for region, stats in regional_stats.items():
            regional_results.append({
                "collection_week": wk_t1,
                "threshold": max_miles,
                "hhs_region": region,
                "inbound_transfers": stats['inbound_transfers'],
                "outbound_transfers": stats['outbound_transfers'],
                "hospitals_in_surge": stats['hospitals_in_surge'],
                "total_hospitals": stats['total_hospitals']
            })
        
        occ_cur = occ_next.copy()

# Create final DataFrames
opt_results = pd.DataFrame(all_results)
regional_opt_results = pd.DataFrame(regional_results)

# --- 6. VISUALIZATION ---
print("Creating visualization...")

# Calculate average hospitals in surge by region and condition
regional_summary = (
    regional_opt_results
    .groupby(['hhs_region', 'threshold'])
    .agg({
        'hospitals_in_surge': 'mean',
        'total_hospitals': 'mean'
    })
    .reset_index()
)

regional_summary['pct_in_surge'] = (
    regional_summary['hospitals_in_surge'] / 
    regional_summary['total_hospitals'] * 100
)

# Create the plot
plt.figure(figsize=(15, 10))

# Pivot for easier plotting, only plot 0 and 50 miles
plot_data = regional_summary[regional_summary['threshold'].isin([0, 50])].pivot(
    index='hhs_region', 
    columns='threshold', 
    values='hospitals_in_surge'
)

# Create the standalone clustered bar plot
ax = plot_data.plot(kind='bar', width=0.8, color=['#2E86AB', '#A23B72'], alpha=0.8)

# Formatting
plt.title('Hospitals in Surge by HHS Region', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('HHS Region', fontsize=14, fontweight='bold')
plt.ylabel('# of Hospitals in Surge', fontsize=14, fontweight='bold')

# Customize legend
plt.legend(title='Transfer Radius (miles)', 
          labels=['No Transfers', '50 Miles'],
          title_fontsize=12, fontsize=11, loc='upper right')

# Rotate x-axis labels to be horizontal and add grid
plt.tick_params(axis='x', rotation=0, labelsize=11)
plt.tick_params(axis='y', labelsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=9, rotation=0)

# Improve layout and show
plt.tight_layout()
plt.show()





# Print summary statistics
print("\n=== SIMULATION SUMMARY ===")
print(f"Total weeks simulated: {len(weeks)-1}")
print(f"Total hospitals: {len(hosp_index)}")

for threshold in DISTANCE_THRESHOLDS:
    subset = opt_results[opt_results['threshold'] == threshold]
    avg_surge = subset['n_near'].mean()
    total_transfers = subset['transfers'].sum()
    label = f"{threshold} miles" if threshold > 0 else "No transfers"
    print(f"\n{label}:")
    print(f"  Average hospitals in surge: {avg_surge:.1f}")
    print(f"  Total transfers: {total_transfers:.0f}")
    
    if threshold > 0:
        reduction = opt_results[opt_results['threshold'] == 0]['n_near'].mean() - avg_surge
        print(f"  Reduction in surge hospitals: {reduction:.1f}")

print("\n=== REGIONAL SUMMARY ===")
for region in range(1, 11):
    region_data = regional_summary[regional_summary['hhs_region'] == region]
    if not region_data.empty:
        print(f"\nHHS Region {region}:")
        for _, row in region_data.iterrows():
            threshold = row['threshold']
            label = f"{threshold} miles" if threshold > 0 else "No transfers"
            print(f"  {label}: {row['hospitals_in_surge']:.1f} hospitals ({row['pct_in_surge']:.1f}%)") 
