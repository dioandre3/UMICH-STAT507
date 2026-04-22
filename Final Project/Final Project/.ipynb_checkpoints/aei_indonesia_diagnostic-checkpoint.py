"""
AEI Indonesia Diagnostic
========================
Downloads the September 2025 AEI geographic release and prints exactly
three things needed to decide what Indonesia analysis is feasible:

1. How many Indonesia rows exist
2. Which facets Indonesia appears in
3. Which Indonesian tasks/clusters clear the privacy threshold

Run: python aei_indonesia_diagnostic.py

Requires: pip install pandas huggingface_hub
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("Missing dependencies. Run: pip install pandas huggingface_hub")
    sys.exit(1)

REPO_ID = "Anthropic/EconomicIndex"
RELEASE = "release_2025_09_15"
LOCAL_DIR = Path("./aei_data")
LOCAL_DIR.mkdir(exist_ok=True)

# Indonesia identifiers - data may use ISO-2 (raw) or ISO-3 (enriched)
INDONESIA_CODES = ["ID", "IDN", "Indonesia"]

# ============================================================
# STEP 1: Discover what files exist in the September 2025 release
# ============================================================
print("=" * 70)
print("STEP 1: Listing files in the September 2025 release")
print("=" * 70)

try:
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    release_files = sorted([f for f in all_files if RELEASE in f])
    print(f"\nFound {len(release_files)} files in {RELEASE}/:\n")
    for f in release_files:
        print(f"  {f}")
except Exception as e:
    print(f"Could not list files: {e}")
    print("Will try direct downloads with known filenames.")
    release_files = []

# ============================================================
# STEP 2: Download the main Claude.ai geographic file
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Downloading Claude.ai geographic data")
print("=" * 70)

# Find the raw Claude.ai CSV (filename may vary slightly)
candidates = [f for f in release_files if "claude_ai" in f and f.endswith(".csv")]
if not candidates:
    # Fallback to known filename from documentation
    candidates = [f"{RELEASE}/aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv"]

geo_file_path = None
for candidate in candidates:
    try:
        print(f"\nDownloading: {candidate}")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=candidate,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
        )
        geo_file_path = path
        print(f"  -> Saved to: {path}")
        break
    except Exception as e:
        print(f"  Failed: {e}")

if not geo_file_path:
    print("\nERROR: Could not download geographic data. Exiting.")
    sys.exit(1)

# Also try to grab the documentation
try:
    doc_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{RELEASE}/data_documentation.md",
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    print(f"  Documentation saved to: {doc_path}")
except Exception:
    print("  (data_documentation.md not retrieved — non-critical)")

# ============================================================
# STEP 3: Load and inspect schema
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Inspecting data schema")
print("=" * 70)

geo = pd.read_csv(geo_file_path)
print(f"\nTotal rows: {len(geo):,}")
print(f"Total columns: {len(geo.columns)}")
print(f"\nColumns and dtypes:")
for col in geo.columns:
    n_unique = geo[col].nunique()
    sample = geo[col].dropna().iloc[0] if geo[col].notna().any() else "N/A"
    print(f"  {col:30s}  dtype={str(geo[col].dtype):10s}  unique={n_unique:>7,}  sample={str(sample)[:50]}")

# ============================================================
# DIAGNOSTIC 1: How many Indonesia rows?
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 1: Indonesia row count")
print("=" * 70)

# Find which column holds country codes
geo_col_candidates = ["geo_id", "country", "country_code", "geography", "iso_code"]
geo_col = None
for c in geo_col_candidates:
    if c in geo.columns:
        geo_col = c
        break

if geo_col is None:
    # Inspect string columns for ISO codes
    print("\nNo standard geo column found. Searching for ISO codes...")
    for col in geo.select_dtypes(include="object").columns:
        sample_vals = geo[col].dropna().unique()[:5]
        if any(v in ["US", "USA", "ID", "IDN", "GB", "DE"] for v in sample_vals):
            geo_col = col
            print(f"  Found country codes in column: {col}")
            break

if geo_col is None:
    print("ERROR: Could not identify country column. Inspect schema above.")
    sys.exit(1)

print(f"\nUsing column: '{geo_col}'")

id_mask = geo[geo_col].isin(INDONESIA_CODES)
id_rows = geo[id_mask]

print(f"\nIndonesia rows: {len(id_rows):,}")
print(f"Indonesia codes found: {id_rows[geo_col].unique().tolist()}")
print(f"Total unique countries in data: {geo[geo_col].nunique()}")

# Compare to a high-AUI country and a peer
benchmarks = {"US/USA": ["US", "USA"], "India": ["IN", "IND"], "Israel": ["IL", "ISR"]}
print(f"\nFor comparison:")
for label, codes in benchmarks.items():
    n = (geo[geo_col].isin(codes)).sum()
    print(f"  {label:12s}: {n:,} rows")

# ============================================================
# DIAGNOSTIC 2: Which facets does Indonesia appear in?
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Facets Indonesia appears in")
print("=" * 70)

facet_col = None
for c in ["facet", "facet_name", "dimension", "metric_type"]:
    if c in geo.columns:
        facet_col = c
        break

if facet_col:
    print(f"\nUsing facet column: '{facet_col}'\n")
    id_facets = id_rows[facet_col].value_counts()
    print(f"Indonesia row counts by facet:")
    for facet, n in id_facets.items():
        print(f"  {str(facet):40s} {n:>6,} rows")
    
    # Also show what facets exist globally
    print(f"\nAll facets in dataset (for reference):")
    all_facets = geo[facet_col].value_counts()
    for facet, n in all_facets.head(15).items():
        id_n = id_facets.get(facet, 0)
        marker = "✓" if id_n > 0 else "✗"
        print(f"  {marker} {str(facet):40s} {n:>8,} global  | {id_n:>5,} Indonesia")
else:
    print("\nNo 'facet' column found.")
    print("Inspect schema above to identify the dimension column.")

# ============================================================
# DIAGNOSTIC 3: What tasks/clusters cleared the threshold for Indonesia?
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Indonesia's visible tasks/clusters")
print("=" * 70)

cluster_col = None
for c in ["cluster_name", "cluster", "task", "category", "value"]:
    if c in geo.columns:
        cluster_col = c
        break

if cluster_col and facet_col:
    # Count unique clusters per facet for Indonesia
    print(f"\nUnique cluster values per facet (Indonesia only):\n")
    for facet in id_rows[facet_col].unique():
        subset = id_rows[id_rows[facet_col] == facet]
        n_clusters = subset[cluster_col].nunique()
        print(f"  {str(facet):40s} {n_clusters:>4} unique values")
    
    # Show actual task-level Indonesia data if present
    task_facets = [f for f in id_rows[facet_col].unique()
                   if any(k in str(f).lower() for k in ["task", "onet", "request"])]
    
    if task_facets:
        primary_task_facet = task_facets[0]
        task_rows = id_rows[id_rows[facet_col] == primary_task_facet].copy()
        
        print(f"\nTop 20 Indonesian tasks/requests in facet '{primary_task_facet}':")
        sort_col = "usage_pct" if "usage_pct" in task_rows.columns else "count"
        if sort_col in task_rows.columns:
            top = task_rows.nlargest(20, sort_col)[[cluster_col, sort_col]]
            for _, r in top.iterrows():
                val = r[sort_col]
                val_str = f"{val:.3f}" if val < 1 else f"{val:,.0f}"
                print(f"  {val_str:>10s}  {str(r[cluster_col])[:80]}")
        else:
            print(f"  (no usage_pct or count column to sort by)")
            print(task_rows[[cluster_col]].head(20).to_string())

# ============================================================
# DIAGNOSTIC 4: AUI value for Indonesia (the headline number)
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 4: Indonesia AUI value")
print("=" * 70)

aui_col = None
for c in ["usage_per_capita_index", "aui", "AUI", "per_capita_index"]:
    if c in geo.columns:
        aui_col = c
        break

if aui_col:
    id_aui_vals = id_rows[aui_col].dropna().unique()
    print(f"\nAUI values found for Indonesia: {id_aui_vals}")
    print(f"(should be a single value around 0.36 per published reports)")
    
    # Compute global median for context
    all_aui = geo.dropna(subset=[aui_col]).groupby(geo_col)[aui_col].first()
    print(f"\nGlobal AUI distribution:")
    print(f"  Median: {all_aui.median():.3f}")
    print(f"  Mean:   {all_aui.mean():.3f}")
    print(f"  Min:    {all_aui.min():.3f} ({all_aui.idxmin()})")
    print(f"  Max:    {all_aui.max():.3f} ({all_aui.idxmax()})")
else:
    print("\nNo AUI column found. Look for usage_per_capita_index or similar.")

# ============================================================
# Summary: what's feasible?
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: What's feasible for an Indonesia thesis?")
print("=" * 70)

n_id = len(id_rows)
n_facets = id_rows[facet_col].nunique() if facet_col else 0
n_task_clusters = 0
if cluster_col and facet_col:
    task_facets = [f for f in id_rows[facet_col].unique()
                   if any(k in str(f).lower() for k in ["task", "onet"])]
    if task_facets:
        n_task_clusters = id_rows[id_rows[facet_col].isin(task_facets)][cluster_col].nunique()

print(f"\n  Indonesia rows total:        {n_id:,}")
print(f"  Indonesia facets present:    {n_facets}")
print(f"  Indonesia task-level values: {n_task_clusters}")

print("\n  Feasibility verdict:")
if n_id < 50:
    print("  - Sample is very small. Stick to AUI + aggregate automation share only.")
elif n_task_clusters < 20:
    print("  - Limited task resolution. SOC major group level analysis is doable.")
    print("  - Per-task or per-occupation analysis is not reliable.")
elif n_task_clusters < 100:
    print("  - Moderate task resolution. SOC major group + selected occupations work.")
    print("  - Be cautious with rare-task claims.")
else:
    print("  - Good task resolution. Full occupation-level analysis is feasible.")

print("\nDone. Review the output above before deciding on the next step.")
