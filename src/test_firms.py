import geopandas as gpd
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Real NASA FIRMS NRT data
firms_file = BASE_DIR / "data" / "firms" / "fire_nrt_SV-C2_809176.shp"

print("Reading NASA FIRMS data...")
print("File:", firms_file)

# Read shapefile
gdf = gpd.read_file(firms_file)

print("\n===== SUCCESS =====")
print("Total records:", len(gdf))

print("\nColumns:")
print(gdf.columns.tolist())

print("\nCoordinate System:")
print(gdf.crs)

print("\nFirst 5 records:")
print(gdf.head())
print("\n===== DATA TYPES =====")
print(gdf.dtypes)

print("\n===== MISSING VALUES =====")
print(gdf.isnull().sum())

print("\n===== SAMPLE DATA =====")
print(gdf.drop(columns="geometry").head(10).to_string())