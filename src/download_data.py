"""
Downloads the three public datasets this project uses.
Run once before build_dataset.py. Everything lands in data/raw/.

Sources (all public):
  - District boundaries: geohacker/india (GADM-derived GeoJSON)
  - District population/literacy/workforce: Census of India 2011
  - District COVID case counts: covid19india community archive
"""
import os
import urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW, exist_ok=True)

FILES = {
    "districts.geojson":
        "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    "census2011.csv":
        "https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv",
    "district_latest.csv":
        "https://raw.githubusercontent.com/imdevskp/covid-19-india-data/master/district_level_latest.csv",
}

for name, url in FILES.items():
    dest = os.path.join(RAW, name)
    if os.path.exists(dest):
        print("already have", name)
        continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, dest)
    print("  saved", dest)

print("done. next: python src/build_dataset.py")
