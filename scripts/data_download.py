import os
import urllib.request

URLS = {
    "collision": "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2024.csv",
    "vehicle":   "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-2024.csv",
    "casualty":  "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-2024.csv",
}

def download(url: str, out_path: str) -> None:
    print(f"Downloading -> {out_path}")
    urllib.request.urlretrieve(url, out_path)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download(URLS["collision"], "data/collision.csv")
    download(URLS["vehicle"], "data/vehicle.csv")
    download(URLS["casualty"], "data/casualty.csv")
    print("Done.")