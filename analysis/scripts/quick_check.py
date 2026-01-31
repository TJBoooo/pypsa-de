from pathlib import Path
import pypsa

# Passe das an deinen Run an:
RESULTS_NC = Path("results/BA_NEPSens_2045/KN2045_Elek/networks/base_s_21_elec_.nc")

def main():
    if not RESULTS_NC.exists():
        raise FileNotFoundError(f"Network file not found: {RESULTS_NC.resolve()}")

    n = pypsa.Network(RESULTS_NC)

    print("Loaded:", RESULTS_NC)
    print("Snapshots:", len(n.snapshots))
    print("Buses:", len(n.buses), "Lines:", len(n.lines), "Links:", len(n.links))
    print("Generators:", len(n.generators), "Loads:", len(n.loads), "StorageUnits:", len(n.storage_units))

    print(n.generators.head())

if __name__ == "__main__":
    main()