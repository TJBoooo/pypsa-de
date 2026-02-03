from pathlib import Path
import pypsa


RESULTS_NC = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_NEP_2045\KN2045_Elek\networks\base_s_50_elec_.nc")
EXPORT_DIR = Path("analysis/tables/BA_NEP_2045_C50_1h")

def main():
    n_import = pypsa.Network(RESULTS_NC)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True) #
    n_import.export_to_csv_folder(EXPORT_DIR)
    print(f'CSV-Export in {EXPORT_DIR} abgeschlossen.')

if __name__ == "__main__":
    main()