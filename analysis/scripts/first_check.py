import pypsa
from pathlib import Path

# ========= Netzwerkpfad =========
pfad = Path(
    r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_NEP_2045\KN2045_Elek\networks\base_s_50_elec_.nc"
)

def ask_for_network_path():
    pfad = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    path = Path(pfad)
    return path

def create_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True) # parents --> , exist_ok --> 
    return path

def main():
    # ========= Abfrage Pfad =========
    # pfad = ask_for_network_path()

    # ========= Zielordner erstellen =========
    # create_output_dir(pfad_out)

    # ========= Netzwerk Laden =========
    print("Lade Netzwerk …")
    n = pypsa.Network(pfad)
    print("Netzwerk geladen:", n.meta['run']['prefix'])
    print("Netzwerk geladen:", n.meta['run']['name'])
    
    # ========= Allgemeiner erster Plot =========
    # n.plot() ??


















































































if __name__ == "__main__":
    main()