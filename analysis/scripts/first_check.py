# ========= Pakete =========
import pypsa
from pathlib import Path
import pandas as pd
import numpy as np
# ========= Pakete für Plots =========
import matplotlib.pyplot as plt
plt.style.use('bmh')
import cartopy.crs as ccrs

# ========= Netzwerkpfad =========
def ask_for_network_path():
    path_row = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    path = Path(path_row)
    return path

# def create_output_folder(base, präfix, name):
#     path = base / präfix / name
#     path.mkdir(parents=True, exist_ok=True) # parents --> fehlende Ordner automatisch anlegen, exist_ok --> Kein Fehler, wenn Order schon da
#     print('Results:', path, '\n')
#     return path # Funktion überschreibt Ordner nicht -- Gibt Pfad zu Ordner zurück.

def get_output_path(folder, filename, suffix):
    if not suffix.startswith("."):
        suffix = "." + suffix
    return folder / f"{filename}{suffix}"

# Overview (Key–Value, super für Excel)
def build_overview_df(n, prefix, name, pfad):
    # objective ist nicht immer vorhanden/gesetzt -> safe
    objective = getattr(n, "objective", None)

    # time range
    start = str(n.snapshots[0]) if len(n.snapshots) > 0 else None
    end = str(n.snapshots[-1]) if len(n.snapshots) > 0 else None
    auflösung = str(n.snapshots[1]-n.snapshots[0])[8:-3]

    rows = [
        {"metric": "run_prefix", "value": prefix},
        {"metric": "run_name", "value": name},
        {"metric": "network_file", "value": str(pfad)},
        {"metric": "pypsa_version_loaded_from_file", "value": n.meta['version']},
        {"metric": "snapshots_count", "value": len(n.snapshots)},
        {"metric": "Auflösung", "value": auflösung},
        {"metric": "snapshots_start", "value": start},
        {"metric": "snapshots_end", "value": end},
        {"metric": "objective", "value": objective},
        {"metric": "cluster", "value": n.meta['scenario']['clusters']},
        {"metric": "planning_horizons", "value": n.meta['scenario']['planning_horizons']},
    ]
    return pd.DataFrame(rows)  # Array in Dataframe überführen

def network_components(n): #, out_path):
    rows = []
    for name, component in n.components.items():  # n.components --> Diconary, items() --> Key-Value
        attr = name.lower() # name ist ein string, .lower() --> Kleinbucstaben
        if hasattr(n, attr): # Prüfen, ob n (das Netzwerk) ein Atribut mit dem Namen hat
            df = getattr(n, attr)
            if isinstance(df, pd.DataFrame):
                rows.append({"component": name, "count": len(df)}) # Componeten in Array schreiben
    
    return pd.DataFrame(rows)  # Array in Dataframe überführen

# ============================================== MAIN ==============================================
def main():
    # ========= Auswahl Ergebnisdatei =========
    # pfad = ask_for_network_path() # Später wieder rein kommentieren!
    pfad = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_NEP_2045\KN2045_Elek\networks\base_s_50_elec_.nc")

    # ========= Netzwerk Laden =========
    print("\nLade Netzwerk …\n")
    n = pypsa.Network(pfad)
    prefix = n.meta['run']['prefix']
    name = n.meta['run']['name']
    print("\nTitel:", prefix, name,'\n')

    # ========= Zielordner erstellen & Pfad zwischenspeichern=========
    base = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\analysis/results")
    path = base / prefix / name # Def for later
    path.mkdir(parents=True, exist_ok=True) # parents --> fehlende Ordner automatisch anlegen, exist_ok --> Kein Fehler, wenn Order schon da
    print('Results:', path, '\n')

    # ========= Owerview Netzwerk erzeugen & ablegen =========
    overview = build_overview_df(n, prefix, name, pfad)
    # print(overview)
    overview.to_csv(path / f'overview_{prefix}_{name}.csv')
    # print('Owerview wurde abgespeichert.\n')

    # ========= Allgemeiner erster Plot =========
    plot_path = path / f"map_{prefix}_{name}.png"
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
    fig.suptitle(
    f"PyPSA-DE – {prefix} | {name}",
    fontsize=12,
    fontweight="bold"
    )
    n.plot(ax=ax)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # print('Karte wurde in Ordner abgelegt.')
    
    # ========= Netzwerkcomponenten =========
    components = network_components(n)
    # print("\nAnzahl der Komponenten:\n", components)
    components.to_csv(path / f'components_{prefix}_{name}.csv', index = False)

    # ========= Generators =========
    # create folder
    path_gen = path / 'generators'
    path_gen.mkdir(parents=True, exist_ok=True)
     # list of all gen
    n.generators.to_csv(path_gen / f'origin_list-gen_{prefix}_{name}.csv', index = True)
    # list of opt gen
    mod_gen = n.generators.query("p_nom_mod > 0")
    mod_gen.to_csv(path_gen / f'mod_gen_{prefix}_{name}.csv', index = True)

    # ========= Carriers =========
    # create folder
    path_carrries = path / 'carriers'
    path_carrries.mkdir(parents=True, exist_ok=True)
        # used carrier
    carrier_table = (
        n.generators
        .groupby("carrier")
        .size()
        .reset_index(name="n_generators")
        .sort_values("n_generators", ascending=False)
    )
    carrier_table.to_csv(path_carrries / f'used_carrier_{prefix}_{name}.csv', index=False)
    # cost per carriere (arithetisches Mittel --> mean())
    carrier_cost = (
        n.generators.groupby("carrier")
        .marginal_cost.mean()
        .reset_index()
        .rename(columns={'marginal_cost': 'marginal_cost [€ / MWh]'}))
    
    carrier_cost.to_csv(path_carrries / f"marginal_cost_by_carrier_{prefix}_{name}.csv", index=False)
    # expanion by carrier
    expansion_by_carrier = (mod_gen.groupby("carrier")["p_nom_mod"].sum().reset_index().rename(columns={"p_nom_mod": "p_nom_mod [MW]"})
                            # .sort_values("capacity_added_mw", ascending=False)
                            )
    expansion_by_carrier.to_csv(path_carrries / f'mod_carriers_{prefix}_{name}.csv', index = False)


    energy_balance = n.statistics.energy_balance().reset_index().rename(columns={'0': 'MWh'})
    energy_balance.to_csv(path / f'Energiebilanz_{prefix}_{name}.csv', index=False)

    # carr snapshot
    gen_by_carrier = (
        n.generators_t.p # original df
        .T # transponieren
        .groupby(n.generators.carrier) # sortieren nach carrier
        .sum() # generatorleistung je carrier aufsummieren
        .T # zurück transponieren
        )
    gen_by_carrier.to_csv(path_carrries / f"carr_snap_{prefix}_{name}_by_carrier.csv",index=True)
# ============================================== Schauen wir ==============================================

    


    #Time-varying component data
    # n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3)) #MW?

















if __name__ == "__main__":
    main()
    print('Finsih')