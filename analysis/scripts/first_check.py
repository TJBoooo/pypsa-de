# =========================== Pakete ===========================
import pypsa
from pathlib import Path
import pandas as pd
import numpy as np
# =========================== Pakete für Plots ===========================
import matplotlib.pyplot as plt
plt.style.use('bmh')
import cartopy.crs as ccrs

# =========================== Funktionen ===========================
# ========= Erzeugen, der Netzwerkkomponenten =========
def network_components(n): #, out_path):
    rows = []
    for name, component in n.components.items():  # n.components --> Diconary, items() --> Key-Value
        attr = name.lower() # name ist ein string, .lower() --> Kleinbucstaben
        if hasattr(n, attr): # Prüfen, ob n (das Netzwerk) ein Atribut mit dem Namen hat
            df = getattr(n, attr)
            if isinstance(df, pd.DataFrame):
                rows.append({"component": name, "count": len(df)}) # Componeten in Array schreiben
    
    return pd.DataFrame(rows)  # Array in Dataframe überführen

def export_component_t(path_secion, component, title_cap, network):
    export_path = (path_secion / 'Zeitreihen')
    export_path.mkdir(parents=True, exist_ok=True)
    prefix = network.meta['run']['prefix']
    name = network.meta['run']['name']
    pd.DataFrame(component.keys()).to_csv(export_path / f'Time_Keys_{title_cap}_{prefix}_{name}.csv', index = False)
    # ===================
    for field in component.keys():
        df = getattr(component, field)
        df.to_csv(export_path / f'{field}_{prefix}_{name}.csv', index = True)

    return 

# ============================================== MAIN ==============================================
def main():
    # ========= Wo liegt das Netzwerk / die .nc-Datei? =========
    # path_row = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    # path = Path(path_row)
    path = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_NEP_2045\KN2045_Elek\networks\base_s_50_elec_.nc") # Auskommentieren, wenn fertig
    
    # ========= Netzwerk Laden =========
    print("\nLade Netzwerk …\n")
    n = pypsa.Network(path)
    prefix = n.meta['run']['prefix']
    name = n.meta['run']['name']
    print(f'\nTitel: {prefix} {name} \n')

    # ========= Zielordner erstellen & Pfad zwischenspeichern=========
    base = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\analysis/results")
    path = base / prefix / name # Def for later
    path.mkdir(parents=True, exist_ok=True) # parents --> fehlende Ordner automatisch anlegen, exist_ok --> Kein Fehler, wenn Order schon da
    print('Results:', path, '\n')

    # ========= Netzwerkdaten erzeugen & ablegen =========
    objective = getattr(n, "objective", None)
    start = str(n.snapshots[0]) if len(n.snapshots) > 0 else None
    end = str(n.snapshots[-1]) if len(n.snapshots) > 0 else None
    auflösung = str(n.snapshots[1]-n.snapshots[0])[8:-3]
    rows = [
            {"metric": "run_prefix", "value": prefix},
            {"metric": "run_name", "value": name},
            {"metric": "network_file", "value": str(path)},
            {"metric": "pypsa_version_loaded_from_file", "value": n.meta['version']},
            {"metric": "snapshots_count", "value": len(n.snapshots)},
            {"metric": "Auflösung", "value": auflösung},
            {"metric": "snapshots_start", "value": start},
            {"metric": "snapshots_end", "value": end},
            {"metric": "objective", "value": objective},
            {"metric": "cluster", "value": n.meta['scenario']['clusters']},
            {"metric": "planning_horizons", "value": n.meta['scenario']['planning_horizons']},
        ]

    overview = pd.DataFrame(rows)  # Array in Dataframe überführen
    overview.to_csv(path / f'overview_{prefix}_{name}.csv')

    # ========= Allgemeiner erster Plot =========
    title = 'Karte_Europa'

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
    fig.suptitle(f"{title} | {prefix} | {name}", fontsize=12, fontweight="bold")
    n.plot(ax=ax)
    fig.savefig(path / f"{title}_{prefix}_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # ========= Netzwerkcomponenten =========
    title = 'components'
    components = network_components(n)
    # print("\nAnzahl der Komponenten:\n", components)
    components.to_csv(path / f'{title}_{prefix}_{name}.csv', index = False)


    # ========= Energiebilanz =========
    title = 'Energiebilanz'
    energy_balance = (n.statistics.energy_balance()
                      .reset_index(name = 'energiy_MWh')
                      )
    energy_balance.to_csv(path / f'{title}_{prefix}_{name}.csv', index = False)

    # =========================== Generators ===========================
    title_cap = 'Generatoren'
    # Ordner erstellen
    path_secion = path / 'Generatoren'
    path_secion.mkdir(parents=True, exist_ok=True)
    # ===================
    title = 'Verwendete_Generatoren'
    n.generators.to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    export_component_t(path_secion, n.generators_t, title_cap, n)
    # ===================
    title = 'Optimierte_Generatore'
    untergrenze_opt = 0.5
    opt_gen = n.generators.p_nom_opt - n.generators.p_nom
    opt_gen_pos = (opt_gen[opt_gen> untergrenze_opt])
    opt_gen_pos.reset_index(name='p_nom_mod').to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = True) 
    # ===================
    title = 'Anzahl_Generator_pro_Carrier'
    (n.generators.carrier.value_counts().reset_index(name="n_generators")
     ).to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index=False)
    # ===================
    title = 'Wirkleistungkurve_der_Generatoren_01_01_2019'
    # Wirkleistung, die in Netz abgegeben wird
    n.generators_t.p.loc['01-01-2019'].to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index=True)

    # ===================
    title = 'Generatorkapazitäten_pro_Technologie'
    # Zeitintervall unabhängig
    untergrenze_opt = 0.5  # MW

    gens = n.generators.copy()
    gens["expansion_mw"] = gens["p_nom_opt"] - gens["p_nom"]          # Zwischenrechnung
    gens["expansion_pos_mw"] = gens["expansion_mw"].where(gens["expansion_mw"] > untergrenze_opt, 0.0)

    liste_technologien = (
    gens.groupby("carrier")
    .agg(
        n_generators=("carrier", "size"),
        p_nom_mw=("p_nom", "sum"),
        p_nom_opt_mw=("p_nom_opt", "sum"),
        expansion_mw=("expansion_mw", "sum"),           # Summe der Differenzen
        expansion_pos_mw=("expansion_pos_mw", "sum"),   # Summe nur über Schwelle
    ).reset_index().sort_values("p_nom_mw", ascending=False))

    liste_technologien.to_csv(path_secion / f"{title}_{prefix}_{name}.csv",index=False)

    df = liste_technologien.set_index("carrier")#.sort_values("p_nom_opt_mw", ascending=False)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        sharex=True)

    #  1 Installierte Kapazität
    df["p_nom_mw"].plot.bar(ax=axes[0])
    axes[0].set_ylabel("MW")
    axes[0].set_title("Installierte Kapazität [p_nom]")
    # 2 Optimierte Kapazität
    df["p_nom_opt_mw"].plot.bar(ax=axes[1])
    axes[1].set_ylabel("MW")
    axes[1].set_title("Optimierte Kapazität [p_nom_opt]")
    # 3 Ausgebaute Kapazität (nur > Schwelle)
    df["expansion_mw"].plot.bar(ax=axes[2])
    axes[2].set_ylabel("MW")
    axes[2].set_title("Ausgebaute Kapazität [p_nom_mod]")
    # Gemeinsamer Titel
    fig.suptitle(
        f"{title} | {prefix} | {name}",
        fontsize=12,
        fontweight="bold")
    # Feinschliff
    axes[2].set_xlabel("Technologie")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Speichern
    fig.savefig(
        path_secion / f"{title}t_{prefix}_{name}.png",
        dpi=300,
        bbox_inches="tight")
    plt.close(fig)

    # ===================
    title = 'Optimierte_Technologien'
    opt_energy_pos = opt_gen_pos.groupby(n.generators.carrier).sum()
    opt_energy_pos.reset_index(name='p_nom_mod').to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = False)
    # ===================
    title = 'Marginale_Kosten'
    # marginale Kosten = varianle Kosten, um eine zusaätzliche MWh zu erzeugen
    (n.generators.groupby("carrier").marginal_cost.mean() # arithmetisches Mittel
        .reset_index().rename(columns={'marginal_cost': 'marginal_cost [€ / MWh]'})).to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index=False)
    # ===================
    title = 'Wirkleistungskurven_der_Technologien'
    # Wirkleistung, die in Netz abgegeben wird
    p_per_tech = (n.generators_t.p.T.groupby(n.generators.carrier).sum().T)
    p_per_tech.to_csv(path_secion / f"{title}_{prefix}_{name}.csv", index=True)
    # ===================
    title = "(EU)_Generatorleistung_zu_Energiearten_zusammengefast_pro_Jahr"
    # Technologien nach Energien zusammenfassen
    carrier_map = pd.Series({ # Serie mit Energiearten
        "CCGT": "Gas",
        "OCGT": "Gas",
        "biomass": "renewable",
        "coal": "fossil",
        "lignite": "fossil", # Braunkohle
        "nuclear": "nuclear",
        "offwind-ac": "renewable",
        "offwind-dc": "renewable",
        "offwind-float": "renewable",
        "oil": "fossil",
        "onwind": "renewable",
        "ror": "water-power",
        "solar": "renewable",
        "solar-hsat": "renewable"
    })
    p_per_energytyp = (p_per_tech.T.groupby(carrier_map).sum().T) # Technologien den Energiearten zuordnen
    sec_prio = p_per_energytyp.sum(axis=0) # Spalten aufsummieren 0 -- Spalten
    first_prio = sec_prio.sort_values(ascending=False).index # Indexe nach Summe sortieren
    p_per_energytyp_sorted = (
        (p_per_energytyp[first_prio]) # Index neu sortieren
        .resample('W') # df.resample("<FREQUENZ>").<Aggregation>() 
        # -- ME - Monatsende
        # -- W - wöchentlich
        # -- D - täglich
        # -- h - stündlich
        # -- für andere auflösungen Chati fragen
        .mean()) # arithmetischer Mittelwert
    p_per_energytyp_sorted.to_csv(path_secion / f'{title}_{prefix}_{name}.csv')
    # Diagrammeinstellungen
    ax = p_per_energytyp_sorted.plot.area(figsize=(20, 8), title=title)
    ax.set_xlabel('t') # Diagrammbeschriftung
    ax.set_ylabel("Power [p | MW]") # Diagrammbeschriftung
    # Ploteinstellungen
    ax.figure.suptitle(f"Auflösung: Wöchentlich", fontsize=12, fontweight="bold")
    ax.figure.savefig(path_secion / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
    plt.close(ax.figure)

    # ===================
    title = 'Gesamterzeugung_pro_Jahr_und_Technologie'
    energie_pro_technologie = p_per_tech.sum().sort_values(ascending=False)  # Summe über Zeit & nach Größe sortiert
    energie_pro_technologie.to_csv(path_secion / f"{title}_{prefix}_{name}.csv",index=True)
    # Diagrammeinstellungen & Speichern
    ax = energie_pro_technologie.plot.bar(figsize=(10,4))
    ax.set_ylabel("Energie pro Jahr [MWh]")
    ax.set_title(title)
    # ax.figure.suptitle(f"Subtitle", fontsize=12, fontweight="bold")
    ax.figure.savefig(path_secion / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
    plt.close(ax.figure)

    # ===================
    title = 'Gesamterzeugung_pro_Jahr_und_Energieart'
    energie_pro_energytyp = p_per_energytyp.sum().sort_values(ascending=False)  # Summe über Zeit & nach Größe sortiert
    energie_pro_energytyp.to_csv(path_secion / f"{title}_{prefix}_{name}.csv",index=True)
    # Diagrammeinstellungen & Speichern
    ax = energie_pro_energytyp.plot.bar(figsize=(10,4))
    ax.set_ylabel("Energie pro Jahr [MWh]")
    ax.set_title(title)
    # ax.figure.suptitle(f"Subtitle", fontsize=12, fontweight="bold")
    ax.figure.savefig(path_secion / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
    plt.close(ax.figure)

    # ===================
    top_n = 8
    title = f"Top {top_n} Technologien & Andere"
    # Top 8 Technologien & andere
    
    s = energie_pro_technologie
    top = s.head(top_n)
    rest = pd.Series({"Other": s.iloc[top_n:].sum()})

    plot_series = pd.concat([top, rest])

    ax = plot_series.plot.bar(figsize=(10,4))
    ax.set_ylabel("Energy [MWh]")
    ax.set_title(title)
    ax.figure.savefig(path_secion / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
    plt.close(ax.figure)

# Kann der Ordner Technologien in Generatoren sinnvoll mit eingepflegt werden?
# ============================================== Schauen wir ==============================================

    # =========================== Buses ===========================
    # Ordner erstellen
    path_secion = path / 'Buses'
    path_secion.mkdir(parents=True, exist_ok=True)

    # ===================
    title = 'n.buses'
    n.buses.to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = False)
    title = 'n.buses_t'
    # n.buses_t.to_csv(path / f'{title}_{prefix}_{name}.csv', index = False)

    # =========================== Carriers ===========================
    # Ordner erstellen
    path_secion = path / 'Carriers'
    path_secion.mkdir(parents=True, exist_ok=True)






    # =========================== Lines / Leitungen ===========================
    title_cap = 'Leitungen'
    # Ordner erstellen
    path_secion = path / f'{title_cap}'
    path_secion.mkdir(parents=True, exist_ok=True)
    # ===================
    title = f'Verwendete_{title_cap}'
    n.lines.to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    export_component_t(path_secion, n.lines_t, title_cap, n)
    # ===================
    title = f'Auslastung_%_{title_cap}'
    loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index()
           ).fillna(0.0)
    loading.to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    title = f'Statistic_Auslastung_{title_cap}'
    loading.describe().to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    title = f'Statistic_Überlastete_{title_cap}'
    ueberlstung = 0.7
    anz = (loading > ueberlstung).sum()   # Anzahl stark ausgelasteter Leitungen
    (loading.sort_values(ascending=False).head(anz + 1)).to_csv(path_secion / f'{title}_{prefix}_{name}.csv', index = True)
    
    # ===================
    title = f'Karte_Auslastung_{title_cap}'
    plot_path = path_secion / f'{title}_{prefix}_{name}.png'

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.Orthographic()}
    )

    n.plot(
        ax=ax,
        bus_colors="gray",
        branch_components=["Line"],
        line_widths=n.lines.s_nom_opt / 3e3,
        line_colors=loading,
        line_cmap=plt.cm.viridis,
        geomap_color=True,
        bus_sizes=0
    )

    ax.set_title(
        f"Leitungsauslastung – {prefix} | {name}",
        fontsize=12,
        fontweight="bold"
    )

    ax.axis("off")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ===================






    # =========================== Links ===========================
    # Ordner erstellen
    path_secion = path / 'Links'
    path_secion.mkdir(parents=True, exist_ok=True)
    # =========================== Loads ===========================
    # Ordner erstellen
    path_secion = path / 'Loads'
    path_secion.mkdir(parents=True, exist_ok=True)
    #Time-varying component data
    title = 'n.loads'
    n.loads_t.p_set.sum(axis=1).to_csv(path_secion / f"{title}_{prefix}_{name}.csv",index=True)
    
    # Diagrammeinstellungen & Speichern
    ax = n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3)) #MW?
    ax.set_ylabel("MW?")
    ax.set_title(title)
    # ax.figure.suptitle(f"Subtitle", fontsize=12, fontweight="bold")
    ax.figure.savefig(path_secion / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight") # Plot speichern
    plt.close(ax.figure)
    # =========================== Storage_Units ===========================
    # Ordner erstellen
    path_secion = path / 'Storage_Units'
    path_secion.mkdir(parents=True, exist_ok=True)
    # =========================== Stores ===========================
    # Ordner erstellen
    path_secion = path / 'Stores'
    path_secion.mkdir(parents=True, exist_ok=True)
    # =========================== Sub_networks ===========================
    # Ordner erstellen
    path_secion = path / 'Sub_networks'
    path_secion.mkdir(parents=True, exist_ok=True)

    















if __name__ == "__main__":
    main()
    print('Finsih')