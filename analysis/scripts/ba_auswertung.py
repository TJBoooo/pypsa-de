# =========================== Pakete ===========================
import pypsa
from pathlib import Path
import pandas as pd
import numpy as np
# =========================== Pakete für Plots ===========================
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('bmh')
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

# =========================== Funktionen ===========================
# =========================== Netzwerk Laden ===========================
def read_network(path_in):
    print("\nLade Netzwerk …\n")
    n = pypsa.Network(path_in)
    prefix = n.meta['run']['prefix']
    print(f'\nTitel: {prefix}\n')
    return n
# =========================== Zielordner erstellen & Pfad zwischenspeichern ===========================
def build_folder(n, folder_name, path_in):
    path = path_in.parents[1] / folder_name / f'c_{n.meta['scenario']['clusters'][0]}'
    # parents --> fehlende Ordner automatisch anlegen, exist_ok --> Kein Fehler, wenn Order schon da
    path.mkdir(parents=True, exist_ok=True) 
    print('Results:', path, '\n')
    return path
# =========================== Netzwerkdaten erzeugen & ablegen ===========================
def get_metadata(n, path, path_in):
    prefix = n.meta['run']['prefix']
    name = n.meta['run']['name']
    objective = getattr(n, "objective", None)
    start = str(n.snapshots[0]) if len(n.snapshots) > 0 else None
    end = str(n.snapshots[-1]) if len(n.snapshots) > 0 else None
    auflösung = str(n.snapshots[1]-n.snapshots[0])[8:-3]
    rows = [
            {"metric": "run_prefix", "value": prefix},
            {"metric": "run_name", "value": name},
            {"metric": "network_file", "value": str(path_in)},
            {"metric": "pypsa_version_loaded_from_file", "value": n.meta['version']},
            {"metric": "snapshots_count", "value": len(n.snapshots)},
            {"metric": "Auflösung", "value": auflösung},
            {"metric": "snapshots_start", "value": start},
            {"metric": "snapshots_end", "value": end},
            {"metric": "objective", "value": objective},
            {"metric": "cluster", "value": n.meta['scenario']['clusters'][0]},
            {"metric": "planning_horizons", "value": n.meta['scenario']['planning_horizons'][0]},
        ]

    overview = pd.DataFrame(rows)  # Array in Dataframe überführen
    overview.to_csv(path / f'Metadaten_{prefix}.csv', index = False)
# =========================== Allgemeiner erster Plot ===========================
def plot_network(n, path_out):
    title = 'Karte_Europa'
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
    fig.suptitle(f"{title} | {n.meta['run']['prefix']} | c: {n.meta['scenario']['clusters']}", fontsize=12, fontweight="bold")
    n.plot(ax=ax,geomap_color=True,bus_size=0.001)
    # fig.savefig(path / f"{title}_{prefix}_{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(path_out / f'{title}_{n.meta['run']['prefix'][0]}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= Erzeugen, der Netzwerkkomponenten =========
def network_components(n, path_out):
    rows = []
    for name, component in n.components.items():
        attr = name.lower()
        if hasattr(n, attr):
            df = getattr(n, attr)
            if isinstance(df, pd.DataFrame):
                rows.append({"Komponenten": name, "N": len(df)})
    pd.DataFrame(rows).to_csv(path_out / f'Komponenten_{n.meta['run']['prefix']}.csv', index = False)
# ========= Hinzufügen Diconary ========= 
def to_energy_dic(dic, category, name, value):
    dic.setdefault(category, {})
    dic[category][name] = float(value)
# ========= Speichern Diconary =========
def save_energy_dic(n, dic, path_out):
    print(dic)
    # dic = pd.DataFrame.from_dict(dic, orient="index", columns=["value"])
    # dic.to_csv(path_out / f'Generatortechnologien_{n.meta['run']['prefix']}.csv', index = True)
# ========= Generatordaten =========
def get_gen_data(n, path_out):
    gen_list = (n.generators.bus.to_frame('bus')
                .join(n.generators[['efficiency', 'p_max_pu', 'p_nom', 'p_nom_opt']])
                .join(((n.generators.p_nom_opt.sort_index() - n.generators.p_nom.sort_index()).rename('extendable_mw')))
                .join(n.generators[['p_nom_extendable', 'carrier', 'marginal_cost', 'capital_cost','overnight_cost']])
                .join((n.generators.capital_cost * (n.generators.p_nom_opt - n.generators.p_nom)).rename('extended_cost'))
                .join(n.generators_t.p.sum().rename('energy_2045_MWh'))
                )
    carrier_list = (
                gen_list.p_nom_opt.groupby(gen_list.carrier).sum().to_frame('p_nom_opt_mw')
                .join(gen_list.p_nom.groupby(gen_list.carrier).sum().rename('p_nom_mw'))
                .join(gen_list.extendable_mw.groupby(gen_list.carrier).sum().rename('extendeble_mw'))
                .join(gen_list.carrier.value_counts().rename('n_generators'))
                .join(gen_list.marginal_cost.groupby(gen_list.carrier).mean().rename('marginal_cost_€_mw'))
                .join(gen_list.capital_cost.groupby(gen_list.carrier).mean().rename('capital_cost_€'))
                .join(gen_list.overnight_cost.groupby(gen_list.carrier).mean().rename('overnight_cost_€'))
                .join(gen_list.extended_cost.groupby(gen_list.carrier).mean().rename('extended_cost'))
                .join((gen_list.energy_2045_MWh.groupby(n.generators.carrier).sum()))
                )
    carrier_list = carrier_list.join((n.carriers["color"]).loc[carrier_list.index])
    gen_list.to_csv(path_out / f'Generatoren_{n.meta['run']['prefix']}.csv', index = True)
    carrier_list.to_csv(path_out / f'Generatortechnologien_{n.meta['run']['prefix']}.csv', index = True)
    return (gen_list, carrier_list)
# ========= Meritorder =========
def merit_order(n, gen_list, path_out):
    df_merit = (gen_list.sort_values('marginal_cost',ascending=True).marginal_cost.to_frame()
            .join(gen_list.sort_values('marginal_cost',ascending=True).p_nom_opt.cumsum().rename('p_nom_opt_cum'))
            .join(gen_list[['carrier', 'p_nom_opt']]))
    df_merit.to_csv(path_out / f'Meritorder_{n.meta['run']['prefix']}.csv', index = True)
    # Erwartet: df_merit mit Spalten: marginal_cost, p_nom_opt, carrier
    df_merit["left_MW"] = df_merit["p_nom_opt_cum"] - df_merit["p_nom_opt"]

    # Farben je Carrier aus Pypsa Carrier
    color_map = n.carriers["color"].to_dict()
    colors = df_merit["carrier"].map(color_map).fillna("#999999").tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(
        df_merit["left_MW"] / 1000.0,       # Start in GW
        df_merit["marginal_cost"],          # Höhe in €/MWh
        width=df_merit["p_nom_opt"] / 1000.0,  # Breite in GW -- Mengenachse!!
        align="edge",
        color=colors
    )

    ax.set_xlabel("Kummulierte Kapazität (GW)")
    ax.set_ylabel("Grenzkosten (€/MWh)")
    ax.set_title("Meritorder ")

    # Generator-Namen auf der x-Achse entfernen
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Maximalwert der kumulierten Leistung in GW
    x_max = df_merit["p_nom_opt_cum"].iloc[-1] / 1000.0

    # Automatisch sinnvolle Tick-Anzahl erzeugen (z.B. 8 Schritte)
    ticks = np.linspace(0, x_max, 9)

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks])

    # Legendeneinträge pro Carrier (Proxy-Patches)
    from matplotlib.patches import Patch
    handles = []
    seen = set()
    for c, col in zip(df_merit["carrier"], colors):
        if c not in seen:
            handles.append(Patch(facecolor=col, label=c))
            seen.add(c)

    ax.legend(handles=handles, title="Carrier", loc="upper left", bbox_to_anchor=(1.02, 1))

    ax.set_xlim(0, df_merit["p_nom_opt_cum"].iloc[-1] / 1000.0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()

    fig.savefig(path_out / f'Meritorder_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= Technologieart nach Energieart sortieren =========
def energy_key(list):
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
    return (list.energy_2045_MWh.T.groupby(carrier_map).sum().T) # Technologien den Energiearten zuordnen
# ========= Technologiefarben mischen =========
def color_of_energy():
    # Feste Farben je Energieart (konstant für alle Plots)
    energy_colors = {
        "renewable": "#54a24b",
        "fossil": "#4c4c4c",
        "Gas": "#f58518",
        "water-power": "#4c78a8",
        "nuclear": "#b279a2"
    }
    return energy_colors
# ========= %-EE-Jahresenergie (Erzeugt) =========
def pc_year_energy(n, carrier_list, path_out):

    energy_by_energyart = energy_key(carrier_list)

    data = energy_by_energyart.dropna()
    data = data[data > 0]

    colors = [color_of_energy()[e] for e in data.index]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.pie(
        data,
        labels=data.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    plt.title(f"Energieverteilung {n.meta['scenario']['planning_horizons'][0]} [MWh]")
    plt.ylabel("")
    
    fig.savefig(path_out / f'Energiearten_prozentual_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
    # ========= AC-Daten =========
def get_ac_data(n):
    if n.lines_t.q0.empty == True: # Prüfung, ob Blindleistung mit Engerichtet wurde
        S = n.lines_t.p0.abs() # |p0| wenn Q≈0 -- abs() -- betrag der Wirkleistung
    else:
        S = (n.lines_t.p0 ** 2 + n.lines_t.q0 ** 2) ** 0.5

    loading_max = ((S.max() # Maximalwert!
        .sort_index() # mean() arithmetisches Mittel aller Zeitwerte -- sort_index() im Nenner & im Zähler! -- richtige Reihenfolge
          / (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index() # .s_max_pu -- Zulässiger Anteil von s_nom_opt
           ).fillna(0.0))
    
    loading_mean = ((
        S.mean() # Mittelwert!!
        .sort_index() # mean() arithmetisches Mittel aller Zeitwerte -- sort_index() im Nenner & im Zähler! -- richtige Reihenfolge
          / 
          (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index() # .s_max_pu -- Zulässiger Anteil von s_nom_opt
           ).fillna(0.0))

    list_lines = (n.lines.bus0.to_frame('bus0')
              .join(n.lines[["bus1", "s_nom", "s_nom_opt", "capital_cost"]])
            #   .join(n.lines[["length", "carrier"]])
              .join(loading_max.rename('max_loading'))
              .join(loading_mean.rename('mean_loading'))
              .join(((n.lines['s_nom_opt'] - n.lines['s_nom'])).rename('expension_mw'))
              .join(n.lines[["overnight_cost"]])
        )
    
    list_lines = list_lines.join((list_lines.overnight_cost.sort_values(ascending=False) * list_lines.expension_mw.sort_values(ascending=False)).rename('expensions_cost')) # Kosten des Ausbaus
    list_lines=list_lines.join(((list_lines.expensions_cost.sort_values(ascending=False) / list_lines.expensions_cost.sum()) * 100).rename('pc_expensions_cost')) # pc-Kosten des Ausbaus
# ============================================== MAIN ==============================================
def main():
    # ===== PFAD =====
    # path_row = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    # path_in = Path(path_row)
    path_in = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_Referenzoptimierung\KN2045_Elek\networks\base_s_all_elec_.nc") # Auskommentieren, wenn fertig
    # ===== Netzwerk einlesen =====
    n = read_network(path_in)
    # ===== Ablageordner erstellen =====
    path_out = build_folder(n, 'analysis_final', path_in)
    # ===== Netzwerkdaten =====
    get_metadata(n, path_out, path_in)
    plot_network(n, path_out)
    network_components(n, path_out)
    # ===== Diconary für Energy anlegen =====
    energy_dic = {}
    # ===== Generatoren =====
    gen_list, carrier_list = get_gen_data(n, path_out)
    merit_order(n, gen_list, path_out)
    pc_year_energy(n, carrier_list, path_out)
    to_energy_dic(energy_dic, 'Gesamtenergie_[MWh]', 'generator', (gen_list.energy_2045_MWh.sum()))
    to_energy_dic(energy_dic,  'Gesamtkostem_[€]', 'generator', gen_list['extended_cost'].sum())
    # ===== Lasten =====
    to_energy_dic(energy_dic, 'Gesamtenergie_[MWh]', 'Last', n.loads_t.p.sum().sum())
























    # ===== Diconary für Energy speichern =====
    save_energy_dic(n, energy_dic, path_out)
if __name__ == "__main__":
    main()
    print('\nFinsih')