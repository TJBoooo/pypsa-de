# =========================== Pakete ===========================
import pypsa
from pathlib import Path
import pandas as pd
import numpy as np
# =========================== Pakete für Plots ===========================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
plt.style.use('bmh')
import cartopy.crs as ccrs
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
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
    fig.suptitle(f"{n.meta['run']['prefix']} | Cluster: {n.meta['scenario']['clusters'][0]} | N-Knoten: {len(n.buses.index)}", fontsize=12, fontweight="bold")
    n.plot(ax=ax,geomap_color=True,bus_size=0.001)
    fig.savefig(path_out / f'Karte_allgemein_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
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
    pd.DataFrame(rows).to_csv(path_out / f'Meta_Komponenten_{n.meta['run']['prefix']}.csv', index = False)
# ========= Hinzufügen Diconary ========= 
def to_run_dic(dic, category, name, value):
    dic.setdefault(category, {})
    dic[category][name] = float(value)
# ========= Speichern Diconary =========
def save_run_dic(n, dic, path_out):
    df = pd.DataFrame(dic)
    df.to_csv(path_out / f'Meta_Diconary_{n.meta['run']['prefix']}.csv', index = True)
# ========= Technologieart nach Energieart sortieren =========
def carrier_key():
    carrier_map = pd.Series({ # Serie mit Energiearten
        "CCGT": "Gas", 
        "OCGT": "Gas", 
        "biomass": "Biomasse", 
        "coal": "Kohle", 
        "lignite": "Braunkohle", # Braunkohle 
        "nuclear": "Kernenergie", 
        "offwind-ac": "Wind",
        "offwind-dc": "Wind",
        "offwind-float": "Wind",
        "oil": "Öl",
        "onwind": "Wind",
        "ror": "Wasserkraft", 
        "solar": "Solar", 
        "solar-hsat": "Solar",
        'loads': 'Last'
    })
    return carrier_map # Technologien den Oberarten zuordnen
# ========= Technologiefarben mischen =========
def color_key(n):
    # Feste Farben je Energieart (konstant für alle Plots)
    energy_colors = {
        "Biomasse": n.carriers.color.get('biomass', '#999999'),
        "Braunkohle": n.carriers.color.get('lignite', '#999999'),
        "Kernenergie":  n.carriers.color.get('nuclear', '#999999'),
        "Kohle":  n.carriers.color.get('coal', '#999999'),
        "Gas":  n.carriers.color.get('CCGT', '#999999'),
        "Solar":  n.carriers.color.get('solar', '#999999'),
        "Wasserkraft":  n.carriers.color.get('ror', '#999999'),
        "Wind":  n.carriers.color.get('onwind', '#999999'),
        "Öl":  n.carriers.color.get('oil', '#999999'),
        'Gesamtlast': "#F90909",
        'Gesamteinspeisung':  "#1DF909",
        'Generatoreinspeisung': "#E68610EE"
    }
    return energy_colors
# ========= Time-Plot (N-Achsen) =========
def time_plot_n_axses(n, data_plot, path_out, file_title, time_period): # nrows !> 1
    n_rows = int(data_plot.columns.value_counts().sum())
    colors = [color_key(n).get(col, '#999999') for col in data_plot.columns]
    linewidth = 0.8 
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(18, 10),
        sharex=True)

    fig.suptitle(f"{n.meta['run']['prefix']}\nZeitverlauf: {time_period}", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for i in range(n_rows):
        data_plot.iloc[:,i:i+1].plot(ax=axes[i], color=[colors[i]], linewidth=linewidth, legend=False)
        axes[i].set_ylabel("MW")
        axes[i].set_title(f"{data_plot.columns[i]}")

    fig.savefig(path_out / f'{file_title}_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= Time-Plot (1-Achse) =========
def time_plot_1_axes(n, data_plot, path_out, file_title):

    # Farbmap holen (dict: name -> hex)
    colors = [color_key(n).get(col, '#999999') for col in data_plot.columns]
    linewidth = 0.8
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.suptitle(f"Zeitverlauf: {n.meta['run']['prefix']}", fontsize=22)
    data_plot.plot(ax=ax, color=colors, linewidth=linewidth)

    ax.set_ylabel("MW")

    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.13, 1.2),
        # ncol=3,
        frameon=False,
        borderaxespad=0
        )

    fig.savefig(path_out / f'{file_title}_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= Generatordaten =========
def get_gen_data(n, path_out):
    gen_list = (n.generators.bus.to_frame('bus')
                .join(n.generators[['efficiency', 'p_max_pu', 'p_nom', 'p_nom_opt']])
                .join(((n.generators.p_nom_opt.sort_index() - n.generators.p_nom.sort_index()).rename('expansion_mw')))
                .join(n.generators[['p_nom_extendable', 'carrier', 'marginal_cost', 'capital_cost','overnight_cost']])
                .join((n.generators.capital_cost * (n.generators.p_nom_opt - n.generators.p_nom)).rename('expansion_cost_EUR'))
                .join(n.generators_t.p.sum().rename('energy_2045_MWh'))
                .join((n.generators_t.p.sum() / n.generators.p_nom_opt).rename('full_load_hours_h'))
                )
    gen_carrier_list = (
                gen_list.p_nom_opt.groupby(gen_list.carrier).sum().to_frame('p_nom_opt_mw')
                .join(gen_list.p_nom.groupby(gen_list.carrier).sum().rename('p_nom_mw'))
                .join(gen_list.expansion_mw.groupby(gen_list.carrier).sum())
                .join(gen_list.carrier.value_counts().rename('n_generators'))
                .join(gen_list.marginal_cost.groupby(gen_list.carrier).mean().rename('marginal_cost_€_mw'))
                .join(gen_list.capital_cost.groupby(gen_list.carrier).mean().rename('capital_cost_€'))
                .join(gen_list.overnight_cost.groupby(gen_list.carrier).mean().rename('overnight_cost_€'))
                .join(gen_list.expansion_cost_EUR.groupby(gen_list.carrier).sum())
                .join((gen_list.energy_2045_MWh.groupby(n.generators.carrier).sum()))
                .join(gen_list.full_load_hours_h.groupby(n.generators.carrier).sum())
                )
    gen_carrier_list = gen_carrier_list.join((n.carriers["color"]).loc[gen_carrier_list.index])
    # gen_list.to_csv(path_out / f'Generatoren_{n.meta['run']['prefix']}.csv', index = True)
    # gen_carrier_list.to_csv(path_out / f'Generatortechnologien_{n.meta['run']['prefix']}.csv', index = True)
    return (gen_list, gen_carrier_list)
# ========= Meritorder =========
def merit_order(n, gen_list, path_out):
    # Sortieren für Meritorder
    df_merit = (gen_list.sort_values('marginal_cost',ascending=True).marginal_cost.to_frame()
            .join(gen_list.sort_values('marginal_cost',ascending=True).p_nom_opt.cumsum().rename('p_nom_opt_cum'))
            .join(gen_list[['carrier', 'p_nom_opt']]))

    # df_merit.to_csv(path_out / f'Meritorder_{n.meta['run']['prefix']}.csv', index = True)

    # Balken-Start (linke Kante)
    df_merit["left_MW"] = df_merit["p_nom_opt_cum"] - df_merit["p_nom_opt"]

    # --------- Energieart + Farben ---------
    carrier_map = carrier_key()              # dict: carrier -> Energieart
    df_merit["energy_type"] = df_merit["carrier"].map(carrier_map)

    energy_colors = color_key(n)               # dict: Energieart -> HEX
    colors = df_merit["energy_type"].map(energy_colors).fillna("#999999").tolist()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(
        df_merit["left_MW"] / 1000.0,           # Start in GW
        df_merit["marginal_cost"],              # Höhe in €/MWh
        width=df_merit["p_nom_opt"] / 1000.0,   # Breite in GW (Mengenachse)
        align="edge",
        color=colors
    )

    ax.set_xlabel("Kummulierte Kapazität (GW)")
    ax.set_ylabel("Grenzkosten (€/MWh)")
    ax.set_title("Meritorder")

    # Maximalwert der kumulierten Leistung in GW
    x_max = df_merit["p_nom_opt_cum"].iloc[-1] / 1000.0

    # Sinnvolle Ticks (z.B. 8 Schritte => 9 Werte inkl. 0)
    ticks = np.linspace(0, x_max, 9)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks])

    # --------- Legende: Energiearten (statt carrier) ---------
    # feste Reihenfolge (optional), nur vorhandene anzeigen
    legend_order = [
        "Braunkohle", "Kohle", "Gas", "Öl",
        "Kernenergie",
        "Wind", "Solar", "Wasserkraft", "Biomasse"
    ]
    present = set(df_merit["energy_type"].dropna().unique())

    handles = [
        Patch(facecolor=energy_colors.get(e, "#999999"), label=e)
        for e in legend_order
        if e in present
    ]

    ax.legend(
        handles=handles,
        title="Energieart",
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    fig.savefig(path_out / f'Meritorder_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= %-EE-Jahresenergie (Erzeugt) =========
def make_cake_dia(n, data, path_out, title):
    energy_pc = data / data.sum() # Prozente rechnen

    data = data.dropna()
    data = data[data > 0]
    data = data[energy_pc > 0.001] # Nur Werte über 0.1 %

    colors = [color_key(n)[e] for e in data.index]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.pie(
        data,
        labels=data.index,
        autopct="%1.2f%%",
        startangle=90,
        colors=colors
    ) 
    plt.title(f"{title} {n.meta['scenario']['planning_horizons'][0]}\nGesamterzeugung: {data.sum() / 1e6:.1f} [TWh]")
    plt.ylabel("")
    
    fig.savefig(path_out / f'{title}_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)

# ========= Plotdaten für Zeitaussschnitt auswahl =========
def data_for_time_plot(n):
    carrier_time = n.generators_t.p.T.groupby(n.generators.carrier).sum().groupby(carrier_key()).sum().T.copy()
    data_pos_load = (
        (n.loads_t.p.T.sum() * 1).to_frame('Gesamtlast')
            .join(((carrier_time.Wind)))
            .join((carrier_time.T.sum().T).rename('Generatoreinspeisung'))
            )
    data_neg_load = (
        (n.loads_t.p.T.sum() * -1).to_frame('Gesamtlast')
            .join(((carrier_time.Wind).rename('Windeinspeisung')))
            .join((carrier_time.T.sum().T).rename('Generatoreinspeisung'))
            )
    return data_pos_load, data_neg_load
# ========= AC-Daten =========
def get_ac_data(n):
    if n.lines_t.q0.empty == True: # Prüfung, ob Blindleistung mit Eingerichtet wurde
        S = n.lines_t.p0.abs() # |p0| wenn Q≈0 -- abs() -- betrag der Wirkleistung
    else:
        S = (n.lines_t.p0 ** 2 + n.lines_t.q0 ** 2) ** 0.5

    loading_max = ((S.max() # Maximalwert!
        .sort_index() #sort_index() im Nenner & im Zähler! -- richtige Reihenfolge
          / (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index() # .s_max_pu -- Zulässiger Anteil von s_nom_opt
           ).fillna(0.0))
    
    loading_mean = ((
        S.mean() # Mittelwert!!
        .sort_index() # sort_index() im Nenner & im Zähler! -- richtige Reihenfolge
          / 
          (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index() # .s_max_pu -- Zulässiger Anteil von s_nom_opt
           ).fillna(0.0))

    list_lines = (n.lines.bus0.to_frame('bus0')
              .join(n.lines[["bus1", "s_nom", "s_nom_opt", "capital_cost", "overnight_cost", "length"]])
              .join(loading_max.rename('max_loading'))
              .join(loading_mean.rename('mean_loading'))
              .join((n.lines.s_nom_opt - n.lines.s_nom).rename('expansion_mw'))
              .join((n.lines.overnight_cost * (n.lines.s_nom_opt - n.lines.s_nom)).rename('expansion_cost_EUR'))
        )
    
    list_lines=list_lines.join(((list_lines.expansion_cost_EUR.sort_values(ascending=False) / list_lines.expansion_cost_EUR.sum()) * 100).rename('pc_expensions_cost')) # pc-Kosten des Ausbaus
    return list_lines
# ============================================== MAIN ==============================================
def main():
    # ===== PFAD =====
    # path_row = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    # path_in = Path(path_row)
    path_in = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_2037_DC_N_S\KN2045_Elek\networks\base_s_all_elec_.nc") # Auskommentieren, wenn fertig
    # ===== Netzwerk einlesen =====
    n = read_network(path_in)
    # ===== Ablageordner erstellen =====
    path_out = build_folder(n, 'analysis_final', path_in)
    # ===== Netzwerkdaten =====
    get_metadata(n, path_out, path_in)
    plot_network(n, path_out)
    network_components(n, path_out)
    # ===== Diconary für RUN anlegen =====
    run_dic = {}
    # ===== Generatoren =====
    gen_list, gen_carrier_list = get_gen_data(n, path_out)
    merit_order(n, gen_list, path_out)
    data_cake = gen_carrier_list.groupby(carrier_key()).sum().energy_2045_MWh
    make_cake_dia(n, data_cake, path_out, 'Energieerzeugung')
    to_run_dic(run_dic, 'Gesamtenergie [MWh]', 'Generator', (gen_list.energy_2045_MWh.sum()))
    to_run_dic(run_dic,  'Investitionskosten [€]', 'Generator', gen_list['expansion_cost_EUR'].sum())
    to_run_dic(run_dic, 'Installierte Leistung [MW]', 'Generator', gen_list.expansion_mw.sum())
    # ===== Lasten =====
    to_run_dic(run_dic, 'Gesamtenergie [MWh]', 'Last', (n.loads_t.p.sum().sum()*-1))
    # ===== Diagram zur Wahl des Zeitausschnitts =====
    data_pos_load, data_neg_load = data_for_time_plot(n)
    time_plot_n_axses(n, data_pos_load, path_out, 'Einspeisung_Wind_Last_2045', '2045') # 2045
    time_plot_n_axses(n, data_pos_load.loc['2013-12'], path_out, file_title='Einspeisung_Wind_Last_12_2045', time_period=('12-2045'))
    # time_plot_1_axes(n, data_neg_load, path_out, file_title='Last_vs_Wind')
    # data.loc["2013-01-01":"2013-01-31"]
    # data.loc["2013-01-01 00:00:00":"2013-01-01 09:00:00"]
    # data.loc[pd.Timestamp("2013-01-01 13:00")]
    # data.resample('D').mean()
    # ===== AC-Leitungen =====
    ac_lines = get_ac_data(n)
    to_run_dic(run_dic, 'Investitionskosten [€]', 'AC-Leitungen', ac_lines.expansion_cost_EUR.sum())
    to_run_dic(run_dic, 'Installierte Leistung [MW]', 'AC-Leitungen', ac_lines.expansion_mw.sum()) # Mit Julian besprechen
    to_run_dic(run_dic, 'Verluste [MWh]', 'AC-Leitungen', n.lines_t.loss.sum().sum())
    # ===== DC-Leitungen =====






















    # ===== Diconary für Energy speichern =====
    save_run_dic(n, run_dic, path_out)
if __name__ == "__main__":
    main()
    print('\nFinsih')