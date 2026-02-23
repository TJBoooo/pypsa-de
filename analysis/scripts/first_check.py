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

# ========= Zeitreihen auspacken =========
def export_component_t(path_secion, component, title_cap, network):
    export_path = (path_secion / 'Zeitreihen')
    export_path.mkdir(parents=True, exist_ok=True)
    prefix = network.meta['run']['prefix']
    name = network.meta['run']['name']

    key_full = []
    for field in component.keys():
        df = getattr(component, field)
        if df.empty == False:
            key_full.append(True)
            df.to_csv(export_path / f'{field}_{prefix}_{name}.csv', index = False)
        else: 
            key_full.append(False)
    
    pd.DataFrame(component.keys()).join(pd.Series(key_full).sort_index().rename('full?')).to_csv(export_path / f'Time_Keys_{title_cap}_{prefix}_{name}.csv', index = True)
    return 

# ========= Auslastungsplots auslagen! =========
def statistic_plot(path_section, title, loading, prefix, name):

    loading_pct = loading * 100

    stats = loading_pct.describe()

    text = (
        f"N = {int(stats['count'])}\n"
        f"Mean = {stats['mean']:.2f}\n"
        f"Std = {stats['std']:.2f}\n"
        f"Min = {stats['min']:.2f}\n"
        f"25% = {stats['25%']:.2f}\n"
        f"Median = {stats['50%']:.2f}\n"
        f"75% = {stats['75%']:.2f}\n"
        f"Max = {stats['max']:.2f}"
    )

    ax = loading_pct.plot.hist(bins=100)

    ax.axvline(stats['mean'], linestyle="--", label="Mean")
    ax.axvline(stats['50%'], linestyle="-", label="Median")
    ax.axvline(stats['75%'], linestyle=":", label="75% Quantil")
    ax.axvline(70, color="red", linewidth=2, label="70% Schwelle")

    ax.legend()

    ax.text(
        0.25, 0.7, text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    ax.set_ylabel('N')
    ax.set_xlabel('%')
    ax.set_title(title)

    # ax.figure.savefig(path_section / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
    ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(ax.figure)

    return 

# ============================================== MAIN ==============================================
def main():
    # =========================== Wo liegt das Netzwerk / die .nc-Datei? ===========================
    # path_row = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    # path_in = Path(path_row)
    path_in = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_Referenzoptimierung\KN2045_Elek\networks\base_s_all_elec_.nc") # Auskommentieren, wenn fertig
    
    # =========================== Netzwerk Laden ===========================
    print("\nLade Netzwerk …\n")
    n = pypsa.Network(path_in)
    prefix = n.meta['run']['prefix']
    name = n.meta['run']['name']
    print(f'\nTitel: {prefix} {name} \n')

    # =========================== Zielordner erstellen & Pfad zwischenspeichern ===========================
    base = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de/results") #\analysis
    path = base / prefix / name / 'analysis' / f'c_{n.meta['scenario']['clusters']}' # Def for later
    path.mkdir(parents=True, exist_ok=True) # parents --> fehlende Ordner automatisch anlegen, exist_ok --> Kein Fehler, wenn Order schon da
    print('Results:', path, '\n')

    # =========================== Netzwerkdaten erzeugen & ablegen ===========================
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
            {"metric": "cluster", "value": n.meta['scenario']['clusters']},
            {"metric": "planning_horizons", "value": n.meta['scenario']['planning_horizons']},
        ]

    overview = pd.DataFrame(rows)  # Array in Dataframe überführen
    overview.to_csv(path / f'overview_{prefix}_{name}.csv')

    # =========================== Allgemeiner erster Plot ===========================
    title = 'Karte_Europa'

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 8))
    fig.suptitle(f"{title} | {prefix} | {name} | c: {n.meta['scenario']['clusters']}", fontsize=12, fontweight="bold")
    n.plot(ax=ax,geomap_color=True,bus_size=0.001)
    # fig.savefig(path / f"{title}_{prefix}_{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(path / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(fig)

    # =========================== Netzwerkcomponenten ===========================
    title = 'components'
    components = network_components(n)
    # print("\nAnzahl der Komponenten:\n", components)
    components.to_csv(path / f'{title}_{prefix}_{name}.csv', index = False)


    # =========================== Energiebilanz ===========================
    title = 'Energiebilanz'
    energy_balance = (n.statistics.energy_balance()
                      .reset_index(name = 'energiy_MWh')
                      )
    energy_balance.to_csv(path / f'{title}_{prefix}_{name}.csv', index = False)

    # =========================== Generators ===========================
    title_cap = 'Generatoren_(Generators)'
    # Ordner erstellen
    path_section = path / f'{title_cap}'
    path_section.mkdir(parents=True, exist_ok=True)
    # ===================
    title = 'Verwendete_Generatoren'
    n.generators.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    export_component_t(path_section, n.generators_t, title_cap, n)
    # ===================
    title = f'Auswertung_{title_cap}'
    gen_list = (n.generators.bus.to_frame('bus')
                .join(n.generators[['efficiency', 'p_max_pu', 'p_nom', 'p_nom_opt']])
                . join(((n.generators.p_nom_opt.sort_index() - n.generators.p_nom.sort_index()).rename('extendable_mw')))
                .join(n.generators[['p_nom_extendable', 'carrier', 'marginal_cost', 'capital_cost','overnight_cost']])
                )
    gen_list.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True) 
    # ===================
    title = f'Auswertung_Carrier_{title_cap}'
    carrier_list = (
                    gen_list.p_nom_opt.groupby(gen_list.carrier).sum().to_frame('p_nom_opt_mw')
                    .join(gen_list.p_nom.groupby(gen_list.carrier).sum().rename('p_nom_mw'))
                    .join(gen_list.extendable_mw.groupby(gen_list.carrier).sum().rename('extendeble_mw'))
                    .join(gen_list.carrier.value_counts().rename('n_generators'))
                    .join(gen_list.marginal_cost.groupby(gen_list.carrier).mean().rename('marginal_cost_€_mw'))
                    .join(gen_list.capital_cost.groupby(gen_list.carrier).mean().rename('capital_cost_€'))
                    .join(gen_list.overnight_cost.groupby(gen_list.carrier).mean().rename('overnight_cost_€'))
                    )
    carrier_list.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True) 
    # ===================
    title = 'Generatorkapazitäten_pro_Technologie'
    df = carrier_list.sort_values('p_nom_mw', ascending = False)
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
    df["extendeble_mw"].plot.bar(ax=axes[2])
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
    # fig.savefig(path_section / f"{title}t_{prefix}_{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(fig)
    # ===================
    title = "Einspeisung_Energiearten"
    p_per_tech = (n.generators_t.p.T.groupby(n.generators.carrier).sum().T)
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
    # p_per_energytyp_sorted.to_csv(path_section / f'{title}_{prefix}_{name}.csv')
    # Diagrammeinstellungen
    ax = p_per_energytyp_sorted.plot.area(figsize=(20, 8), title=title)
    ax.set_xlabel('t') # Diagrammbeschriftung
    ax.set_ylabel("Power [p | MW]") # Diagrammbeschriftung
    # Ploteinstellungen
    ax.figure.suptitle(f"Auflösung: Wöchentlich", fontsize=12, fontweight="bold")
    # ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight") # Plot speichern
    ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(ax.figure)
    # ===================
    title = f'Einspeisung_gesammt_{title_cap}'
    ax = n.generators_t.p.resample('W').mean().sum(axis=1).plot.area(figsize=(15,3)) # Gesamteinspeisung
    # Ploteinstellungen
    ax.figure.suptitle(f"Auflösung: Wöchentlich", fontsize=12, fontweight="bold")
    # ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight") # Plot speichern
    ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(ax.figure)

    # =========================== Lines / Leitungen ===========================
    title_cap = 'AC-Leitungen_(Lines)'
    # Ordner erstellen
    path_section = path / f'{title_cap}'
    path_section.mkdir(parents=True, exist_ok=True)
    # ===================
    title = f'Verwendete_{title_cap}'
    n.lines.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.lines_t, title_cap, n)
    # ===================
    title = f'Statistic_max_Auslastung_in_%_{title_cap}'
    if n.lines_t.q0.empty == True: # Prüfung, ob Blindleistung mit Engerichtet wurde
        S = n.lines_t.p0.abs() # |p0| wenn Q≈0 -- abs() -- betrag der Wirkleistung
    else:
        S = (n.lines_t.p0 ** 2 + n.lines_t.q0 ** 2) ** 0.5
    loading_max = ((S.max() # Maximalwert!
        .sort_index() # mean() arithmetisches Mittel aller Zeitwerte -- sort_index() im Nenner & im Zähler! -- richtige Reihenfolge
          / (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index() # .s_max_pu -- Zulässiger Anteil von s_nom_opt
           ).fillna(0.0))
    statistic_plot(path_section, title, loading_max, prefix, name)
    # ===================
    title = f'Statistic_mittlere_Auslastung_in_%_{title_cap}'
    loading_mean = ((
        S.mean() # Mittelwert!!
        .sort_index() # mean() arithmetisches Mittel aller Zeitwerte -- sort_index() im Nenner & im Zähler! -- richtige Reihenfolge
          / 
          (n.lines.s_nom_opt * n.lines.s_max_pu).sort_index() # .s_max_pu -- Zulässiger Anteil von s_nom_opt
           ).fillna(0.0)) 
    statistic_plot(path_section, title, loading_mean, prefix, name)
    # ===================
    title = f'Auswertungen_{title_cap}'
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
    list_lines.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    title = f'Ausbau_{title_cap}'

    min_mw_ac = 0.5 # Rauschen rausnehmen

    head_expension = (list_lines.expension_mw > min_mw_ac).sum() # Anzahl stark ausgebauten Leitungen
    if head_expension > 0:
        ax = ((list_lines.expension_mw.sort_values(ascending=False)).head(head_expension)).plot.bar(figsize=(10,5))
        sum = ((list_lines.expension_mw.sort_values(ascending=False)).head(head_expension)).sum()
        ax.figure.suptitle(f'Gesamtausbau: {sum:.2f} MW')
        ax.set_ylabel('MW')
        ax.set_title(title)
        # ax.figure.savefig(path_section / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
        ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
        plt.close(ax.figure)
    else: print('Kein AC-Links mit Ausbau gefunden.')
    # ===================
    title = f'Karte_Ausbau_{title_cap}'
    # --- Figure / Projektion ---
    fig, ax = plt.subplots(
        figsize=(20, 20),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent([-10, 30, 35, 65], crs=ccrs.PlateCarree())
    # --- Auswahl: ausgebaute Leitungen (rot) ---
    expanded_lines = list_lines.expension_mw.sort_values(ascending=False).head(head_expension).index
    # --- Farben/Breiten ---
    line_colors = pd.Series("#e6a0a0", index=n.lines.index)   # Original hellrot
    line_colors.loc[expanded_lines] = "red"                   # Ausgebaut rot
    line_widths = pd.Series(0.5, index=n.lines.index)
    line_widths.loc[expanded_lines] = 3
    # --- Plot ---
    n.plot(
        ax=ax,
        branch_components=["Line", "Link"],
        bus_colors="gray",
        bus_size=0.001,
        line_widths=line_widths,
        line_colors=line_colors,
        geomap_color=True,
        # bus_sizes=0.02
    )
    # --- Bus-Beschriftung: Buses der ausgebauten Leitungen ---   ---> Könnte uahc nochmal überarbeitet werden.!!
    buses_to_label = pd.Index(
        n.lines.loc[expanded_lines, ["bus0", "bus1"]].values.ravel()
    ).unique()
    for bus in buses_to_label:
        if bus in n.buses.index:
            row = n.buses.loc[bus]
            label = str(bus).replace(" battery", "").replace(" H2", "")
            ax.text(
                row.x, row.y, label,
                transform=ccrs.PlateCarree(),
                fontsize=13,  # 17 geht, wird aber schnell groß
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                zorder=10
            )

        ax.set_title(
        f"{title} | {prefix} | {name}",
        fontsize=18,
        fontweight="bold"
    )  
    ax.axis("off")
    # fig.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
    fig.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(fig)
    # ===================
    title = f' Ausbaukosten_{title_cap}'
    if head_expension > 0:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        # --- Balken: Kostenanteil ---
        ax1.bar(
            np.arange(len(list_lines.expensions_cost.sort_values(ascending=False).head(head_expension))),
            list_lines.expensions_cost.sort_values(ascending=False).head(head_expension).values,
            color="steelblue",
            alpha=0.8
        )
        # --- zweite y-Achse: kumuliert ---
        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(len(list_lines.expensions_cost.sort_values(ascending=False).head(head_expension))),
            (list_lines.pc_expensions_cost.sort_values(ascending=False).head(head_expension).cumsum()).values,
            color="darkred",
            marker="o"
        )
        # --- x-Achse: Labels als Leitungs-IDs ---
        ax1.set_xticks(np.arange(len(list_lines.expensions_cost.sort_values(ascending=False).head(head_expension))))
        ax1.set_xticklabels(list_lines.expensions_cost.sort_values(ascending=False).head(head_expension).index.astype(str), rotation=45)
        # --- Vereinheitlichung beider y-Achsen ---
        ax1.set_ylabel("Ausbauvolumen je AC-Leitung [€]")
        ax2.set_ylabel("Kumulierte Kosten [%]")
        ax1.set_xlabel("Leitung (ID)")
        left_max = 1.05 * list_lines.expensions_cost.sort_values(ascending=False).head(head_expension).max()
        right_max = 105
        n_ticks = 6  # gleiche Anzahl!
        ax1.set_ylim(0, left_max)
        ax2.set_ylim(0, right_max)
        ax1.set_yticks(np.linspace(0, left_max, n_ticks))
        ax2.set_yticks(np.linspace(0, right_max, n_ticks))
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        # --- Titel ---
        ax1.set_title(f"Ausbaukosten Leitungen\nGesamtausbau: {list_lines.expensions_cost.sum():,.2f} €")

        plt.tight_layout()
        # fig.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
        fig.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
        plt.close(fig)
    else: print('Keine Ausbaukosten für AC-Leitungen')
    # ===================
    title = f'Auslastung_mean_EU_{title_cap}'

    loading_min_ac = 0.1 # %-Auslastung

    head_loading = (list_lines.mean_loading > loading_min_ac).sum() # Liste!!
    if head_loading != 0:
        if head_loading > 20:
            head_loading = 20

        ax = ((list_lines["mean_loading"].sort_values(ascending=False)
            .head(head_loading).to_frame(name="mean_loading")) * 100).plot.bar(figsize=(10, 5))
        ax.set_ylabel("Mittlere Auslastung [%]")
        ax.set_xlabel("Leitung (ID)")
        ax.set_title(f"{title} | Mindestauslastung [%]: {loading_min_ac}")
        # ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
        ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
        plt.close(ax.figure)

    else: 
        print('loading_min_ac zu groß')
    # ===================
    title = f'Karte_Auslastung_{title_cap}'
    # Karte ansehnlicher machen!
    fig, ax = plt.subplots(
        figsize=(20, 20),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent([-10, 30, 35, 65], crs=ccrs.PlateCarree())
    # --- Plot ---
    n.plot(
        ax=ax,
        branch_components=["Line", "Link"],
        bus_colors="gray",
        bus_size=0.001,
        line_widths=(n.lines.s_nom_opt / n.lines.s_nom_opt.max()) * 20 + 5,
        line_colors=(loading_mean.clip(0, 1.2) * 100),
        line_cmap=plt.cm.RdYlGn_r,   # grün → gelb → rot
        geomap_color=True,
        # bus_sizes=0.02
    )
    ax.axis("off")
    # --- Colorbar (nur für AC-Auslastung) ---
    norm = mpl.colors.Normalize(vmin=0, vmax=120)
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Mittlere Leitungsauslastung [%] (AC)")
    # --- Empfohlene Bus-Beschriftung: nur Top-10 kritische Leitungen ---
    top_lines = loading_mean.sort_values(ascending=False).head(10).index
    buses_to_label = pd.Index(
        n.lines.loc[top_lines, ["bus0", "bus1"]].values.ravel()
    ).unique()

    for bus in buses_to_label:
        if bus in n.buses.index:
            row = n.buses.loc[bus]
            ax.text(
                row.x, row.y, str(bus),
                transform=ccrs.PlateCarree(),
                fontsize=17,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

    ax.set_title(
        f"{title} | {prefix} | {name}",
        fontsize=18,
        fontweight="bold"
    )
    ax.axis("off")
    # fig.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
    fig.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(fig)


    # =========================== Links ===========================
    title_cap = 'Links'
    # Ordner erstellen
    path_section = path / f'{title_cap}'
    path_section.mkdir(parents=True, exist_ok=True)
    # ===================
    title = f'Verwendete_{title_cap}'
    n.links.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.links_t, title_cap, n)
    # ===================
    title = f'Auswertungen_{title_cap}'
    loading_max = n.links_t.p1.abs().sort_index().max() / (n.links.p_nom_opt * n.links.p_max_pu).sort_index()
    loading_mean = n.links_t.p1.abs().sort_index().mean() / (n.links.p_nom_opt * n.links.p_max_pu).sort_index()
    list_links = (
            n.links.bus0.to_frame('bus0')
            .join(n.links[["bus1", "carrier", "p_nom" ,"p_nom_opt","capital_cost"]])
            .join(loading_max.rename('max_loading'))
            .join(loading_mean.rename('mean_loading'))
            .join(((n.links['p_nom_opt'] - n.links['p_nom'])).rename('expension_mw'))
            .join(n.links[["overnight_cost"]])
    )
    list_links = list_links.join((list_links.overnight_cost.sort_values(ascending=False) * list_links.expension_mw.sort_values(ascending=False)).rename('expensions_cost')) # Kosten des Ausbaus
    list_links.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    title_undercap = 'DC-Leitungen_(Links)'
    # Ordner erstellen
    path_undersection = path_section / f'{title_undercap}'
    path_undersection.mkdir(parents=True, exist_ok=True)
    # ===================
    title = f'Vernwendete_{title_undercap}'
    list_dc = list_links.loc[list_links['carrier'] == 'DC']
    list_dc = list_dc.join(((list_dc.expensions_cost.sort_values(ascending=False) / list_dc.expensions_cost.sum()) * 100).rename('pc_expensions_cost')) # pc-Kosten des Ausbaus
    list_dc.to_csv(path_undersection / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    title = f'Statistic_max_Auslastung_in_%_{title_undercap}'
    loading_max = list_dc.max_loading
    statistic_plot(path_undersection, title, loading_max, prefix, name)
    # ===================
    title = f'Statistic_mittlere_Auslastung_in_%_{title_undercap}'
    loading_mean = list_dc.mean_loading
    statistic_plot(path_undersection, title, loading_mean, prefix, name)
    # ===================
    title = f'Ausbau_{title_undercap}'

    min_mw_dc = 0.5 # MW Rauschen rausnehmen

    head_expension = (list_dc.expension_mw > min_mw_dc).sum() # Anzahl stark ausgebauten Leitungen
    if head_expension > 0:
        ax = ((list_dc.expension_mw.sort_values(ascending=False)).head(head_expension)).plot.bar(figsize=(10,5))
        sum = ((list_dc.expension_mw.sort_values(ascending=False)).head(head_expension)).sum()
        ax.figure.suptitle(f'Gesamtausbau: {sum:.2f} MW')
        ax.set_ylabel('MW')
        ax.set_title(title)
        # ax.figure.savefig(path_undersection / f'{title}_{prefix}_{name}', dpi=300, bbox_inches="tight") # Plot speichern
        ax.figure.savefig(path_undersection / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
        plt.close(ax.figure)
    else: print('Kein DC-Links mit Ausbau gefunden.')
    # ===================
    title = f' Ausbaukosten_{title_undercap}'
    if head_expension > 0:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        # --- Balken: Kostenanteil ---
        ax1.bar(
            np.arange(len(list_dc.expensions_cost.sort_values(ascending=False).head(head_expension))),
            list_dc.expensions_cost.sort_values(ascending=False).head(head_expension).values,
            color="steelblue",
            alpha=0.8
        )
        # --- zweite y-Achse: kumuliert ---
        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(len(list_dc.expensions_cost.sort_values(ascending=False).head(head_expension))),
            (list_dc.pc_expensions_cost.sort_values(ascending=False).head(head_expension).cumsum()).values,
            color="darkred",
            marker="o"
        )

        # --- x-Achse: Labels als Link-IDs ---
        ax1.set_xticks(np.arange(len(list_dc.expensions_cost.sort_values(ascending=False).head(head_expension))))
        ax1.set_xticklabels(list_dc.expensions_cost.sort_values(ascending=False).head(head_expension).index.astype(str), rotation=45)

        # -- Vereinheitlichung beider y-Achsen --
        ax1.set_ylabel("Ausbauvolumen je DC-Link [€]")
        ax2.set_ylabel("Kumulierte Kosten [%]")
        ax1.set_xlabel("Link (ID)")

        left_max = 1.05 * list_dc.expensions_cost.sort_values(ascending=False).head(head_expension).max()
        right_max = 105
        n_ticks = 6

        ax1.set_ylim(0, left_max)
        ax2.set_ylim(0, right_max)

        ax1.set_yticks(np.linspace(0, left_max, n_ticks))
        ax2.set_yticks(np.linspace(0, right_max, n_ticks))

        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # --- Titel ---
        ax1.set_title(f"Ausbaukosten DC-Links\nGesamtausbau: {list_dc.expensions_cost.sum():,.2f} €")

        plt.tight_layout()
        # fig.savefig(path_undersection / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
        fig.savefig(path_undersection / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
        plt.close(fig)
    else: print('Kein Ausbaukosten für DC-Leitungen')
    # ===================
    title = f'Ausbau_Karte_{title_undercap}'
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-10, 30, 35, 65], crs=ccrs.PlateCarree())
    expanded_links = list_dc.expension_mw.sort_values(ascending=False).head(head_expension).index
    link_colors = pd.Series("#3DB44B", index=n.links.index)
    link_colors.loc[expanded_links] = "red"
    link_widths = pd.Series(0.5, index=n.links.index)
    link_widths.loc[expanded_links] = 3
    n.plot(
        ax=ax,
        branch_components=["Line","Link"],
        bus_colors="gray",
        bus_size=0.001,
        link_widths=link_widths,
        link_colors=link_colors,
        geomap_color=True,
    )
    buses_to_label = pd.Index(n.links.loc[expanded_links, ["bus0", "bus1"]].values.ravel()).unique()
    for bus in buses_to_label:
        if bus in n.buses.index:
            row = n.buses.loc[bus]
            label = str(bus).replace(" battery", "").replace(" H2", "")
            ax.text(row.x, row.y, label, transform=ccrs.PlateCarree(),
                    fontsize=13, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    zorder=10)

    ax.set_title(f"{title} | {prefix} | {name}", fontsize=18, fontweight="bold")
    ax.axis("off")
    # fig.savefig(path_undersection / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
    fig.savefig(path_undersection / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(fig)
    # ===================
    title = f'Auslastung_mean_EU_{title_undercap}'

    loading_min_dc = 0.1 # %-Auslastung

    head_loading = (list_dc.mean_loading > loading_min_dc).sum()
    if head_loading != 0:
        if head_loading > 20:
            head_loading = 20

        ax = ((list_dc["mean_loading"].sort_values(ascending=False)
            .head(head_loading).to_frame(name="mean_loading")) * 100).plot.bar(figsize=(10, 5))
        ax.set_ylabel("Mittlere Auslastung [%]")
        ax.set_xlabel("Leitung (ID)")
        ax.set_title(f"{title} | Mindestauslastung [%]: {loading_min_dc}")
        # ax.figure.savefig(path_undersection / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
        ax.figure.savefig(path_undersection / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
        plt.close(ax.figure)
    
    else: 
        print('loading_min_dc zu groß')
    # ===================
    title = f'Auslastung_mean_Karte_{title_undercap}'
    fig, ax = plt.subplots(
        figsize=(20, 20),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent([-10, 30, 35, 65], crs=ccrs.PlateCarree())
    n.plot(
        ax=ax,
        branch_components=["Line", "Link"],
        bus_colors="gray",
        bus_size=0.001,
        link_widths=(list_dc.p_nom_opt / list_dc.p_nom_opt.max()) * 20 + 5, # Breite im Vergleich zur breitesten Linie
        link_colors=(((list_dc.mean_loading).clip(0, 1.2)) * 100),
        link_cmap=plt.cm.RdYlGn_r,   # grün → gelb → rot
        geomap_color=True,
    )
    ax.axis("off")
    norm = mpl.colors.Normalize(vmin=0, vmax=120)
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Mittlere Leitungsauslastung [%] (DC)")

    top_links = (list_dc.mean_loading).sort_values(ascending=False).head(head_loading).index
    buses_to_label = pd.Index(
        n.links.loc[top_links, ["bus0", "bus1"]].values.ravel()
    ).unique()

    for bus in buses_to_label:
        if bus in n.buses.index:
            row = n.buses.loc[bus]
            ax.text(
                row.x, row.y, str(bus),
                transform=ccrs.PlateCarree(),
                fontsize=17,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )
    ax.set_title(
        f"{title} | {prefix} | {name}",
        fontsize=18,
        fontweight="bold"
    )
    ax.axis("off")
    # fig.savefig(path_undersection / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight")
    fig.savefig(path_undersection / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(fig)

    # =========================== Lasten_(Loads) ===========================
    title_cap = 'Lasten_(Loads)'
    # Ordner erstellen
    path_section = path / f'{title_cap}'
    path_section.mkdir(parents=True, exist_ok=True)
    # ===================
    title = f'Verwendete_{title_cap}'
    n.loads.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)
    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.links_t, title_cap, n)
    # ===================
    #Time-varying component data
    title = 'n.loads'
    n.loads_t.p_set.sum(axis=1).to_csv(path_section / f"{title}_{prefix}_{name}.csv",index=True)
    # Diagrammeinstellungen & Speichern
    ax = n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3)) #MW?
    ax.set_ylabel("MW?")
    ax.set_title(title)
    # ax.figure.suptitle(f"Subtitle", fontsize=12, fontweight="bold")
    # ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.png', dpi=300, bbox_inches="tight") # Plot speichern
    ax.figure.savefig(path_section / f'{title}_{prefix}_{name}.svg', bbox_inches="tight")
    plt.close(ax.figure)

    # =========================== Speicher_(Storage_Units) ===========================
    title_cap = 'Speicher_(Storage_Units)'
    # Ordner erstellen
    path_section = path / title_cap
    path_section.mkdir(parents=True, exist_ok=True)

    # ===================
    title = f'Verwendete_{title_cap}'
    n.storage_units.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)

    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.storage_units_t, title_cap, n)

      # =========================== Stores ===========================
    title_cap = 'Stores'
    # Ordner erstellen
    path_section = path / title_cap
    path_section.mkdir(parents=True, exist_ok=True)

    # ===================
    title = f'Verwendete_{title_cap}'
    n.stores.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)

    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.stores_t, title_cap, n)
    
    # =========================== Sub_networks ===========================
    title_cap = 'Sub_networks'
    # Ordner erstellen
    path_section = path / title_cap
    path_section.mkdir(parents=True, exist_ok=True)

    # ===================
    title = f'Verwendete_{title_cap}'
    n.sub_networks.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)

    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.sub_networks_t, title_cap, n)

    # =========================== Buses ===========================
    title_cap = 'Knoten_(Buses)'
    # Ordner erstellen
    path_section = path / f'{title_cap}'
    path_section.mkdir(parents=True, exist_ok=True)

    # ===================
    title = f'Verwendete_{title_cap}'
    n.buses.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)

    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.buses_t, title_cap, n)

    # =========================== Carriers ===========================
    title_cap = 'Technologien_(Carriers)'
    # Ordner erstellen
    path_section = path / title_cap
    path_section.mkdir(parents=True, exist_ok=True)

    # ===================
    title = f'Verwendete_{title_cap}'
    n.carriers.to_csv(path_section / f'{title}_{prefix}_{name}.csv', index = True)

    # ===================
    # Auspacken der Zeitreihen
    export_component_t(path_section, n.carriers_t, title_cap, n)















if __name__ == "__main__":
    main()
    print('\nFinsih')