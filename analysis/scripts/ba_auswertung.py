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
    path = path_in.parents[1]/folder_name/f'c_{n.meta['scenario']['clusters'][0]}'
    # parents --> fehlende Ordner automatisch anlegen, exist_ok --> Kein Fehler, wenn Order schon da
    path.mkdir(parents=True, exist_ok=True) 
    print('Results:', path, '\n')
    return path
# ========= CSV-holen =========
def get_csv(path):
    path = Path(rf'{path}')
    return pd.read_csv(path, index_col=0)
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
def make_plot(
    n,
    title,
    path_out=None,
    pc_title=None,
    line_color=None,
    line_width=None,
    link_color=None,
    link_width=None,
    *,
    scale_width=0.0008,
    min_width=3,
    vmin=0,
    vmax=100,
    cmap=plt.cm.RdYlGn_r,
    norm=None,
    show_colorbar=True,
    line_style="-",
    link_style="--",          
    link_capstyle="round",   
):
    if norm is None:
        norm = mpl.colors.Normalize(vmin, vmax)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(20, 20)
    )

    plot_kwargs = dict(
        ax=ax,
        branch_components=["Line", "Link"],
        line_widths=min_width,
        link_widths=min_width,
        bus_size=0.001,
        geomap=True,
        geomap_color={"ocean": "lightblue", "land": "#E6E7D8F9"},
    )

    # ---- Optional Line Settings ----
    if line_color is not None:
        plot_kwargs.update(
            line_colors=(line_color * 100),
            line_cmap=cmap,
            line_cmap_norm=norm,
        )

    if line_width is not None:
        plot_kwargs.update(
            line_widths=(line_width * scale_width + min_width),
        )

    # ---- Optional Link Settings ----
    if link_color is not None:
        plot_kwargs.update(
            link_colors=(link_color * 100),
            link_cmap=cmap,
            link_cmap_norm=norm,
        )

    if link_width is not None:
        plot_kwargs.update(
            link_widths=(link_width * scale_width + min_width),
        )

    # ---- Plot ----
    coll = n.plot(**plot_kwargs)

    # ---- Styling nachträglich ----
    if "Line" in coll.get("branches", {}):
        line_coll = coll["branches"]["Line"]
        line_coll.set_linestyle(line_style)
        line_coll.set_capstyle(line_capstyle)

    if "Link" in coll.get("branches", {}):
        link_coll = coll["branches"]["Link"]
        link_coll.set_linestyle(link_style)
        link_coll.set_capstyle(link_capstyle)

    # ---- Colorbar ----
    if show_colorbar and (line_color is not None or link_color is not None):
        if pc_title is None:
            pc_title = "?!?!?!"

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.05)
        cbar.set_label(f"{pc_title}", fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    ax.set_title(
        f"{title} | {n.meta['run']['prefix']}",
        fontsize=18,
        fontweight="bold"
    )

    ax.axis("off")

    fig.savefig(path_out/f'{title}_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
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
    
    if isinstance(value, (int, float)):
        dic[category][name] = float(value)
    else:
        dic[category][name] = value
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
        "offwind-ac": "Offwind",
        "offwind-dc": "Offwind",
        "offwind-float": "Offwind",
        "oil": "Öl",
        "onwind": "Onwind",
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
        "Onwind":  n.carriers.color.get('onwind', '#999999'),
        "Offwind": n.carriers.color.get('offwind-ac', '#999999'),
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
def time_plot_1_axes(n, data_plot, path_out, file_title, time_period):

    # Farbmap holen (dict: name -> hex)
    colors = [color_key(n).get(col, '#999999') for col in data_plot.columns]
    linewidth = 0.8
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.suptitle(f"{n.meta['run']['prefix']}\nZeitverlauf: {time_period}", fontsize=22)
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
                .join(n.generators[['p_nom_extendable', 'carrier', 'marginal_cost', 'capital_cost']])
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
    ax.set_title(f"Meritorder {n.meta['run']['prefix']}")

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
        title="Energieträger",
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    fig.savefig(path_out / f'Meritorder_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= %-EE-Jahresenergie (Erzeugt) =========
def make_cake_dia(n, data, path_out, title): #Wind in off und on unterteilen!
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
    plt.title(f"{title} {n.meta['scenario']['planning_horizons'][0]}\n{n.meta['run']['prefix']}\nGesamterzeugung: {data.sum() / 1e6:.1f} [TWh]")
    plt.ylabel("")
    
    fig.savefig(path_out / f'{title}_{n.meta['run']['prefix']}.svg', bbox_inches="tight")
    plt.close(fig)
# ========= Plotdaten für Zeitaussschnitt auswahl =========
def data_for_time_plot(n):
    carrier_time = n.generators_t.p.T.groupby(n.generators.carrier).sum().groupby(carrier_key()).sum().T.copy()
    data_pos_load = (
        (n.loads_t.p.T.sum() * 1).to_frame('Gesamtlast')
            .join(((carrier_time.Onwind)))
            .join(((carrier_time.Offwind)))
            .join((carrier_time.T.sum().T).rename('Generatoreinspeisung'))
            )
    data_neg_load = (
        (n.loads_t.p.T.sum() * -1).to_frame('Gesamtlast')
            .join(((carrier_time.Onwind).rename('Onwindeinspeisung')))
            .join(((carrier_time.Offwind).rename('Offwindeinspeisung')))
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
              .join(n.lines[["bus1", "s_nom", "s_nom_opt", "capital_cost", "length"]])
            #   .join(loading_max.rename('max_loading'))
              .join(loading_mean.rename('mean_loading'))
              .join((n.lines.s_nom_opt - n.lines.s_nom).rename('expansion_mw'))
              .join((n.lines.overnight_cost * (n.lines.s_nom_opt - n.lines.s_nom)).rename('expansion_cost_EUR'))
              .join((n.lines_t.loss.sum()).rename('loss_mwh'))
        )
    # list_lines=list_lines.join(((list_lines.expansion_cost_EUR.sort_values(ascending=False) / list_lines.expansion_cost_EUR.sum()) * 100).rename('pc_expensions_cost')) # pc-Kosten des Ausbaus
    return list_lines
# ========= DC-Daten =========
def get_dc_data(n):
    loading_max = n.links_t.p1.abs().sort_index().max() / (n.links.p_nom_opt * n.links.p_max_pu).sort_index()
    loading_mean = n.links_t.p1.abs().sort_index().mean() / (n.links.p_nom_opt * n.links.p_max_pu).sort_index()
    list_links = (
            n.links.bus0.to_frame('bus0')
            .join(n.links[["bus1", "carrier", "p_nom" ,"p_nom_opt","capital_cost"]])
            # .join(loading_max.rename('max_loading'))
            .join(loading_mean.rename('mean_loading'))
            .join(((n.links['p_nom_opt'] - n.links['p_nom'])).rename('expansion_mw'))
    )
    list_links = list_links.join((list_links.capital_cost.sort_values(ascending=False) * list_links.expansion_mw.sort_values(ascending=False)).rename('expansion_cost_EUR')) # Kosten des Ausbaus
    list_dc = list_links.loc[list_links['carrier'] == 'DC']
    # list_dc = list_dc.join(((list_dc.expansion_cost_EUR.sort_values(ascending=False) / list_dc.expansion_cost_EUR.sum()) * 100).rename('pc_expensions_cost')) # pc-Kosten des Ausbaus (Nur DC-Leitungen)
    return list_dc
# ========= Leitungs-Filter Basisnetz und NEP (AC & DC) =========
def filter_connections(network_connection_raw, nep_connection_raw):
    nep_connections_in_network = network_connection_raw.loc[network_connection_raw.index.intersection(nep_connection_raw.index)]
    connections_in_network = network_connection_raw.drop(nep_connections_in_network.index)
    return connections_in_network, nep_connections_in_network
# ========= Neuberechnung der Ausgebauten Leistung & entstandenen Kosten Basisnetz und NEP (AC & DC) =========
def make_real_expansion_data(nep_connections_in_network, type):
    if type == 'AC':
        nep_connections_in_network['expansion_mw'] = nep_connections_in_network.s_nom_opt
        nep_connections_in_network['expansion_cost_EUR'] = nep_connections_in_network.s_nom_opt * nep_connections_in_network.capital_cost
        nep_connections_in_network['s_nom'] = 0
    if type =='DC':
        nep_connections_in_network['expansion_mw'] = nep_connections_in_network.p_nom_opt
        nep_connections_in_network['expansion_cost_EUR'] = nep_connections_in_network.p_nom_opt * nep_connections_in_network.capital_cost
        nep_connections_in_network['p_nom'] = 0
    return nep_connections_in_network
# ========= Basisnetz & NEP-Projekte zusammenführen =========
def two_to_one(list_1, list_2):
    if (list_1.index.intersection(list_2.index)).empty:
        list = pd.concat(
            [list_1, list_2],
            axis=0 # 0 --> Spalten werden zusammengeführt, 1 --> Zeilen werden drunter gehängt und Spalten werden neu geschrieben
        )
    else: print('Listen können nicht zusammengführt werden!')
    return list
# ========= Netzwerkkennzahlen ins Dic mit aufnehmen =========
def multiple_to_run_dic(run_dic, component, list): # component: Generator, AC-Leitungen (Basisnetz), AC-Leitung (NEP), DC-Leitungen (Basisnetz), DC-Leitung (NEP)
    if component == 'Generator':
        to_run_dic(run_dic,  'Investitionskosten [€]', 'Generator', list.expansion_cost_EUR.sum())
        to_run_dic(run_dic, 'Ausgebaute Leistung [MW]', 'Generator', list.expansion_mw.sum())
        to_run_dic(run_dic, 'Einspeisung / Verluste [MWh]', 'Generator', (list.energy_2045_MWh.sum()))
        to_run_dic(run_dic, 'Anzahl [N]', 'Generator', len(list.index))
    elif component == 'AC-Leitungen (Basisnetzwerk)':
        # AC-Leitungen die im Base-Netz waren (base_network)
        to_run_dic(run_dic, 'Investitionskosten [€]', 'AC-Leitungen (Basisnetzwerk)', list.expansion_cost_EUR.sum())
        to_run_dic(run_dic, 'Ausgebaute Leistung [MW]', 'AC-Leitungen (Basisnetzwerk)', list.expansion_mw.sum())
        to_run_dic(run_dic, 'Einspeisung / Verluste [MWh]', 'AC-Leitungen (Basisnetzwerk)', (list.loss_mwh.sum()*-1))
        to_run_dic(run_dic, 'Anzahl [N]', 'AC-Leitungen (Basisnetzwerk)', len(list.index))
    elif component == 'AC-Leitungen (NEP)':
         # AC-Leitungen, die durch NEP hinzugefügt wurden (add_transmission_projects_and_dir & simplify_network)
        to_run_dic(run_dic, 'Investitionskosten [€]', 'AC-Leitungen (NEP)', list.expansion_cost_EUR.sum())
        to_run_dic(run_dic, 'Ausgebaute Leistung [MW]', 'AC-Leitungen (NEP)', list.expansion_mw.sum()) #expansion_mw.sum())
        to_run_dic(run_dic, 'Einspeisung / Verluste [MWh]', 'AC-Leitungen (NEP)', (list.loss_mwh.sum()*-1))
        to_run_dic(run_dic, 'Anzahl [N]', 'AC-Leitungen (NEP)', len(list.index))
    elif component == 'DC-Leitungen (Basisnetzwerk)':
        # DC-Leitungen die im Base-Netz waren (base_network)
        to_run_dic(run_dic, 'Investitionskosten [€]', 'DC-Leitungen (Basisnetzwerk)', list.expansion_cost_EUR.sum())
        to_run_dic(run_dic, 'Ausgebaute Leistung [MW]', 'DC-Leitungen (Basisnetzwerk)', list.expansion_mw.sum())
        to_run_dic(run_dic, 'Einspeisung / Verluste [MWh]', 'DC-Leitungen (Basisnetzwerk)', 'loss = |p0| * (1-η), efficiency = η = 1 <-> loss = 0') #\text{loss} = |p0| \cdot (1-\eta)
        to_run_dic(run_dic, 'Anzahl [N]', 'DC-Leitungen (Basisnetzwerk)', len(list.index))
    elif component == 'DC-Leitungen (NEP)':
        # DC-Leitungen, die durch NEP hinzugefügt wurden (add_transmission_projects_and_dir & simplify_network)
        to_run_dic(run_dic, 'Investitionskosten [€]', 'DC-Leitungen (NEP)', list.expansion_cost_EUR.sum())
        to_run_dic(run_dic, 'Ausgebaute Leistung [MW]', 'DC-Leitungen (NEP)', list.expansion_mw.sum()) #expansion_mw.sum())
        to_run_dic(run_dic, 'Einspeisung / Verluste [MWh]', 'DC-Leitungen (NEP)', 'loss = |p0| * (1-η), efficiency = η = 1 <-> loss = 0')
        to_run_dic(run_dic, 'Anzahl [N]', 'DC-Leitungen (NEP)', len(list.index))
# ============================================== MAIN ==============================================
def main():
    # ===== PFAD =====
    # path_row = input("Bitte vollständigen Pfad zur .nc-Datei eingeben:\n> ").strip()
    # path_in = Path(path_row)
    path_in = Path(r"C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\1_BA_Referenzoptimierung\KN2045_Elek\networks\base_s_all_elec_.nc") # Auskommentieren, wenn fertig
    # ===== Netzwerk einlesen =====
    n = read_network(path_in)
    # ===== Ablageordner erstellen =====
    path_out = build_folder(n, 'analysis_final', path_in)
    # ===== Netzwerkdaten =====
    get_metadata(n, path_out, path_in)
    network_components(n, path_out)
    # ===== Diconary für RUN anlegen =====
    run_dic = {}
    # ===== Generatoren =====
    gen_list, gen_carrier_list = get_gen_data(n, path_out)
    merit_order(n, gen_list, path_out)
    data_cake = gen_carrier_list.groupby(carrier_key()).sum().energy_2045_MWh
    make_cake_dia(n, data_cake, path_out, 'Energieerzeugung')
    multiple_to_run_dic(run_dic, 'Generator', gen_list)
    # ===== Lasten =====
    to_run_dic(run_dic, 'Einspeisung / Verluste [MWh]', 'Last', (n.loads_t.p.sum().sum()*-1))
    # ===== Diagram zur Wahl des Zeitausschnitts =====
    data_pos_load, data_neg_load = data_for_time_plot(n) #Spezieller Plot, um Zeitpunkt des Netzengpasses zu bestimmen.
    time_plot_n_axses(n, data_pos_load.resample('D').mean(), path_out, file_title='Jahresüberblick', time_period='2045') # 2045
    # time_plot_n_axses(n, data_pos_load.loc['2013-12-04':'2013-12-06'], path_out, file_title='Einspeisung_Wind_Last_2045_2', time_period='04-12-2045 bis 06-12-2045')
    time_plot_n_axses(n, data_pos_load.loc['2013-12-05 09:00:00':'2013-12-05 13:00:00'], path_out, file_title='Ausschnitt Jahreshöchstleistung', time_period='05-12-2045') # 2045
    # time_plot_1_axes(n, data_neg_load.loc['2013-12-05 09:00:00':'2013-12-05 13:00:00'], path_out, file_title='Einspeisung_Wind_Last_2045_4')
    # ===== AC-Leitungen =====
    ac_lines_raw = get_ac_data(n)
    ac_nep_lines=get_csv(r'C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\data\transmission_projects\nep\new_lines.csv')
    ac_basenetwork, ac_transmission_projects_nep = filter_connections(ac_lines_raw, ac_nep_lines) # Gibt die verbauten Connections im Netzwerk aufgeteilt nach Roh- & NEP-Netzwerk zurück
    ac_transmission_projects_nep = make_real_expansion_data(ac_transmission_projects_nep, 'AC') # Neurechnen des Ausbau und Kosten für NEP-Projekte
    multiple_to_run_dic(run_dic, 'AC-Leitungen (Basisnetzwerk)', ac_basenetwork)
    multiple_to_run_dic(run_dic, 'AC-Leitungen (NEP)', ac_transmission_projects_nep)
    ac_lines = two_to_one(ac_basenetwork, ac_transmission_projects_nep) # Übersicht zum Speichern
    # ===== DC-Leitungen =====
    dc_links_raw = get_dc_data(n)
    links_nep=get_csv(r'C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\data\transmission_projects\nep\new_links.csv')
    dc_basenetwork, dc_transmission_projects_nep = filter_connections(dc_links_raw, links_nep)
    dc_transmission_projects_nep = make_real_expansion_data(dc_transmission_projects_nep, 'DC') # Neurechnen des Ausbau und Kosten für NEP-Projekte
    multiple_to_run_dic(run_dic, 'DC-Leitungen (Basisnetzwerk)', dc_basenetwork)
    multiple_to_run_dic(run_dic, 'DC-Leitungen (NEP)', dc_transmission_projects_nep)
    dc_links = two_to_one(dc_basenetwork, dc_transmission_projects_nep) # Übersicht zum Speichern
    # ===== Plots =====
    # make_plot(
    #     n,
    #     title,
    #     path_out=None,
    #     pc_title=None,
    #     line_color=None,
    #     line_width=None,
    #     link_color=None,
    #     link_width=None,
    #     *,
    #     scale_width=0.0008,
    #     min_width=1,
    #     cmap=plt.cm.RdYlGn_r,
    #     norm=None,
    #     show_colorbar=True,
    #     line_style="-",          
    #     link_style="-",          
    #     line_capstyle="round",  
    #     link_capstyle="round",   
    # )
    make_plot(n, 'Erster Überblick', path_out=path_out)
    make_plot(n, 'Normalbetrieb (Mittelwert 2045)', path_out=path_out,
            pc_title='Farbe: Auslastung [%], Breite: Installierte Leistung',  # (100 %: Zulässig installierte Leistung)
            line_color=ac_lines.mean_loading, 
            line_width=ac_lines.s_nom_opt, 
            link_color=dc_links.mean_loading, 
            link_width=dc_links.p_nom_opt
            )
    make_plot(n, 'Ausbau durch Solver', path_out=path_out,
            pc_title='Farbe: Ausbau [%] (Basis: ursprünglich installierte Leistung) ', 
            line_color= (ac_lines_raw.expansion_mw / ac_lines_raw.s_nom),
            link_color= (dc_links_raw.expansion_mw / dc_links_raw.p_nom)
            )
    
    ac_basenetwork_copy = ac_basenetwork.copy()
    ac_basenetwork_copy['expansion_mw'] = 0

    ac_transmission_projects_nep_copy = ac_transmission_projects_nep.copy()
    ac_transmission_projects_nep['s_nom'] = 1

    dc_basenetwork_copy = dc_basenetwork.copy()
    dc_basenetwork_copy['expansion_mw'] = 0

    dc_transmission_projects_nep_copy = dc_transmission_projects_nep.copy()
    dc_transmission_projects_nep_copy['p_nom'] = 1
    ac_nep_plot = two_to_one(ac_basenetwork_copy, ac_transmission_projects_nep_copy)
    dc_nep_plot = two_to_one(dc_basenetwork_copy, dc_transmission_projects_nep_copy)
    make_plot(n, 'Ausbau-NEP', path_out=path_out,
          line_color=ac_nep_plot.expansion_mw / ac_nep_plot.s_nom,
          link_color=dc_nep_plot.expansion_mw / dc_nep_plot.p_nom,
          pc_title='Farbe: Ausbau durch NEP-Projekte [%] (Basis: ursprünglich installierte Leistung), '
          )
    make_plot(n, 'Gesamtausbau (Solver & NEP)', path_out=path_out,
          line_color=ac_lines.expansion_mw/ac_lines.s_nom,
          line_width=ac_lines.expansion_cost_EUR/(ac_lines.expansion_cost_EUR.sum()+dc_links.expansion_cost_EUR.sum()),
          link_color=dc_links.expansion_mw/dc_links.p_nom,
          link_width=dc_links.expansion_cost_EUR/(ac_lines.expansion_cost_EUR.sum()+dc_links.expansion_cost_EUR.sum()),
          scale_width=60,
          pc_title='Farbe: Ausbau [%] (Basis: ursprünglich installierte Leistung), \nBreite: Investition [%] (Basis: GEsamtinvestition)'
          )
















    # ===== Diconary für Energy speichern =====
    save_run_dic(n, run_dic, path_out)
if __name__ == "__main__":
    main()
    print('\nFinsih')