# Läuft noch nicht!!!!

from pathlib import Path
import pypsa
import matplotlib.pyplot as plt

RESULTS_NC = Path("results/BA_NEPSens_2045/KN2045_Elek/networks/base_s_21_elec_.nc")

n = pypsa.Network(RESULTS_NC)

fig, ax = plt.subplots(figsize=(10, 10))

# Plot: Busse + Leitungen + Links
n.plot(
    ax=ax,
    bus_sizes=0.001,          # später anpassen
    line_widths=0.3,          # später anpassen
    link_widths=0.3,          # HVDC/Links etc.
)

ax.set_title("PyPSA network: buses, lines, links")
plt.tight_layout()
plt.show()