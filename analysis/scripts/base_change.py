# Idee: Schnelles, einheitliches umschreiben des Basisnetzwerkes.





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
def get_base_network():
    # Netzwerk liegt immer an der selben stelle und wird nach jedem Durchgang neu erstellt.
    path = r'C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\resources\networks\base.nc' 
    n = pypsa.Network(path)
    print('\n',n.meta['run']['name'], n.meta['run']['prefix'],'\n', n.component)

    return n
    






# ============================================== MAIN ==============================================
def main():
    n = get_base_network()
    n.plot()


    



























if __name__ == "__main__":
    main()
    print('\nFinsih')