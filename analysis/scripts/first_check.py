import pypsa
from pathlib import Path
# ========= Was ist das? ========

print('Hello')

# ========= Einlesen Netzwerk =========
pfad = r'C:\Users\peterson_stud\Desktop\BA_PyPSA\pypsa-de\results\BA_NEP_2045_C50_1h\KN2045_Elek\networks\base_s_50_elec_.nc'
n = pypsa.Network(pfad)

