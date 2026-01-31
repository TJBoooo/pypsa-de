# ========= Erster Plot =========
import matplotlib.pyplot as plt
# Plot-Style´s
#plt.style.available
#plt.style.use('seaborn-v0_8')
#plt.style.use('default')
plt.style.use('bmh')
# Jupiter-Notebook (plot direkt)
# %matplotlib inline
#n.plot()



# ========= Abfrage Netzwerk =========
print("Objective [bn €/a]:", n.objective / 1e9)

print("\nInstalled capacity [GW]:")
print(n.generators.groupby("carrier").p_nom_opt.sum() / 1e3)

print("\nLine expansion [GW]:")
print((n.lines.s_nom_opt - n.lines.s_nom).sum() / 1e3)