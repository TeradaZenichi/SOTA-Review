from matplotlib import font_manager
import matplotlib.pyplot as plt
import os

font_path = 'Gulliver.otf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    print("Fonte 'Gulliver.otf' não encontrada, usando Times New Roman.")
    plt.rcParams['font.family'] = 'Times New Roman'

# Configurar tamanho de fonte para artigo científico
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# Definir tamanho do gráfico
fig_width_mm = 90 * 2
fig_height_mm = 90 * 2
fig_width_inch = fig_width_mm / 25.4
fig_height_inch = fig_height_mm / 25.4