import pymongo
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from PIL import Image, ImageDraw, ImageFont
import glob
import math

# 1.KROK => PRIPOJENIE DATABÁZY A NAČÍTANIE DATASETU
# 2.KROK => DYNAMICKÉ NAČÍTANIE ATRIBÚTOV, ODSTRÁNENIE KATEGORICKÝCH PREMENNÝCH
# 3.KROK => VYHOTOVENIE SUMÁRNYCH ŠTATISTÍK
    # 1.VÝSTUP => TABUĽKA S VÝSLEDKAMI SUMÁRNYCH ŠTATISTÍK
    # + GRAFY SUMÁRNYCH ŠTATISTÍK (BAR CHARTS)
# 4.KROK => ANALÝZA KORELÁCIÍ (2 MATICE)
    # 2.VÝSTUP => KORELAČNÉ MATICE SO ZVÝRAZNENÝMI NAJSILNEJŠÍMI KORELÁCIAMI A ANTIKORELÁCIAMI
    # + GRAFY KORELAČNÝCH MATÍC (HEATMAPS)
# 5.KROK => VIZUALIZÁCIA SILNÝCH KORELAČNÝCH A ANTIKORELAČNÝCH VZŤAHOV
    # 3.VÝSTUP => BODOVÉ GRAFY DVOJÍC ATRIBÚTOV SO SILNÝMI KORELÁCIAMI A ANTIKORELÁCIAMI
# ---------------------------------------------------------------------------------------------

# 1.KROK
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["local"]

print("Zadajte názov kolekcie: ")
name = input()

collection_names = db.list_collection_names()
if name in collection_names:
    collection = db[name]
    print("Pracujem na exploratívnej analýze vášho datasetu ...")
else:
    print("ERROR: Kolekcia sa nenašla.")
    exit()
# ----------------------------------------------------------------

# 2.KROK
pipeline = []
keys = []
# prechádzanie kolekciou a ukladanie číselných polí do pipeliny
for field in collection.find_one().keys():
    if field != "_id" and type(collection.find_one()[field]) in [int, float]:
        pipeline.append({"$group": {"_id": None,
                                    f"{field}_avg": {"$avg": f"${field}"},
                                    f"{field}_min": {"$min": f"${field}"},
                                    f"{field}_max": {"$max": f"${field}"},
                                    f"{field}_sd": {"$stdDevPop": f"${field}"}
                                    }})
        keys.append(field)
# ----------------------------------------------------------------------------

# 3.KROK
result = {}
# vyhotovenie sumárnych štatistíc
for key in keys:
    stats = collection.aggregate([
        {"$group": {"_id": None,
                    f"{key}_avg": {"$avg": f"${key}"},
                    f"{key}_min": {"$min": f"${key}"},
                    f"{key}_max": {"$max": f"${key}"},
                    f"{key}_sd": {"$stdDevPop": f"${key}"}
                    }}
    ])
    result[key] = stats.next()
# --------------------------------------------------------

# 1.VÝSTUP
# načítanie dát pre tabuľku
data = [["", *keys],
        ["Priemer"] + [round(result[key][f"{key}_avg"], 2) for key in keys],
        ["Minimum"] + [result[key][f"{key}_min"] for key in keys],
        ["Maximum"] + [result[key][f"{key}_max"] for key in keys],
        ["Št.odchýlka"] + [round(result[key][f"{key}_sd"], 2) for key in keys]]

# vytvorenie tabuľky
table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 14),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
    ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 12),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))

# pridanie názvu
title_style = ParagraphStyle(name='title', fontSize=16, leading=inch*0.4, alignment=1, fontName='Helvetica-Bold')
title = Paragraph("Sumárne štatistiky", title_style)

pdf_file = "main_outputs/vystup1.pdf"
pdf = SimpleDocTemplate(pdf_file, pagesize=letter)
pdf.build([title, table])
# ---------------------------------------------------

# 1.VÝSTUP pokračovanie
# vytvorenie stĺpcových grafov z výsledkov pre každú premennú a uloženie do PDF súboru
fig, axs = plt.subplots(nrows=len(keys)//2 + len(keys)%2, ncols=2, figsize=(10, len(keys)*2), gridspec_kw={"hspace": 1, "wspace": 0.5}) #kw = keyword

for i, variable in enumerate(keys):
    row = i // 2
    col = i % 2
    labels = ["priemer", "min", "max", "sd"]
    values = [result[variable][f"{variable}_avg"],
              result[variable][f"{variable}_min"],
              result[variable][f"{variable}_max"],
              result[variable][f"{variable}_sd"]]
    x_values = range(len(labels))
    axs[row, col].bar(x_values, values)
    axs[row, col].set_xticks(x_values)
    axs[row, col].set_xticklabels(labels)
    axs[row, col].set_title(f"{variable} sumárne štatistiky")

# uloženie všetkých grafov na jednu stranu PDF súboru
with PdfPages("main_outputs/vystup1-2.pdf") as pdf:
    pdf.savefig(fig)
    plt.close(fig)
# -------------------------------------------------------------------------------

# 4.KROK
df = pd.DataFrame(list(collection.find()))
df.drop("_id", axis=1, inplace=True)    # vynechanie ID
df.dropna(how="any")    # vynechanie prázdnych riadkov
df = df.select_dtypes(include=[np.number])  # vybranie iba číselných polí

pearson_corr_matrix = df.corr(method='pearson').round(2)
spearman_corr_matrix = df.corr(method='spearman').round(2)
# --------------------------------------------------------

# 2.VÝSTUP
# definovanie veľkosti a typu písma, definovanie veľkosti buniek + zvlášť prvého stĺpca
font_size = 14
font = ImageFont.truetype('arial.ttf', font_size)
cell_width = 100
cell_height = 40
first_col_width = 120

# definovanie šírky a výšky obrázka na základe počtu stĺpcov a riadkov v matici
image_width = first_col_width + len(df.columns) * cell_width
image_height = (len(df.columns) + 1) * cell_height

# vytvorenie základu obrázka
img = Image.new('RGB', (image_width, image_height), color='white')
draw = ImageDraw.Draw(img)

# pridanie názvov atribútov v stĺpci
for i, col_name in enumerate(df.columns):
    x = i * cell_width + first_col_width + cell_width / 2 - font_size * len(col_name) / 4
    y = cell_height / 2 - font_size / 2
    draw.text((x, y), col_name, font=font, fill='black')

# pridanie názvov atribútov v riadku
for i, row_name in enumerate(df.columns):
    x = 0
    y = (i + 1) * cell_height + cell_height / 2 - font_size / 2
    draw.text((x, y), row_name, font=font, fill='black')

# prechádzanie koreláciami a pridávanie hodnôt do obrázku
for i in range(len(pearson_corr_matrix)):
    for j in range(len(pearson_corr_matrix.columns)):
        value = pearson_corr_matrix.iloc[i, j]
        text = str(value)
        x = j * cell_width + first_col_width + cell_width / 2 - font_size * len(text) / 4
        y = (i + 1) * cell_height + cell_height / 2 - font_size / 2
        draw.text((x, y), text, font=font, fill='black')

        # zvýraznie hodnôt so silnými koreláciami a antikoreláciami
        if value > 0.7 or value < -0.7:
            x1 = j * cell_width + first_col_width
            y1 = i * cell_height + cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            draw.rectangle([(x1, y1), (x2, y2)], fill='yellow')
            draw.text((x, y), text, font=font, fill='black')

# pridanie nadpisu
title_font = ImageFont.truetype('arial.ttf', 18)
title_text = "PCC matica"
draw.text((3, 10), title_text, (0, 0, 255), font=title_font)

# uloženie obrázku
img.save("main_outputs/matrices/PCC_matrix.png", format='PNG')

# --- skoro to isté pre druhú maticu ---

img2 = Image.new('RGB', (image_width, image_height), color='white')
draw = ImageDraw.Draw(img2)

for i, col_name in enumerate(df.columns):
    x = i * cell_width + first_col_width + cell_width / 2 - font_size * len(col_name) / 4
    y = cell_height / 2 - font_size / 2
    draw.text((x, y), col_name, font=font, fill='black')

for i, row_name in enumerate(df.columns):
    x = 0
    y = (i + 1) * cell_height + cell_height / 2 - font_size / 2
    draw.text((x, y), row_name, font=font, fill='black')

for i in range(len(spearman_corr_matrix)):
    for j in range(len(spearman_corr_matrix.columns)):
        value = spearman_corr_matrix.iloc[i, j]
        text = str(value)
        x = j * cell_width + first_col_width + cell_width / 2 - font_size * len(text) / 4
        y = (i + 1) * cell_height + cell_height / 2 - font_size / 2
        draw.text((x, y), text, font=font, fill='black')

        if value > 0.69 or value < -0.69:
            x1 = j * cell_width + first_col_width
            y1 = i * cell_height + cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            draw.rectangle([(x1, y1), (x2, y2)], fill='yellow')
            draw.text((x, y), text, font=font, fill='black')

title_text2 = "SRCC matica"
draw.text((3, 10), title_text2, (0, 255, 100), font=title_font)

img2.save("main_outputs/matrices/SRCC_matrix.png", format='PNG')
# --------------------------------------------------------------

# 2.VÝSTUP pokračovanie
# vytvorenie teplotných máp pre obidve matice a uloženie ako obrázok
sns.heatmap(pearson_corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.title("PCC matica - teplotná mapa", fontsize=16)
plt.savefig("main_outputs/matrices/PCC_heatmap.png")
plt.clf()
sns.heatmap(spearman_corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.title("SRCC matica - teplotná mapa", fontsize=16)
plt.savefig("main_outputs/matrices/SRCC_heatmap.png")

# uloženie obrázkov teplotných máp do PDF súboru
pdf = FPDF()
pdf.add_page()
pdf.image("main_outputs/matrices/PCC_matrix.png", w=200)
pdf.image("main_outputs/matrices/SRCC_matrix.png", y=150, w=200)
pdf.add_page()
pdf.image("main_outputs/matrices/PCC_heatmap.png", w=150)
pdf.image("main_outputs/matrices/SRCC_heatmap.png", y=150, w=150)
pdf.output("main_outputs/vystup2.pdf")
# ----------------------------------------------------------------

# 5.KROK
# vytvorenie zoznamu pre stĺpce s najsilnejšou koreláciou a antikoreláciou
# v PCC matici
strong_corr_cols_pearson = set()
for i in range(len(pearson_corr_matrix)):
    for j in range(len(pearson_corr_matrix.columns)):
        value = pearson_corr_matrix.iloc[i, j]
        if (0.69 < value < 1.1) or (-0.69 > value > -1.1):
            col1 = pearson_corr_matrix.index[i]
            col2 = pearson_corr_matrix.columns[j]
            if col1 != col2 and ((col1, col2) not in strong_corr_cols_pearson and (col2, col1) not in strong_corr_cols_pearson):
                strong_corr_cols_pearson.add((col1, col2))

# v SRCC matici
strong_corr_cols_spearman = set()
for i in range(len(spearman_corr_matrix)):
    for j in range(len(spearman_corr_matrix.columns)):
        value = spearman_corr_matrix.iloc[i, j]
        if (0.69 < value < 1.1) or (-0.69 > value > -1.1):
            col1 = spearman_corr_matrix.index[i]
            col2 = spearman_corr_matrix.columns[j]
            if col1 != col2 and ((col1, col2) not in strong_corr_cols_spearman and (col2, col1) not in strong_corr_cols_spearman):
                strong_corr_cols_spearman.add((col1, col2))

# porovnanie matíc a vytvorenie bodových grafov pre dvojice s najsilnejšou koreláciou a antikoreláciou
for cols in strong_corr_cols_pearson:
    if cols in strong_corr_cols_spearman:
        plot = sns.pairplot(data=df, x_vars=cols[0], y_vars=cols[1], height=5, plot_kws={"color": "black", "s": 3})
        plot.fig.suptitle(f"Bodový graf pre {cols[0]} a {cols[1]}")
        plt.savefig(f"main_outputs/scatterplots/scatterplot_{cols[0]}-{cols[1]}.png")

# načítanie všetkých obrázkov zo súboru
image_list = []
for filename in glob.glob("main_outputs/scatterplots/*.png"):
    img = Image.open(filename)
    image_list.append(img)

# počet grafov, stĺpcov a riadkov
num_plots = len(image_list)
num_cols = 2
num_rows = math.ceil(num_plots / num_cols)

# obrázok vo veľkosti A4 strany a vytvorenie gridu
fig = plt.figure(figsize=(8.27, 11.69))  # A4 (inches)
axes_list = fig.subplots(num_rows, num_cols).ravel() # The ravel() method in NumPy is used to convert a multi-dimensional array into a one-dimensional array.

# prechádzanie obrázkami a pridanie ich do nového grafu
for i, img in enumerate(image_list):
    axes_list[i].imshow(img)
    axes_list[i].axis("off")

# odstránenie "nepoužitých" podgrafov
for i in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes_list[i])

# uloženie do PDF súboru
with PdfPages("main_outputs/vystup3.pdf") as pdf:
    pdf.savefig(fig)
    plt.close(fig)
# ------------------------------------------------------------

print("\nHOTOVO :)")
exit()