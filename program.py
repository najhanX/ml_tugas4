import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv("./kelulusan_realistic.csv")


for col in ["IPK", "Jumlah_Absensi", "Waktu_Belajar_Jam", "Lulus"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.drop_duplicates()


df["IPK"] = df["IPK"].fillna(df["IPK"].median())
df["Jumlah_Absensi"] = df["Jumlah_Absensi"].fillna(df["Jumlah_Absensi"].median())
df["Waktu_Belajar_Jam"] = df["Waktu_Belajar_Jam"].fillna(df["Waktu_Belajar_Jam"].median())


df["IPK"] = df["IPK"].clip(1.0, 4.0)
df["Jumlah_Absensi"] = df["Jumlah_Absensi"].clip(0, 14).round().astype(int)
df["Waktu_Belajar_Jam"] = df["Waktu_Belajar_Jam"].clip(1, 14)


print(df.head())
print(df.describe())


sns.set_theme()


sns.boxplot(x=df["IPK"])
plt.title("Boxplot IPK (setelah cleaning)")
plt.tight_layout()
plt.show()


ipk = df["IPK"].replace([np.inf, -np.inf], np.nan).dropna()

plt.figure()
# hist tanpa kde di histplot
sns.histplot(data=df, x="IPK", bins=10, stat="density")
# tambahkan KDE terpisah hanya kalau variasi memadai
if ipk.nunique() > 1:
    sns.kdeplot(x=ipk)
else:
    print("[Info] IPK variannya 0, KDE dilewati.")
plt.title("Distribusi IPK (setelah cleaning)")
plt.tight_layout()
plt.show()

sns.scatterplot(x="IPK", y="Waktu_Belajar_Jam", data=df, hue="Lulus")
plt.title("IPK vs Waktu Belajar (label Lulus)")
plt.tight_layout()
plt.show()

num_cols = df.select_dtypes(include="number")
sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Numerik (setelah cleaning)")
plt.tight_layout()
plt.show()


df["Rasio_Absensi"] = df["Jumlah_Absensi"] / 14.0

df["IPK_x_Study"] = df["IPK"] * df["Waktu_Belajar_Jam"]


df.to_csv("processed_kelulusan.csv", index=False)
print("✅ processed_kelulusan.csv disimpan")


X = df.drop(columns=["Lulus"])
y = df["Lulus"].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)


