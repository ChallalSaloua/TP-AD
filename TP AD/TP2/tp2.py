import numpy as np
import matplotlib.pyplot as plt

# ========================================
# DONNÉES (MÊMES QUE TP1)
# ========================================
X = np.array([
    [1881.9,  96.8, 14.2, 25.2, 1135.5, 278.3],
    [3369.8,  96.8, 10.8, 51.6, 1331.7, 284.0],
    [4467.4, 138.2,  9.5, 34.2, 2346.1, 312.3],
    [1862.1,  83.2,  8.8, 27.6,  972.6, 203.4],
    [3499.8, 287.0, 11.5, 49.4, 2139.4, 358.0],
    [3903.2, 170.7,  6.3, 42.0, 1935.2, 292.9],
    [2620.7, 129.5,  4.2, 16.8, 1346.0, 131.8],
    [3678.4, 157.0,  6.0, 24.9, 1682.6, 194.2],
    [3840.5, 187.9, 10.2, 39.6, 1859.9, 449.1],
    [2170.2, 140.5, 11.7, 31.1, 1351.1, 256.5],
    [3920.4, 128.0,  7.2, 25.5, 1911.5,  64.1],
    [2599.6,  39.6,  5.5, 19.4, 1050.8, 172.5],
    [2828.5, 211.3,  9.9, 21.8, 1085.0, 209.0],
    [2498.7, 123.2,  7.4, 26.5, 1086.2, 153.5],
    [2685.1,  41.2,  2.3, 10.6,  812.5,  89.8],
    [2739.3, 100.7,  6.6, 22.0, 1270.4, 180.5],
    [1662.1,  81.1, 10.1, 19.1,  872.2, 123.3],
    [2469.9, 142.9, 15.5, 30.9, 1165.5, 335.5],
    [2350.7,  38.7,  2.4, 13.5, 1253.1, 170.0],
    [3177.7, 292.1,  8.0, 34.8, 1400.0, 358.9]
])

villes = [f'V{i+1}' for i in range(20)]
sports = ['Hand', 'Basket', 'Tennis', 'Gym', 'Nata', 'Foot']

# ========================================
# FONCTIONS D'AFFICHAGE
# ========================================
def afficher_tableau(matrice, titre, noms_colonnes=None, noms_lignes=None,
                     format_spec="{:>10.4f}"):
    print(f"\n{'='*100}")
    print(titre)
    print('='*100)

    matrice = np.atleast_2d(matrice)

    if noms_colonnes:
        en_tete = " " * 10
        for col in noms_colonnes:
            en_tete += f"{col:>12}"
        print(en_tete)
        print("-" * 100)

    for i in range(matrice.shape[0]):
        nom_ligne = noms_lignes[i] if noms_lignes is not None else f"[{i}]"
        ligne_str = f"{nom_ligne:<8} |"
        for j in range(matrice.shape[1]):
            ligne_str += format_spec.format(matrice[i, j]) + " "
        print(ligne_str)
    print('='*100)


def titre_question(texte):
    print(f"\n\n### {texte} ###\n")


print("\n" + "#"*40)
print("   TP 2 : ANALYSE EN COMPOSANTES PRINCIPALES (ACP NORMÉE)   ")
print("#"*40)

# ========================================
# PRÉPARATION : CENTRAGE, RÉDUCTION, MATRICE DE CORRÉLATION
# ========================================
titre_question("Préparation : centrage, réduction et matrice de corrélation")

moyennes = np.mean(X, axis=0)
ecarts_types = np.std(X, axis=0, ddof=0)

Z = (X - moyennes) / ecarts_types    # centrée-réduite

afficher_tableau(Z, "MATRICE CENTRÉE-RÉDUITE Z",
                 noms_colonnes=sports, noms_lignes=villes, format_spec="{:>10.4f}")

n, p = X.shape
R = (1/(n-1)) * Z.T @ Z              # corrélation

afficher_tableau(R, "MATRICE DE CORRÉLATION R",
                 noms_colonnes=sports, noms_lignes=sports, format_spec="{:>10.4f}")

print("Remarque : l’ACP normée est réalisée à partir de la matrice de corrélation R (et non directement sur X).")

# ========================================
# Q1) VALEURS PROPRES ET VECTEURS PROPRES
# ========================================
titre_question("Q1) Valeurs propres et vecteurs propres de la matrice R")

valeurs_propres, vecteurs_propres = np.linalg.eig(R)
indices_trie = np.argsort(valeurs_propres)[::-1]
valeurs_propres = valeurs_propres[indices_trie]
vecteurs_propres = vecteurs_propres[:, indices_trie]

afficher_tableau(valeurs_propres.reshape(-1, 1),
                 "VALEURS PROPRES (ordonnées décroissantes)",
                 noms_colonnes=["lambda"],
                 noms_lignes=[f"Axe {i+1}" for i in range(p)],
                 format_spec="{:>10.4f}")

afficher_tableau(vecteurs_propres,
                 "VECTEURS PROPRES (colonnes = axes F1..F6)",
                 noms_colonnes=[f"F{i+1}" for i in range(p)],
                 noms_lignes=sports, format_spec="{:>10.4f}")

print("Commentaire Q1 :")
print("- La première valeur propre est voisine de 3.7 et la deuxième d’environ 1.5.")
print("- Les quatre dernières valeurs propres sont nettement plus petites (≤ 0.5).")
print("- Cela montre que l’essentiel de l’inertie est porté par les deux premiers axes factoriels (F1 et F2).")

# ========================================
# Q2) TEST SI v EST VECTEUR PROPRE ASSOCIÉ À λ
# ========================================
titre_question("Q2) Vérifier si v est un vecteur propre associé à λ pour une matrice A")

def est_vecteur_propre(A, v, lam, eps=1e-6):
    diff = A @ v - lam * v
    return np.linalg.norm(diff) < eps

print("Test pour le premier vecteur propre de R :")
lambda1 = valeurs_propres[0]
v1 = vecteurs_propres[:, 0]
norme_diff = np.linalg.norm(R @ v1 - lambda1 * v1)
ok1 = est_vecteur_propre(R, v1, lambda1)
print(f"  λ1 = {lambda1:.4f}")
print(f"  ||R v1 - λ1 v1|| = {norme_diff:.6e}")
print(f"  v1 est bien un vecteur propre associé à λ1 ? {ok1}")

print("\nTests pour les autres axes :")
for k in range(1, p):
    lam = valeurs_propres[k]
    vk = vecteurs_propres[:, k]
    ok = est_vecteur_propre(R, vk, lam)
    print(f"  Axe {k+1} : λ = {lam:.4f}, vecteur propre correct ? {ok}")

print("\nCommentaire Q2 :")
print("- Pour chaque valeur propre λk, la fonction vérifie que R vk ≈ λk vk.")
print("- Les normes des différences sont très faibles, ce qui confirme numériquement que les vecteurs trouvés sont bien des vecteurs propres de R.")

# ========================================
# Q3) REPRÉSENTATION GRAPHIQUE DES VALEURS PROPRES
# ========================================
titre_question("Q3) Diagramme des valeurs propres (éboulement)")

axes_num = np.arange(1, p+1)

plt.figure(figsize=(6, 4))
plt.plot(axes_num, valeurs_propres, marker='o', linestyle='-')
plt.axhline(y=1.0, color='red', linestyle='--', label='Seuil λ = 1 (Kaiser)')
plt.xlabel("Axe factoriel")
plt.ylabel("Valeur propre")
plt.title("Diagramme des valeurs propres (ACP normée)")
plt.xticks(axes_num)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("tp2_valeurs_propres.png")
plt.show()

print("Commentaire Q3 :")
print("- On observe une très forte première valeur propre (≈ 3.7), puis une deuxième valeur autour de 1.5.")
print("- À partir du troisième axe, les valeurs propres tombent nettement en dessous de 1 (≈ 0.5, 0.3, 0.2, 0.1).")
print("- La courbe présente un 'coude' marqué après F2, ce qui indique que les axes F1 et F2 concentrent l’essentiel de l’information.")

# ========================================
# Q4) TABLEAU (λ, TAUX D'INERTIE, TAUX CUMULÉS)
# ========================================
titre_question("Q4) Tableau des valeurs propres, taux d'inertie et taux cumulés")

inertie_totale = p
taux_inertie = (valeurs_propres / inertie_totale) * 100
taux_cumules = np.cumsum(taux_inertie)

tableau_inertie = np.column_stack((valeurs_propres, taux_inertie, taux_cumules))

afficher_tableau(tableau_inertie,
                 "INERTIE DES AXES FACTORIELS",
                 noms_colonnes=["lambda", "Taux (%)", "Cumul (%)"],
                 noms_lignes=[f"Axe {i+1}" for i in range(p)],
                 format_spec="{:>12.4f}")

print("Instruction utilisée pour les taux cumulés :")
print("  taux_cumules = np.cumsum(taux_inertie)")
print("Commentaire Q4 :")
print(f"- Le premier axe explique à lui seul environ {taux_inertie[0]:.1f}% de l’inertie totale.")
print(f"- Les deux premiers axes cumulent environ {taux_cumules[1]:.1f}% de l’inertie (plus de 80%).")
print("- Les axes suivants apportent chacun une part très faible de variance supplémentaire.")
print("- Le taux cumulé permet donc de choisir un nombre minimal d’axes qui expliquent une grande partie de la dispersion des données.")

# ========================================
# Q5) DIMENSION DU SOUS-ESPACE FACTORIEL
# ========================================
titre_question("Q5) Dimension du sous-espace factoriel à retenir")

axes_kaiser = np.where(valeurs_propres > 1)[0] + 1
dimension_kaiser = len(axes_kaiser)
print(f"Avec le critère de Kaiser (λ > 1), on retient {dimension_kaiser} axes : {list(axes_kaiser)}")
print(f"Selon le tableau d’inertie, les deux premiers axes expliquent environ {taux_cumules[1]:.1f}% de la variance totale.")
print("Nous choisissons donc un sous-espace factoriel de dimension d = 2 pour la visualisation des données (plan (F1,F2)).")

d = 2

# ========================================
# Q6) AXES FACTORIELS CHOISIS
# ========================================
titre_question("Q6) Axes factoriels choisis pour le sous-espace principal (F1, F2)")

P_d = vecteurs_propres[:, :d]

afficher_tableau(P_d,
                 f"COEFFICIENTS DES AXES FACTORIELS RETENUS (F1..F{d})",
                 noms_colonnes=[f"F{i+1}" for i in range(d)],
                 noms_lignes=sports, format_spec="{:>12.4f}")

print("Commentaire Q6 :")
print("- Les deux colonnes de ce tableau correspondent aux axes principaux F1 et F2 associés aux plus grandes valeurs propres.")
print("- Les coefficients de F1 sont élevés pour les variables fortement liées à la pratique globale des sports (par exemple Hand et Natation),")
print("  ce qui fait de F1 un axe de 'niveau général de pratique sportive'.")
print("- F2 met en évidence une opposition entre certains sports (par exemple entre Basket/Natation et Tennis/Foot selon les signes des coefficients).")

# ========================================
# Q7) PROJECTIONS DES INDIVIDUS
# ========================================
titre_question("Q7) Projections des individus sur les axes principaux (coordonnées factorielles)")

coord_ind = Z @ P_d

afficher_tableau(coord_ind,
                 f"COORDONNÉES FACTORIELLES DES VILLES SUR (F1..F{d})",
                 noms_colonnes=[f"F{i+1}" for i in range(d)],
                 noms_lignes=villes, format_spec="{:>12.4f}")

print("Commentaire Q7 :")
print("- Chaque ville est représentée par ses coordonnées factorielles sur F1 et F2.")
print("- Ces coordonnées résultent de la projection des profils centrés-réduits sur les axes factoriels (F = Z @ P_d).")

# ========================================
# Q8) REPRÉSENTATION GRAPHIQUE DES INDIVIDUS
# ========================================
titre_question("Q8) Représentation des villes dans le plan factoriel (F1, F2)")

plt.figure(figsize=(7, 6))
plt.scatter(coord_ind[:, 0], coord_ind[:, 1], color='blue', alpha=0.7)

for i, nom in enumerate(villes):
    plt.annotate(nom, (coord_ind[i, 0] + 0.05, coord_ind[i, 1] + 0.05), fontsize=8)

plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)
plt.xlabel("Axe 1 (F1)")
plt.ylabel("Axe 2 (F2)")
plt.title("Projection des villes dans le plan factoriel (F1, F2)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("tp2_plan_factoriel_F1F2.png")
plt.show()

# ========================================
# Q9) ANALYSE DU NUAGE DES INDIVIDUS
# ========================================
titre_question("Q9) Analyse du nuage des villes dans le plan (F1, F2)")

print("Commentaire Q9 :")
print("- Les villes situées à gauche du plan (F1 fortement négatif), comme V3, V5, V6 ou V9,")
print("  correspondent à des profils où le niveau global de pratique des sports considérés est relativement plus faible.")
print("- À l’inverse, les villes situées à droite (F1 positif), comme V15, V19, V12, V7 ou V16,")
print("  présentent des profils plus élevés sur les sports les plus corrélés positivement avec F1.")
print("- Sur l’axe vertical F2, certaines villes comme V1 et V18 sont en haut du plan,")
print("  ce qui traduit une sur‑représentation des sports associés positivement à F2.")
print("- La ville V11 se trouve en bas du plan (F2 très négatif), ce qui indique un comportement sportif opposé")
print("  à celui des villes en haut pour les variables fortement chargées sur F2.")
print("- Les villes proches les unes des autres (par exemple V2 et V20, ou encore V7, V12 et V16)")
print("  ont des profils sportifs très similaires, tandis que des villes éloignées comme V3 et V15")
print("  ont des profils très différents dans l’ensemble des sports étudiés.")

