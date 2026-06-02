import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA

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
n, p = X.shape
print("Nombre d'individus (villes) =", n)
print("Nombre de variables (sports) =", p)



# ============================================================


n, p = X.shape

# Pour avoir de "belles" valeurs (4 décimales, pas d'exposant)
np.set_printoptions(precision=4, suppress=True)

print("Nombre d'individus (villes) =", n)
print("Nombre de variables (sports) =", p)


# ============================================================
# Q1 : MATRICE UTILISÉE, VALEURS PROPRES ET VECTEURS PROPRES
# ============================================================
titre_question("Q1 :MATRICE UTILISÉE, VALEURS PROPRES ET VECTEURS PROPRES")
# Théorie :
# On fait une ACP NORMEE => on travaille sur la matrice de corrélation R
# 1) Centrer : c_ij = x_ij - moyenne_j
# 2) Réduire : z_ij = c_ij / sigma_j
# 3) R = (1/n) * Z^T * Z
# Valeurs propres λk et vecteurs propres vk issus de R donnent :
#  - λk : part d'inertie (variance) expliquée par l'axe k
#  - vk : direction de l'axe factoriel k (combinaison linéaire des variables)

def calcule_centre(m):
    """Calcule le vecteur des moyennes par variable (centre de gravité)."""
    g = np.zeros(m.shape[1])
    for j in range(m.shape[1]):
        g[j] = np.mean(m[:, j])
    return g

def matrice_centree(m):
    """Retourne la matrice centrée C = X - g (en colonne)."""
    g = calcule_centre(m)
    c = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            c[i, j] = m[i, j] - g[j]
    return c

def matrice_centree_reduite(m):
    """Retourne la matrice centrée-réduite Z (moyenne 0, variance 1)."""
    C = matrice_centree(m)
    Z = np.zeros(m.shape)
    for j in range(m.shape[1]):
        sigma = np.std(m[:, j], ddof=0)    # écart-type de la variable j
        for i in range(m.shape[0]):
            Z[i, j] = C[i, j] / sigma
    return Z

Z = matrice_centree_reduite(X)
print("\nMatrice centrée-réduite Z =\n", Z)

def calcule_correlation(m):
    """Calcule la matrice de corrélation R = (1/n) * Z^T * Z."""
    n = m.shape[0]
    return (1/n) * np.matmul(m.T, m)

R = calcule_correlation(Z)
print("\nMatrice de corrélation R =\n", R)
print("\n=> C'est cette matrice R que l'on diagonalise pour l'ACP normée.")

# Détermination des valeurs propres et vecteurs propres de R
valp, vecp = LA.eig(R)

print("\nValeurs propres (non triées) =\n", valp)
print("\nVecteurs propres (lignes = axes factoriels) =\n", vecp.T)


# ============================================================
# Q2 : TEST "v EST-IL VECTEUR PROPRE ASSOCIE A λ D'UNE MATRICE A ?"
# ============================================================
titre_question("Q2 :TEST v EST-IL VECTEUR PROPRE ASSOCIE A λ D'UNE MATRICE ")
# Théorie :
# v est vecteur propre de A associé à λ si : A v = λ v
# Numériquement on vérifie que ||A v - λ v|| est très petite.

def est_vecteur_propre(A, v, lamb, eps=1e-6):
    """
    Teste si v est vecteur propre de A associé à lamb :
    renvoie (True/False, norme de Av - λv).
    """
    diff = np.matmul(A, v) - lamb * v
    norme = LA.norm(diff)
    return norme < eps, norme

print("\n--- Vérification des couples (λk, vk) pour R ---")
for k in range(len(valp)):
    ok, err = est_vecteur_propre(R, vecp[:, k], valp[k])
    print(f"Axe {k+1} : vecteur propre ? {ok}  (erreur = {err:.2e})")

# Intérêt :
# - Vérifier la définition : A v ≈ λ v
# - Vérifier que la décomposition propre retournée par l'algorithme est correcte.


# ============================================================
# Q3 : DIAGRAMME DES VALEURS PROPRES (SCREE PLOT)
# ============================================================
titre_question("Q3 :DIAGRAMME DES VALEURS PROPRES ")
# Théorie :
# On trie les valeurs propres par ordre décroissant.
# Critère de Kaiser dans une ACP normée : garder les λk > 1.

valp_sorted = np.sort(valp)[::-1]

plt.figure(figsize=(7,5))
plt.plot(range(1, len(valp_sorted)+1), valp_sorted, marker='o')
plt.axhline(1, color='red', linestyle='--', label="Seuil λ = 1 (Kaiser)")
plt.title("Diagramme des valeurs propres (ACP normée)")
plt.xlabel("Axe factoriel")
plt.ylabel("Valeur propre")
plt.grid(True)
plt.legend()
plt.show()

# Intérêt :
# - Visualiser la décroissance des λk (rupture de pente)
# - Repérer les axes importants (λk grands) et les axes négligeables.


# ============================================================
# Q4 : TABLEAU (λk, TAUX D'INERTIE, TAUX D'INERTIE CUMULEE)
# ============================================================
titre_question("Q4 : TABLEAU (λk, TAUX D'INERTIE, TAUX D'INERTIE CUMULEE) ")
# Théorie :
# TI_k (taux d'inertie expliqué par l'axe k) :
#     TI_k = λk / (somme_j λj)
# TC_k (taux d'inertie cumulée) :
#     TC_k = somme_{j=1..k} TI_j


variance_expliquee = valp_sorted / np.sum(valp_sorted)    # TI_k
variance_cumulee = np.cumsum(variance_expliquee)          # TC_k

print("\nTableau des valeurs propres et inerties :\n")
print("Axe\tLambda\t\t% Inertie\t% Inertie cumulée")
for i in range(len(valp_sorted)):
    lam = valp_sorted[i]
    ti = variance_expliquee[i] * 100
    tc = variance_cumulee[i] * 100
    print(f"{i+1}\t{lam:.4f}\t\t{ti:.2f}\t\t{tc:.2f}")

# Intérêt :
# - TI_k : importance de CHAQUE axe (en % de variance expliquée).
# - TC_k : permet de choisir un nombre d'axes minimal pour atteindre
#          un certain pourcentage d'information (ex : 80 %).


# ============================================================
# Q5 : DIMENSION DU SOUS-ESPACE FACTORIEL A RETENIR
# ============================================================
titre_question("Q5 : DIMENSION DU SOUS-ESPACE FACTORIEL A RETENIR ")
# Théorie :
# On choisit le plus petit k tel que TC_k >= seuil (souvent 70% ou 80%).
# Ici on prend 80% comme dans beaucoup de cours.

seuil = 0.80
nb_axes = np.argmax(variance_cumulee >= seuil) + 1

print(f"\nNombre d'axes à retenir (taux d'inertie cumulée >= 80%) : {nb_axes}")

# Justification :
# - Les {nb_axes} premiers axes expliquent au moins 80% de l'inertie totale.
# - Ils résument donc l'essentiel de l'information contenue dans les données.


# ============================================================
# Q6 : AXES FACTORIELS CHOISIS POUR LE SOUS-ESPACE PRINCIPAL
# ============================================================
titre_question("Q6 : AXES FACTORIELS CHOISIS POUR LE SOUS-ESPACE PRINCIPAL")
# Théorie :
# On ordonne les valeurs propres décroissantes et on garde les vecteurs propres
# associés aux nb_axes plus grandes valeurs.
# Chaque vecteur propre vk donne les coefficients de l'axe factoriel Fk.

indices = np.argsort(valp)[::-1]     # indices triés décroissant
valp_ord = valp[indices]             # λ ordonnés
vecp_ord = vecp[:, indices]          # vecteurs propres ordonnés (colonnes)

print("\nValeurs propres ordonnées :\n", valp_ord)

print("\nAxes factoriels (vecteurs propres ordonnés) :")
for k in range(p):
    print(f"Axe F{k+1} (λ = {valp_ord[k]:.4f}) : ", end='')
    for j in range(p):
        print(f"{vecp_ord[j, k]:.4f} ", end='')
    print()

# Sous-espace principal d'ajustement : axes retenus
axes_retenus = vecp_ord[:, :nb_axes]

print(f"\nAxes factoriels retenus (dimension {nb_axes}) :")
for k in range(nb_axes):
    print(f"Axe F{k+1} : ", end='')
    for j in range(p):
        print(f"{axes_retenus[j, k]:.4f} ", end='')
    print()

# Intérêt :
# - Ces axes définissent le nouveau repère dans lequel on va représenter
#   les villes (sous-espace principal).
# - Les coefficients forts (en valeur absolue) indiquent quelles variables
#   caractérisent chaque axe.


# ============================================================
# Q7 : PROJECTIONS DES INDIVIDUS SUR LES AXES RETENUS
# ============================================================
titre_question("Q7 : PROJECTIONS DES INDIVIDUS SUR LES AXES RETENUS")
# Théorie :
# Coordonnées factorielles des individus :
#   F = Z * A
# où Z est la matrice centrée-réduite et A la matrice des vecteurs propres.
# Géométriquement, chaque coordonnée F_ik est le produit scalaire
# entre la ville i et l'axe factoriel k.

F = np.matmul(Z, axes_retenus)

print("\nCoordonnées factorielles des villes sur les axes retenus :\n")
entete = "Ville\t" + "\t".join([f"F{k+1}" for k in range(nb_axes)])
print(entete)
for i in range(n):
    ligne = f"V{i+1}\t" + "\t".join([f"{F[i, k]:.4f}" for k in range(nb_axes)])
    print(ligne)

# Intérêt :
# - Ces coordonnées servent pour les graphiques (plan factoriel),
#   et pour analyser les ressemblances / oppositions entre villes.


# ============================================================
# Q8 : REPRESENTATION GRAPHIQUE DES VILLES DANS LE SOUS-ESPACE
# ============================================================

titre_question("Q8 : REPRESENTATION GRAPHIQUE DES VILLES DANS LE SOUS-ESPACE")
# Théorie :
# Si nb_axes >= 2, on représente les points (F1, F2) de chaque ville.
# On trace aussi les axes F1 et F2 (droites x=0 et y=0).

if nb_axes >= 2:
    plt.figure(figsize=(7,7))
    plt.scatter(F[:, 0], F[:, 1], color='blue')

    # étiquettes des villes (V1, V2, ..., Vn)
    for i in range(n):
        plt.text(F[i, 0] + 0.05, F[i, 1] + 0.05, f"V{i+1}")

    plt.axhline(0, color='grey', linewidth=1)
    plt.axvline(0, color='grey', linewidth=1)
    plt.xlabel("Axe 1 (F1)")
    plt.ylabel("Axe 2 (F2)")
    plt.title("Projection des villes dans le plan factoriel (F1, F2)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
else:
    print("Moins de deux axes retenus : pas de plan (F1, F2).")

# Intérêt :
# - Permet de visualiser la structure des données :
#   groupes de villes proches, villes opposées, villes isolées.




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
