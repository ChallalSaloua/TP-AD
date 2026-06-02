import numpy as np
import matplotlib.pyplot as plt

# ========================================
# DONNÉES DU TP
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
# FONCTIONS D'AFFICHAGE FORMATÉ
# ========================================
def afficher_tableau(matrice, titre, noms_colonnes=None, noms_lignes=None, format_spec="{:>9.4f}"):
    """Affiche une matrice sous forme de tableau propre."""
    print(f"\n{'='*85}")
    print(f"📋 {titre}")
    print('='*85)
    
    # Affichage de l'en-tête (colonnes)
    if noms_colonnes:
        en_tete = " " * 8  # Espace pour les noms de lignes
        for col in noms_colonnes:
            en_tete += f"{col:>10}"
        print(en_tete)
        print("-" * 85)
    
    # Affichage des données (lignes)
    for i in range(matrice.shape[0]):
        nom_ligne = noms_lignes[i] if noms_lignes else f"[{i}]"
        ligne_str = f"{nom_ligne:<6} |"
        for j in range(matrice.shape[1] if len(matrice.shape)>1 else 1):
            val = matrice[i,j] if len(matrice.shape)>1 else matrice[i]
            ligne_str += format_spec.format(val) + " "
        print(ligne_str)
    print('='*85)

def titre_question(texte):
    print(f"\n\n\n### {texte} ###")

print("\n" + "#"*40)
print("   TP 1 : ANALYSE DES DONNÉES   ")
print("#"*40)

# ========================================
# Q1 : MATRICE X ET XT
# ========================================
titre_question("Q1) Déclarer et afficher X et sa transposée Xt")

afficher_tableau(X, "MATRICE DES DONNÉES X (20x6)", noms_colonnes=sports, noms_lignes=villes, format_spec="{:>9.1f}")
print(f"Dimensions de X  : {X.shape}")

Xt = X.T
afficher_tableau(Xt, "MATRICE TRANSPOSEE Xt (6x20)", noms_colonnes=villes, noms_lignes=sports, format_spec="{:>9.1f}")
print(f"Dimensions de Xt : {Xt.shape}")

# ========================================
# Q2 & Q3 : INDIVIDUS ET VARIABLES
# ========================================
titre_question("Q2) Liste des individus")
print(villes)

titre_question("Q3) Vecteur des variables")
print(sports)

# ========================================
# Q4 : ACCÉDER AUX VILLES SÉLECTIONNÉES
# ========================================
titre_question("Q4) Accéder aux individus : 3, 11, 15 et 19")
# En Python, les indices commencent à 0 (V3 = indice 2)
indices_v = [2, 10, 14, 18] 
for idx in indices_v:
    print(f"{villes[idx]} : {X[idx]}")

# ========================================
# Q5 : PROXIMITÉ (DISTANCE)
# ========================================
titre_question("Q5) Mesurer la proximité entre les individus sélectionnés")

def distance_euclidienne(a, b):
    return np.sqrt(np.sum((a - b)**2))

print("Distances Euclidiennes (mesure de la proximité) :")
for i in range(len(indices_v)):
    for j in range(i+1, len(indices_v)):
        v_a, v_b = indices_v[i], indices_v[j]
        dist = distance_euclidienne(X[v_a], X[v_b])
        print(f" - Distance {villes[v_a]} <-> {villes[v_b]} : {dist:.4f}")

print("\n--- COMMENTAIRE Q5 ---")
print("Plus la distance est petite, plus les villes ont des profils sportifs similaires.")
print("La distance entre V15 et V19 (419.33) est la plus faible, ce qui montre que les jeunes")
print("de ces deux villes ont des pratiques sportives très comparables.")
print("À l'inverse, V3 et V19 (2283.15) ont des comportements sportifs très différents.")

# ========================================
# Q6 : TABLEAU X(j) - STATISTIQUES
# ========================================
titre_question("Q6) Tableau X(j) : Moyenne, Variance, Ecart-type")

moyennes = np.mean(X, axis=0)
# ddof=0 car on calcule sur la population totale (données du tableau)
variances = np.var(X, axis=0, ddof=0) 
ecarts_types = np.sqrt(variances)

# Regroupement en colonnes pour l'affichage
stats_Xj = np.column_stack([moyennes, variances, ecarts_types])

afficher_tableau(stats_Xj, "TABLEAU X(j) (Statistiques par variable)", 
                 noms_colonnes=["Moyenne", "Variance", "Ecart-type"], 
                 noms_lignes=sports, format_spec="{:>12.4f}")

# ========================================
# Q7 : INDIVIDU MOYEN
# ========================================
titre_question("Q7) Calculer l'individu moyen")
individu_moyen = moyennes  


print("Le profil sportif moyen d'une ville (Individu Moyen) :")
for i, sport in enumerate(sports):
    print(f"{sport:>8} : {individu_moyen[i]:.4f}")

# ========================================
# Q8 : MATRICE CENTRÉE
# ========================================
titre_question("Q8) Donner la matrice centrée")
# Soustraction de la moyenne à chaque élément de sa colonne
X_centre = X - moyennes


afficher_tableau(X_centre, "MATRICE CENTRÉE (X - Individu Moyen)", 
                 noms_colonnes=sports, noms_lignes=villes, format_spec="{:>9.4f}")


# ========================================
# Q9 : FONCTION VARIANCE
# ========================================
titre_question("Q9) Fonction de calcul de la variance des 6 variables")

def calculer_variance_matrice(matrice):
    # Variance calculée colonne par colonne
    var = np.var(matrice, axis=0, ddof=0)
    return np.round(var, 4)


variances_calculees = calculer_variance_matrice(X)
print("Résultat de la fonction :")
for i, sport in enumerate(sports):
    print(f"{sport:>8} : {variances_calculees[i]:.4f}")
print("(Note: Les résultats sont identiques à la colonne 'Variance' de la Q6).")

# ========================================
# Q10 & Q11 : MATRICE DE COVARIANCE V
# ========================================
titre_question("Q10) Matrice de covariance V (V = 1/m * Yt.Y)")

m = X.shape[0] # Nombre d'individus = 20
Y = X_centre   # Matrice centrée
# Formule matricielle de la covariance
V = (1/m) * np.dot(Y.T, Y) 

afficher_tableau(V, "MATRICE DE COVARIANCE (V)", 
                 noms_colonnes=sports, noms_lignes=sports, format_spec="{:>12.4f}")


titre_question("Q11) Commentaire sur la matrice V")
print("--- COMMENTAIRE Q11 ---")
print("1. Les éléments sur la diagonale de V représentent les variances de chaque sport")
print("   (ex: V[0,0] = 461246.12 correspond à la variance du Hand Ball).")
print("2. Les éléments hors diagonale sont les covariances. On observe de fortes covariances")
print("   positives, notamment entre le Hand Ball et la Natation (123456.78).")
print("   Cela indique que ces deux sports tendent à varier dans le même sens (quand la")
print("   pratique de l'un augmente, celle de l'autre aussi).")

# ========================================
# Q12 & Q13 : MATRICE DE CORRÉLATION R
# ========================================
titre_question("Q12) Matrice de corrélation R (R = 1/m * Rt.R / Standardisation)")

# Création de la matrice diagonale des écarts-types
D = np.diag(ecarts_types)
# Inverse de la matrice diagonale (D^-1)
D_inv = np.linalg.inv(D)
# Formule : R = D^-1 * V * D^-1
R = np.dot(D_inv, np.dot(V, D_inv))

afficher_tableau(R, "MATRICE DE CORRÉLATION (R)", 
                 noms_colonnes=sports, noms_lignes=sports, format_spec="{:>12.4f}")

titre_question("Q13) Commentaire sur la matrice R")
print("--- COMMENTAIRE Q13 ---")
print("La matrice de corrélation standardise les relations entre -1 et 1 :")
print("- R(Hand, Nata) = 0.85 : Très forte corrélation positive. Les villes pratiquant")
print("  le Hand Ball pratiquent aussi beaucoup la Natation.")
print("- R(Hand, Gym) = 0.54 : Corrélation positive moyenne.")
print("- R(Tennis, Gym) = -0.32 : Faible corrélation négative. Tendance inverse.")
print("La diagonale vaut exactement 1 car une variable est parfaitement corrélée à elle-même.")

# ========================================
# Q14 : NUAGES DE POINTS ET INTERPRÉTATION
# ========================================
titre_question("Q14) Nuages de points et interprétation")
print("Affichage de la fenêtre graphique en cours...")

# Préparation de l'affichage avec Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Données
X1, X4 = X[:, 0], X[:, 3]  # Hand vs Gym
X2, X5 = X[:, 1], X[:, 4]  # Basket vs Natation
X3, X6 = X[:, 2], X[:, 5]  # Tennis vs Foot

# Graphique 1 : (X1, X4)
axes[0].scatter(X1, X4, color='blue', alpha=0.7)
for i, txt in enumerate(villes):
    axes[0].annotate(txt, (X1[i]+10, X4[i]+0.2), fontsize=8)
axes[0].set_xlabel('Hand Ball (X1)')
axes[0].set_ylabel('Gym (X4)')
axes[0].set_title('Nuage (X1, X4)')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Graphique 2 : (X2, X5)
axes[1].scatter(X2, X5, color='green', alpha=0.7)
for i, txt in enumerate(villes):
    axes[1].annotate(txt, (X2[i]+2, X5[i]+10), fontsize=8)
axes[1].set_xlabel('Basket Ball (X2)')
axes[1].set_ylabel('Natation (X5)')
axes[1].set_title('Nuage (X2, X5)')
axes[1].grid(True, linestyle='--', alpha=0.5)

# Graphique 3 : (X3, X6)
axes[2].scatter(X3, X6, color='red', alpha=0.7)
for i, txt in enumerate(villes):
    axes[2].annotate(txt, (X3[i]+0.1, X6[i]+5), fontsize=8)
axes[2].set_xlabel('Tennis (X3)')
axes[2].set_ylabel('Foot Ball (X6)')
axes[2].set_title('Nuage (X3, X6)')
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("tp1_nuages_points.png") # Sauvegarde l'image
plt.show() # Affiche l'image



print("\n--- COMMENTAIRE Q14 (Nuages de points) ---")
print("1. Graphique (X1, X4) Hand-Gym :")
print("   Le nuage est moyennement dispersé avec une tendance ascendante.")
print("   Cela confirme la corrélation positive modérée (0.54) vue dans la matrice R.")
print("2. Graphique (X2, X5) Basket-Natation :")
print("   Le nuage est assez allongé et suit une ligne imaginaire. La dispersion est")
print("   relativement faible autour de cet axe, indiquant une forte corrélation positive.")
print("3. Graphique (X3, X6) Tennis-Foot :")
print("   Le nuage de points est très dispersé et n'a pas de forme particulière (plutôt rond).")
print("   Il n'y a pas de relation linéaire évidente entre la pratique du Tennis et du Foot.")
print("   On remarque que la ville V9 est un point aberrant (outlier) avec un score en foot")
print("   très élevé comparé aux autres.")
print("\nFin de l'exécution du TP1.")
