import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========
# DONNÉES
# ==========

# Coordonnées centrées des étudiants (Système, Réseaux, Algo)
Xc = np.array([
    [-1,  2, -1],   # Étudiant 1
    [ 1, -2,  1],   # Étudiant 2
    [-1,  2,  1],   # Étudiant 3
    [ 1, -2, -1]    # Étudiant 4
])

# Vecteurs propres (axes principaux) déjà déterminés
sqrt5 = np.sqrt(5)
e1 = np.array([1/sqrt5, -2/sqrt5, 0])   # Axe 1
e2 = np.array([2/sqrt5,  1/sqrt5, 0])   # Axe 2 (val propre 0)
e3 = np.array([0, 0, 1])                # Axe 3 (Algo)

# ==========
# COMPOSANTES PRINCIPALES (C1, C3)
# ==========

# On travaille dans le plan (C1, C3) car il explique 100 % de l'inertie
C1 = Xc @ e1    # produit matriciel : chaque ligne projetée sur e1
C3 = Xc @ e3    # C3 = juste la colonne Algo centrée

etiquettes_indiv = ['E1', 'E2', 'E3', 'E4']

# ==========
# COORDONNÉES DES VARIABLES SUR LE PLAN (C1, C3)
# ==========

# Pour le cercle des corrélations, on projette les vecteurs des variables
# Ici, comme tout est très simple :
# - Système ~ e1
# - Réseaux ~ opposé e1
# - Algo ~ e3
var_C1 = np.array([ 1/np.sqrt(5), -2/np.sqrt(5), 0.0])  # corr(var, C1)
var_C3 = np.array([ 0.0,          0.0,           1.0])  # corr(var, C3)
etiquettes_var = ['Système', 'Réseaux', 'Algo']

# ==========
# FIGURE AVEC 4 SOUS-GRAPHIQUES
# ==========

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Individus (plan C1-C3)",
        "Variables (plan C1-C3)",
        "Cercle des corrélations",
        "Biplot : individus + variables"
    )
)

# 1) INDIVIDUS
fig.add_trace(
    go.Scatter(
        x=C1, y=C3,
        mode="markers+text",
        text=etiquettes_indiv,
        textposition="top center",
        marker=dict(size=12, color="blue"),
        name="Étudiants"
    ),
    row=1, col=1
)
fig.update_xaxes(title_text="C1 (83 %)", row=1, col=1)
fig.update_yaxes(title_text="C3 (17 %)", row=1, col=1)

# 2) VARIABLES (seules)
colors_vars = ['red', 'orange', 'green']
for i in range(3):
    fig.add_trace(
        go.Scatter(
            x=[0, var_C1[i]],
            y=[0, var_C3[i]],
            mode="lines+markers+text",
            text=[etiquettes_var[i]],
            textposition="middle center",
            line=dict(width=4, color=colors_vars[i]),
            marker=dict(size=10),
            name=etiquettes_var[i]
        ),
        row=1, col=2
    )
fig.update_xaxes(title_text="C1", row=1, col=2)
fig.update_yaxes(title_text="C3", row=1, col=2)

# 3) CERCLE DES CORRÉLATIONS
theta = np.linspace(0, 2*np.pi, 200)
fig.add_trace(
    go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode="lines",
        line=dict(color="gray", dash="dash"),
        showlegend=False
    ),
    row=2, col=1
)

# On place aussi les variables dans ce plot
for i in range(3):
    fig.add_trace(
        go.Scatter(
            x=[0, var_C1[i]],
            y=[0, var_C3[i]],
            mode="lines+markers+text",
            text=[etiquettes_var[i]],
            textposition="middle center",
            line=dict(width=4, color=colors_vars[i]),
            marker=dict(size=10),
            showlegend=False
        ),
        row=2, col=1
    )
fig.update_xaxes(title_text="C1", row=2, col=1, range=[-1.2, 1.2])
fig.update_yaxes(title_text="C3", row=2, col=1, range=[-1.2, 1.2])

# 4) BIPLOT (INDIVIDUS + VARIABLES)
# Individus
fig.add_trace(
    go.Scatter(
        x=C1, y=C3,
        mode="markers+text",
        text=etiquettes_indiv,
        textposition="top center",
        marker=dict(size=10, color="blue"),
        name="Étudiants"
    ),
    row=2, col=2
)

# Variables
for i in range(3):
    fig.add_trace(
        go.Scatter(
            x=[0, var_C1[i]],
            y=[0, var_C3[i]],
            mode="lines+markers+text",
            text=[etiquettes_var[i]],
            textposition="middle center",
            line=dict(width=4, color=colors_vars[i]),
            marker=dict(size=10),
            name=etiquettes_var[i]
        ),
        row=2, col=2
    )

fig.update_xaxes(title_text="C1 (83 %)", row=2, col=2)
fig.update_yaxes(title_text="C3 (17 %)", row=2, col=2)

# ==========
# AFFICHAGE
# ==========

fig.update_layout(
    title="ACP - Individus, variables, cercle des corrélations et biplot",
    height=800
)

# Ouvre une fenêtre de navigateur avec les 4 graphes
fig.show()

# Si tu veux aussi sauvegarder en PNG :
# fig.write_image("acp_graphs.png")
