"""
Module de tracé de figures géométriques
Détection automatique + Tracé interactif
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import re


# ============================================================
# DÉTECTION AUTOMATIQUE
# ============================================================

def detect_figure_needed(text: str) -> Dict[str, any]:
    """
    Détecte si l'énoncé demande un tracé de figure
    
    Returns:
        {
            'needs_figure': bool,
            'figure_type': str,  # 'triangle', 'cercle', 'repere', 'vecteur', etc.
            'parameters': dict,
            'description': str
        }
    """
    text_lower = text.lower()
    
    # Mots-clés de détection
    plot_keywords = [
        'trace', 'tracer', 'dessine', 'dessiner', 'représente', 'représenter',
        'graphique', 'courbe', 'figure', 'schéma', 'construction','représentation graphique','courbe représentative'
    ]
    
    needs_figure = any(keyword in text_lower for keyword in plot_keywords)
    
    if not needs_figure:
        return {'needs_figure': False}
    
    # Noms de fonctions mathématiques nommées → formule Python
    # name → (python_expr, display_latex)
    NAMED_FUNCTIONS = {
        # Algébriques
        'carré': ('x**2', 'x^2'),
        'cube': ('x**3', 'x^3'),
        'affine': ('2*x+1', '2x+1'),
        'linéaire': ('2*x', '2x'),
        'inverse': ('1/x', '1/x'),
        'puissance': ('x**2', 'x^2'),
        # Racines
        'racine carrée': ('np.sqrt(x)', '\\sqrt{x}'),
        'racine': ('np.sqrt(x)', '\\sqrt{x}'),
        # Partie entière
        'entière': ('np.floor(x)', 'E(x)'),
        'partie entière': ('np.floor(x)', 'E(x)'),
        'floor': ('np.floor(x)', 'E(x)'),
        # Logarithme / exponentielle
        'logarithme népérien': ('np.log(x)', '\\ln(x)'),
        'logarithme': ('np.log(x)', '\\ln(x)'),
        'log': ('np.log(x)', '\\ln(x)'),
        'exponentielle': ('np.exp(x)', 'e^x'),
        'exp': ('np.exp(x)', 'e^x'),
        # Trigonométriques
        'cosinus': ('np.cos(x)', '\\cos(x)'),
        'sinus': ('np.sin(x)', '\\sin(x)'),
        'tangente': ('np.tan(x)', '\\tan(x)'),
        'cos': ('np.cos(x)', '\\cos(x)'),
        'sin': ('np.sin(x)', '\\sin(x)'),
        'tan': ('np.tan(x)', '\\tan(x)'),
        # Valeur absolue
        'valeur absolue': ('np.abs(x)', '|x|'),
        'absolue': ('np.abs(x)', '|x|'),
        # Hyperboliques
        'sinus hyperbolique': ('np.sinh(x)', '\\sinh(x)'),
        'cosinus hyperbolique': ('np.cosh(x)', '\\cosh(x)'),
    }

    # Si "fonction" est mentionné, vérifier si c'est une fonction nommée
    # (avant de tester 'rectangle'/'carré' pour éviter la confusion "fonction carré")
    named_func_formula = None
    named_func_display = None
    if 'fonction' in text_lower or 'courbe' in text_lower or 'graphe' in text_lower:
        for name, (formula, display) in NAMED_FUNCTIONS.items():
            if name in text_lower:
                named_func_formula = formula
                named_func_display = display
                break

    # Détection du type de figure
    figure_types = {
        'triangle': ['triangle'],
        'cercle': ['cercle', 'rond'],
        'rectangle': ['rectangle'],
        'repere': ['repère', 'repere', 'axes', 'plan cartésien', 'repère orthonormé'],
        'fonction': ['fonction', 'courbe de', 'graphe de', 'f(x)', 'g(x)'],
        'vecteur': ['vecteur', 'vecteurs'],
        'angle': ['angle'],
        'polygone': ['polygone', 'pentagone', 'hexagone'],
        'carre': ['carré'],
    }

    # Priorité : si une fonction nommée est détectée, forcer le type 'fonction'
    if named_func_formula:
        detected_type = 'fonction'
    else:
        detected_type = None
        for fig_type, keywords in figure_types.items():
            if any(kw in text_lower for kw in keywords):
                detected_type = fig_type
                break
        # Normaliser 'carre' → 'rectangle' (carré géométrique)
        if detected_type == 'carre':
            detected_type = 'rectangle'

    # Extraction des paramètres
    parameters = extract_parameters(text, detected_type)
    # Injecter la formule de fonction nommée si trouvée et pas déjà extraite
    if named_func_formula and not parameters.get('function_py'):
        parameters['function'] = named_func_display or named_func_formula
        parameters['function_py'] = named_func_formula

    # Si une formule f(x)=... a été extraite, forcer le type 'fonction'
    # (ex : "repère orthonormé" + "f(x) = 2x−3" → tracer la fonction, pas les axes vides)
    if parameters.get('function_py'):
        detected_type = 'fonction'

    return {
        'needs_figure': True,
        'figure_type': detected_type or 'general',
        'parameters': parameters,
        'description': text
    }


def _to_python_expr(expr: str) -> str:
    """Convertit une expression mathématique en expression Python évaluable."""
    expr = expr.strip()
    expr = expr.replace('^', '**')
    expr = expr.replace('²', '**2').replace('³', '**3')
    # Multiplication implicite : 2x → 2*x, 3x² → 3*x**2, 2(x+1) → 2*(x+1)
    expr = re.sub(r'(\d)(x)', r'\1*x', expr)
    expr = re.sub(r'(\d)\(', r'\1*(', expr)
    expr = re.sub(r'(x)\(', r'x*(', expr)
    return expr


def extract_parameters(text: str, figure_type: str) -> dict:
    """Extrait les paramètres géométriques et analytiques de l'énoncé."""
    params = {}

    # ── Fonctions analytiques ────────────────────────────────────────────────
    func_patterns = [
        r'f\(x\)\s*=\s*([^,\n;]+)',
        r'g\(x\)\s*=\s*([^,\n;]+)',
        r'h\(x\)\s*=\s*([^,\n;]+)',
        r'y\s*=\s*([^,\n;]+)',
    ]
    for pattern in func_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            params['function'] = raw
            params['function_py'] = _to_python_expr(raw)
            break

    # ── Rayon (cercle) ───────────────────────────────────────────────────────
    # Cherche "rayon 5", "rayon = 5", "r = 5", "r=5"
    m = re.search(r'(?:rayon|radius)\s*[=:de]*\s*(\d+(?:[.,]\d+)?)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'\br\s*=\s*(\d+(?:[.,]\d+)?)', text)
    if m:
        params['radius'] = float(m.group(1).replace(',', '.'))

    # ── Dimensions rectangle ─────────────────────────────────────────────────
    # "longueur 4", "largeur 3", "côté 5", "4 cm", "3 m"
    m_l = re.search(r'(?:longueur|length|l)\s*[=:de]*\s*(\d+(?:[.,]\d+)?)', text, re.IGNORECASE)
    m_w = re.search(r'(?:largeur|width|larg)\s*[=:de]*\s*(\d+(?:[.,]\d+)?)', text, re.IGNORECASE)
    if m_l:
        params['length'] = float(m_l.group(1).replace(',', '.'))
    if m_w:
        params['width'] = float(m_w.group(1).replace(',', '.'))

    # ── Coordonnées de points : A(1,2) ou A(1;2) ────────────────────────────
    coord_matches = re.findall(
        r'\b([A-Z])\s*\(\s*(-?\d+(?:[.,]\d+)?)\s*[;,]\s*(-?\d+(?:[.,]\d+)?)\s*\)',
        text
    )
    if coord_matches:
        params['coords'] = {
            lbl: (float(x.replace(',', '.')), float(y.replace(',', '.')))
            for lbl, x, y in coord_matches
        }

    # ── Labels de points simples (sans coordonnées) ──────────────────────────
    point_labels = re.findall(r'\b([A-Z])\b', text)
    if point_labels:
        params['points'] = list(dict.fromkeys(point_labels))  # dédoublonné, ordre conservé

    return params


# ============================================================
# TRACEUR DE FIGURES - MATPLOTLIB
# ============================================================

class GeometryPlotter:
    """Classe pour tracer des figures géométriques"""
    
    def __init__(self, figsize=(8, 6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
    def plot_triangle(self, points: List[Tuple[float, float]] = None, 
                     labels: List[str] = None):
        """Trace un triangle"""
        if points is None:
            # Triangle par défaut
            points = [(0, 0), (4, 0), (2, 3)]
        
        if labels is None:
            labels = ['A', 'B', 'C']
        
        # Tracer les côtés
        triangle = plt.Polygon(points, fill=False, edgecolor='#FF6B35', linewidth=2)
        self.ax.add_patch(triangle)
        
        # Ajouter les points et labels
        for i, (point, label) in enumerate(zip(points, labels)):
            self.ax.plot(point[0], point[1], 'o', color='#FF6B35', markersize=8)
            offset = (0.2, 0.2)
            self.ax.text(point[0] + offset[0], point[1] + offset[1], label, 
                        fontsize=14, fontweight='bold')
        
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-1, 4)
        return self.fig
    
    def plot_circle(self, center: Tuple[float, float] = (0, 0), 
                   radius: float = 2):
        """Trace un cercle"""
        circle = plt.Circle(center, radius, fill=False, 
                          edgecolor='#FF6B35', linewidth=2)
        self.ax.add_patch(circle)
        
        # Centre
        self.ax.plot(center[0], center[1], 'o', color='#FF6B35', markersize=8)
        self.ax.text(center[0] + 0.2, center[1] + 0.2, 'O', 
                    fontsize=14, fontweight='bold')
        
        # Rayon
        self.ax.plot([center[0], center[0] + radius], [center[1], center[1]], 
                    'r--', linewidth=1.5, label=f'r = {radius}')
        self.ax.legend()
        
        self.ax.set_xlim(center[0] - radius - 1, center[0] + radius + 1)
        self.ax.set_ylim(center[1] - radius - 1, center[1] + radius + 1)
        return self.fig
    
    def plot_function(self, func_str: str, x_range: Tuple[float, float] = (-5, 5)):
        """Trace une fonction"""
        try:
            # Convertir la fonction string en fonction Python
            x = np.linspace(x_range[0], x_range[1], 400)
            
            # Remplacer les notations mathématiques
            func_str = func_str.replace('^', '**')
            func_str = func_str.replace('x2', 'x**2')
            func_str = func_str.replace('x²', 'x**2')
            
            # Évaluer la fonction
            y = eval(func_str, {'x': x, 'np': np, 'sin': np.sin, 
                               'cos': np.cos, 'tan': np.tan, 
                               'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log})
            
            self.ax.plot(x, y, color='#FF6B35', linewidth=2, label=f'f(x) = {func_str}')
            self.ax.axhline(y=0, color='k', linewidth=0.5)
            self.ax.axvline(x=0, color='k', linewidth=0.5)
            self.ax.legend()
            self.ax.set_xlabel('x', fontsize=12)
            self.ax.set_ylabel('y', fontsize=12)
            
        except Exception as e:
            st.error(f"Erreur lors du tracé : {e}")
        
        return self.fig
    
    def plot_vector(self, vectors: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                   labels: List[str] = None):
        """Trace des vecteurs"""
        if labels is None:
            labels = [f'v{i+1}' for i in range(len(vectors))]
        
        colors = ['#FF6B35', '#4CAF50', '#2196F3', '#FFC107']
        
        for i, ((x0, y0), (x1, y1)) in enumerate(vectors):
            color = colors[i % len(colors)]
            self.ax.arrow(x0, y0, x1-x0, y1-y0, head_width=0.3, head_length=0.2,
                         fc=color, ec=color, linewidth=2, label=labels[i])
            
            # Point d'origine
            self.ax.plot(x0, y0, 'o', color=color, markersize=6)
        
        self.ax.axhline(y=0, color='k', linewidth=0.5)
        self.ax.axvline(x=0, color='k', linewidth=0.5)
        self.ax.legend()
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        return self.fig
    
    def plot_coordinate_system(self, x_range: Tuple[int, int] = (-5, 5), 
                              y_range: Tuple[int, int] = (-5, 5)):
        """Trace un repère orthonormé"""
        self.ax.axhline(y=0, color='k', linewidth=1)
        self.ax.axvline(x=0, color='k', linewidth=1)
        
        # Flèches
        self.ax.arrow(x_range[1] - 0.5, 0, 0.3, 0, head_width=0.3, head_length=0.2, fc='k', ec='k')
        self.ax.arrow(0, y_range[1] - 0.5, 0, 0.3, head_width=0.3, head_length=0.2, fc='k', ec='k')
        
        # Labels des axes
        self.ax.text(x_range[1] - 0.5, -0.5, 'x', fontsize=14, fontweight='bold')
        self.ax.text(0.3, y_range[1] - 0.5, 'y', fontsize=14, fontweight='bold')
        
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        
        return self.fig


# ============================================================
# TRACEUR INTERACTIF - PLOTLY
# ============================================================

def create_interactive_triangle(points: List[Tuple[float, float]] = None):
    """Crée un triangle interactif avec Plotly"""
    if points is None:
        points = [(0, 0), (4, 0), (2, 3)]
    
    # Fermer le triangle
    x = [p[0] for p in points] + [points[0][0]]
    y = [p[1] for p in points] + [points[0][1]]
    
    fig = go.Figure()
    
    # Triangle
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=10, color='#FF6B35'),
        name='Triangle'
    ))
    
    # Labels
    labels = ['A', 'B', 'C']
    for point, label in zip(points, labels):
        fig.add_annotation(
            x=point[0], y=point[1],
            text=label,
            showarrow=False,
            font=dict(size=16, color='#FF6B35'),
            xshift=15, yshift=15
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True, scaleanchor="x", scaleratio=1),
        width=600, height=500,
        hovermode='closest'
    )
    
    return fig


def create_interactive_function(func_str: str, x_range: Tuple[float, float] = (-5, 5)):
    """Crée un graphique de fonction interactif avec axes gradués."""
    try:
        is_step = 'floor' in func_str or 'ceil' in func_str or 'round' in func_str
        # Fonction en escalier : points denses pour chaque entier
        if is_step:
            x = np.array([v for i in range(int(x_range[0])-1, int(x_range[1])+2)
                          for v in [i, i + 0.9999]])
            x = np.sort(x)
        else:
            x = np.linspace(x_range[0], x_range[1], 600)

        func_str = func_str.replace('^', '**').replace('x2', 'x**2').replace('x²', 'x**2')
        y = eval(func_str, {'x': x, 'np': np, 'sin': np.sin, 'cos': np.cos,
                            'tan': np.tan, 'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log,
                            'floor': np.floor, 'ceil': np.ceil, 'abs': np.abs})

        y_finite = y[np.isfinite(y)]
        y_min = float(np.min(y_finite)) if len(y_finite) else -5
        y_max = float(np.max(y_finite)) if len(y_finite) else 5
        y_pad = max((y_max - y_min) * 0.12, 0.5)
        y_lo, y_hi = y_min - y_pad, y_max + y_pad

        # Pas de graduation automatique (entiers si plage ≤ 20, sinon tous les 2 ou 5)
        def nice_tick(lo, hi):
            span = hi - lo
            if span <= 12:   return 1
            if span <= 30:   return 2
            if span <= 60:   return 5
            return 10

        dx = nice_tick(x_range[0], x_range[1])
        dy = nice_tick(y_lo, y_hi)

        fig = go.Figure()

        # Courbe
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#FF6B35', width=2.5),
            name=f'f(x) = {func_str}',
            connectgaps=False
        ))

        # Axe des abscisses (y = 0) et des ordonnées (x = 0)
        fig.add_hline(y=0, line_color='black', line_width=1.5)
        fig.add_vline(x=0, line_color='black', line_width=1.5)

        fig.update_layout(
            xaxis=dict(
                title='x',
                title_font=dict(size=14),
                range=[x_range[0], x_range[1]],
                dtick=dx,
                tick0=0,
                tickmode='linear',
                showgrid=True,
                gridcolor='#e0e0e0',
                gridwidth=1,
                zeroline=False,           # géré par add_vline
                showline=True,
                linecolor='black',
                ticks='outside',
                ticklen=5,
                tickfont=dict(size=12),
                minor=dict(showgrid=True, gridcolor='#f0f0f0', dtick=dx/2 if dx >= 2 else None),
            ),
            yaxis=dict(
                title='y',
                title_font=dict(size=14),
                range=[y_lo, y_hi],
                dtick=dy,
                tick0=0,
                tickmode='linear',
                showgrid=True,
                gridcolor='#e0e0e0',
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor='black',
                ticks='outside',
                ticklen=5,
                tickfont=dict(size=12),
                scaleanchor=None,
            ),
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
            width=720, height=520,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=30, t=40, b=60),
        )

        return fig

    except Exception as e:
        st.error(f"Erreur tracé : {e}")
        return None


# ============================================================
# TRACÉ AUTOMATIQUE — DISPATCHER UNIVERSEL
# ============================================================

def _base_layout(title: str = '') -> dict:
    """Layout Plotly commun à toutes les figures."""
    return dict(
        title=title,
        showlegend=False,
        width=660, height=520,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode='closest',
    )


def _axis_cfg(lo: float, hi: float, label: str = '') -> dict:
    span = hi - lo
    dtick = 1 if span <= 12 else 2 if span <= 30 else 5 if span <= 60 else 10
    return dict(
        title=label,
        range=[lo, hi],
        dtick=dtick, tick0=0, tickmode='linear',
        showgrid=True, gridcolor='#e0e0e0',
        zeroline=False, showline=True, linecolor='black',
        ticks='outside', ticklen=5, tickfont=dict(size=12),
    )


def _create_circle(center=(0, 0), radius=2.0, label='O') -> go.Figure:
    theta = np.linspace(0, 2 * np.pi, 400)
    cx, cy = center
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cx + radius * np.cos(theta),
        y=cy + radius * np.sin(theta),
        mode='lines', line=dict(color='#FF6B35', width=2.5),
        name=f'cercle r={radius}',
    ))
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers+text',
        marker=dict(size=8, color='#FF6B35'),
        text=[label], textposition='top right', textfont=dict(size=14)))
    # rayon
    fig.add_trace(go.Scatter(
        x=[cx, cx + radius], y=[cy, cy],
        mode='lines+text', line=dict(color='#FF6B35', width=1.5, dash='dash'),
        text=['', f'r={radius}'], textposition='top center',
    ))
    pad = radius * 0.2 + 0.5
    lo, hi = cx - radius - pad, cx + radius + pad
    fig.update_layout(
        **_base_layout(f'Cercle — centre {label}, rayon {radius}'),
        xaxis=dict(**_axis_cfg(lo, hi, 'x'), scaleanchor='y'),
        yaxis=_axis_cfg(cy - radius - pad, cy + radius + pad, 'y'),
    )
    fig.add_hline(y=0, line_color='black', line_width=1.2)
    fig.add_vline(x=0, line_color='black', line_width=1.2)
    return fig


def _create_rectangle(w=4.0, h=3.0) -> go.Figure:
    xs = [0, w, w, 0, 0]
    ys = [0, 0, h, h, 0]
    labels = ['A', 'B', 'C', 'D']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color='#FF6B35', width=2.5),
    ))
    for (px, py), lbl in zip([(0, 0), (w, 0), (w, h), (0, h)], labels):
        fig.add_annotation(x=px, y=py, text=lbl, showarrow=False,
                           font=dict(size=14, color='#FF6B35'),
                           xshift=-14 if px == 0 else 14,
                           yshift=-14 if py == 0 else 14)
    pad = max(w, h) * 0.15 + 0.5
    fig.update_layout(
        **_base_layout(f'Rectangle — {w} × {h}'),
        xaxis=dict(**_axis_cfg(-pad, w + pad, 'x'), scaleanchor='y'),
        yaxis=_axis_cfg(-pad, h + pad, 'y'),
    )
    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.add_vline(x=0, line_color='black', line_width=1)
    return fig


def _create_triangle(points=None) -> go.Figure:
    if points is None:
        points = [(0, 0), (4, 0), (2, 3)]
    labels = ['A', 'B', 'C']
    closed = points + [points[0]]
    xs = [p[0] for p in closed]
    ys = [p[1] for p in closed]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines+markers',
        line=dict(color='#FF6B35', width=2.5),
        marker=dict(size=9, color='#FF6B35'),
    ))
    for (px, py), lbl in zip(points, labels):
        fig.add_annotation(x=px, y=py, text=lbl, showarrow=False,
                           font=dict(size=15, color='#FF6B35', family='bold'),
                           xshift=12, yshift=12)
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    pad = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.2 + 1
    lo_x, hi_x = min(all_x) - pad, max(all_x) + pad
    lo_y, hi_y = min(all_y) - pad, max(all_y) + pad
    fig.update_layout(
        **_base_layout('Triangle'),
        xaxis=dict(**_axis_cfg(lo_x, hi_x, 'x'), scaleanchor='y'),
        yaxis=_axis_cfg(lo_y, hi_y, 'y'),
    )
    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.add_vline(x=0, line_color='black', line_width=1)
    return fig


def _create_repere(x_range=(-5, 5), y_range=(-5, 5)) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0, line_color='black', line_width=2)
    fig.add_vline(x=0, line_color='black', line_width=2)
    fig.update_layout(
        **_base_layout('Repère orthonormé'),
        xaxis=dict(**_axis_cfg(x_range[0], x_range[1], "x'x")),
        yaxis=dict(**_axis_cfg(y_range[0], y_range[1], "y'y"), scaleanchor='x'),
    )
    return fig


def _create_vector(vectors=None) -> go.Figure:
    if vectors is None:
        vectors = [((0, 0), (3, 2))]
    colors = ['#FF6B35', '#4CAF50', '#2196F3', '#FFC107']
    fig = go.Figure()
    for i, ((x0, y0), (x1, y1)) in enumerate(vectors):
        color = colors[i % len(colors)]
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowwidth=2.5,
            arrowcolor=color, arrowsize=1.5,
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='markers',
            marker=dict(size=6, color=color),
            showlegend=False,
        ))
    all_x = [p for v in vectors for p in (v[0][0], v[1][0])]
    all_y = [p for v in vectors for p in (v[0][1], v[1][1])]
    pad = 1.5
    lo_x, hi_x = min(all_x) - pad, max(all_x) + pad
    lo_y, hi_y = min(all_y) - pad, max(all_y) + pad
    fig.update_layout(
        **_base_layout('Représentation vectorielle'),
        xaxis=dict(**_axis_cfg(lo_x, hi_x, 'x')),
        yaxis=dict(**_axis_cfg(lo_y, hi_y, 'y'), scaleanchor='x'),
    )
    fig.add_hline(y=0, line_color='black', line_width=1.5)
    fig.add_vline(x=0, line_color='black', line_width=1.5)
    return fig


def auto_draw_figure(detection: dict):
    """
    Trace automatiquement la figure détectée.
    Retourne un go.Figure Plotly, ou None si le type n'est pas supporté.
    """
    fig_type = detection.get('figure_type', 'general')
    params = detection.get('parameters', {})
    values = params.get('values', [])

    try:
        # Priorité absolue : si une formule est présente, tracer la fonction
        func_py = params.get('function_py') or params.get('function')
        if func_py:
            return create_interactive_function(func_py)

        if fig_type == 'triangle':
            # Utiliser les coordonnées extraites si disponibles
            coords = params.get('coords', {})
            labels = params.get('points', ['A', 'B', 'C'])[:3]
            if len(coords) >= 3:
                pts = [coords[l] for l in labels if l in coords][:3]
                return _create_triangle(pts if len(pts) == 3 else None)
            return _create_triangle()

        elif fig_type == 'cercle':
            radius = params.get('radius', 2.0)
            return _create_circle(radius=radius)

        elif fig_type in ('rectangle', 'carre'):
            w = params.get('length', 4.0)
            h = params.get('width', 3.0)
            if w == 4.0 and h == 3.0 and fig_type == 'carre':
                w = h = 4.0
            return _create_rectangle(w, h)

        elif fig_type == 'repere':
            return _create_repere()

        elif fig_type == 'vecteur':
            return _create_vector()

        elif fig_type == 'angle':
            return _create_repere()

        elif fig_type in ('polygone', 'general'):
            return _create_repere()

    except Exception:
        pass

    return None


# ============================================================
# INTERFACE UTILISATEUR
# ============================================================

def render_figure_tool(exercise_text: str = None):
    """Interface complète pour tracer des figures"""
    
    st.markdown("### 📐 Outil de Tracé de Figures")
    
    # Détection automatique
    if exercise_text:
        detection = detect_figure_needed(exercise_text)
        
        if detection['needs_figure']:
            st.info(f"✨ Figure détectée : **{detection['figure_type']}**")
            
            # Suggestions basées sur la détection
            if detection['parameters'].get('points'):
                st.caption(f"Points détectés : {', '.join(detection['parameters']['points'])}")
            if detection['parameters'].get('values'):
                st.caption(f"Valeurs : {detection['parameters']['values']}")
    
    # Mode de tracé
    tab1, tab2, tab3 = st.tabs(["🎨 Tracé Manuel", "🤖 Tracé Auto", "📊 Fonctions"])
    
    with tab1:
        render_manual_plot()
    
    with tab2:
        if exercise_text:
            render_auto_plot(exercise_text)
        else:
            st.info("Collez un énoncé pour générer automatiquement la figure")
    
    with tab3:
        render_function_plot()


def render_manual_plot():
    """Interface de tracé manuel"""
    
    figure_type = st.selectbox(
        "Type de figure",
        ["Triangle", "Cercle", "Rectangle", "Vecteurs", "Repère"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**⚙️ Paramètres**")
        
        if figure_type == "Triangle":
            st.text_input("Point A (x,y)", value="0,0", key="pA")
            st.text_input("Point B (x,y)", value="4,0", key="pB")
            st.text_input("Point C (x,y)", value="2,3", key="pC")
            
            if st.button("🎨 Tracer", type="primary"):
                points = [
                    tuple(map(float, st.session_state.pA.split(','))),
                    tuple(map(float, st.session_state.pB.split(','))),
                    tuple(map(float, st.session_state.pC.split(',')))
                ]
                fig = create_interactive_triangle(points)
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif figure_type == "Cercle":
            center_x = st.number_input("Centre X", value=0.0)
            center_y = st.number_input("Centre Y", value=0.0)
            radius = st.number_input("Rayon", value=2.0, min_value=0.1)
            
            if st.button("🎨 Tracer", type="primary"):
                plotter = GeometryPlotter()
                fig = plotter.plot_circle((center_x, center_y), radius)
                with col1:
                    st.pyplot(fig)
        
        elif figure_type == "Vecteurs":
            n_vectors = st.number_input("Nombre de vecteurs", 1, 5, 2)
            vectors = []
            
            for i in range(int(n_vectors)):
                st.markdown(f"**Vecteur {i+1}**")
                col_a, col_b = st.columns(2)
                with col_a:
                    origin = st.text_input(f"Origine (x,y)", value="0,0", key=f"vo{i}")
                with col_b:
                    end = st.text_input(f"Fin (x,y)", value=f"{i+1},{i+1}", key=f"ve{i}")
                
                vectors.append((
                    tuple(map(float, origin.split(','))),
                    tuple(map(float, end.split(',')))
                ))
            
            if st.button("🎨 Tracer", type="primary"):
                plotter = GeometryPlotter()
                fig = plotter.plot_vector(vectors)
                with col1:
                    st.pyplot(fig)
        
        elif figure_type == "Repère":
            x_min = st.number_input("X min", value=-5)
            x_max = st.number_input("X max", value=5)
            y_min = st.number_input("Y min", value=-5)
            y_max = st.number_input("Y max", value=5)
            
            if st.button("🎨 Tracer", type="primary"):
                plotter = GeometryPlotter()
                fig = plotter.plot_coordinate_system((x_min, x_max), (y_min, y_max))
                with col1:
                    st.pyplot(fig)


def render_auto_plot(exercise_text: str):
    """Génération automatique basée sur l'énoncé"""
    
    st.markdown("**🤖 Génération Automatique**")
    st.text_area("Énoncé", value=exercise_text, height=100, key="auto_text")
    
    if st.button("✨ Générer la Figure", type="primary"):
        detection = detect_figure_needed(st.session_state.auto_text)
        
        if not detection['needs_figure']:
            st.warning("Aucune figure à tracer détectée")
            return
        
        with st.spinner("🎨 Génération en cours..."):
            fig_type = detection['figure_type']
            params = detection['parameters']
            
            if fig_type == 'triangle':
                points = [(0, 0), (4, 0), (2, 3)]  # Défaut
                fig = create_interactive_triangle(points)
                st.plotly_chart(fig, use_container_width=True)
            
            elif fig_type == 'cercle':
                plotter = GeometryPlotter()
                radius = params.get('values', [2])[0] if params.get('values') else 2
                fig = plotter.plot_circle((0, 0), radius)
                st.pyplot(fig)
            
            elif fig_type == 'fonction':
                func = params.get('function', 'x**2')
                fig = create_interactive_function(func)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info(f"Type '{fig_type}' en développement. Utilisez le tracé manuel.")


def render_function_plot():
    """Interface pour tracer des fonctions"""
    
    st.markdown("**📊 Tracé de Fonctions**")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        func = st.text_input(
            "Fonction f(x) =",
            value="x**2",
            help="Utilisez: x**2, sin(x), cos(x), sqrt(x), exp(x), log(x)"
        )
        
        x_min = st.number_input("X min", value=-5.0)
        x_max = st.number_input("X max", value=5.0)
        
        if st.button("📈 Tracer", type="primary"):
            fig = create_interactive_function(func, (x_min, x_max))
            if fig:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Exemples
    st.markdown("**💡 Exemples de fonctions**")
    examples = {
        "Parabole": "x**2",
        "Cube": "x**3",
        "Sinus": "np.sin(x)",
        "Cosinus": "np.cos(x)",
        "Exponentielle": "np.exp(x)",
        "Logarithme": "np.log(x)",
        "Racine": "np.sqrt(x)",
        "Polynôme": "x**3 - 2*x**2 + x - 1"
    }
    
    cols = st.columns(4)
    for i, (name, ex_func) in enumerate(examples.items()):
        with cols[i % 4]:
            if st.button(name, key=f"ex_{i}"):
                st.session_state.func_input = ex_func
                st.rerun()
