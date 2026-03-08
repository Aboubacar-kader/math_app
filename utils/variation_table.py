"""
Rendu SVG des tableaux de variations (format mathématique français).
Utilisé pour visualiser l'étude des fonctions.
"""

import re
import random
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────
# PARSEUR
# ─────────────────────────────────────────────────────────────

def parse_variation_block(text: str) -> Optional[Dict]:
    """
    Extrait les données du bloc [TABLEAU_VARIATIONS]...[/TABLEAU_VARIATIONS].

    Format attendu dans le texte LLM :
        [TABLEAU_VARIATIONS]
        x: -∞, -1, 1, +∞
        f_prime: +, 0, -, 0, +
        f: -∞, 3, -1, +∞
        [/TABLEAU_VARIATIONS]

    Returns:
        dict avec 'x_labels', 'signs', 'f_values' ou None si absent/invalide
    """
    match = re.search(
        r'\[TABLEAU_VARIATIONS\](.*?)\[/TABLEAU_VARIATIONS\]',
        text, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return None

    raw = {}
    for line in match.group(1).split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, val = line.partition(':')
        key = key.strip().lower().replace("'", '').replace(' ', '_')
        items = [v.strip() for v in val.split(',') if v.strip()]
        if key == 'x':
            raw['x_labels'] = items
        elif key in ('f_prime', 'fprime', 'f_prime_signs'):
            raw['fprime_full'] = items
        elif key == 'f':
            raw['f_values'] = items

    if not all(k in raw for k in ('x_labels', 'fprime_full', 'f_values')):
        return None

    # fprime_full = ["+", "0", "-", "0", "+"]  (alternance signe / valeur critique)
    # signs = ["+", "-", "+"]  (un par intervalle)
    fp = raw['fprime_full']
    signs = [fp[i] for i in range(0, len(fp), 2)]

    n = len(raw['x_labels'])
    if len(signs) != n - 1 or len(raw['f_values']) != n:
        return None

    return {
        'x_labels': raw['x_labels'],
        'signs':    signs,
        'f_values': raw['f_values'],
    }


def strip_variation_block(text: str) -> str:
    """Supprime le bloc [TABLEAU_VARIATIONS] du texte."""
    return re.sub(
        r'\[TABLEAU_VARIATIONS\].*?\[/TABLEAU_VARIATIONS\]',
        '', text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


# ─────────────────────────────────────────────────────────────
# RENDU SVG
# ─────────────────────────────────────────────────────────────

_INFINITY_VALS = {'-∞', '+∞', '−∞', '–∞', '-inf', '+inf', '- ∞', '+ ∞'}


def render_variation_table(
    x_labels: List[str],
    signs:    List[str],
    f_values: List[str],
) -> str:
    """
    Génère un tableau de variations au format SVG embarqué dans du HTML.

    Args:
        x_labels : valeurs de x  — ex. ["-∞", "-1", "1", "+∞"]
        signs     : signe de f'  — ex. ["+", "-", "+"]  (longueur n-1)
        f_values  : valeurs de f — ex. ["-∞", "3", "-1", "+∞"]

    Returns:
        Chaîne HTML avec <svg> inline
    """
    n = len(x_labels)

    # ── Dimensions ────────────────────────────────────────────
    LABEL_W = 55
    X_W     = 62   # colonne critique / borne
    I_W     = 88   # colonne intervalle
    ROW0_H  = 40   # rangée x
    ROW1_H  = 40   # rangée f'(x)
    ROW2_H  = 96   # rangée f

    total_w = LABEL_W + n * X_W + (n - 1) * I_W
    total_h = ROW0_H + ROW1_H + ROW2_H

    # ── Centres de colonnes ────────────────────────────────────
    xc = [LABEL_W + i * (X_W + I_W) + X_W // 2 for i in range(n)]
    ic = [LABEL_W + i * (X_W + I_W) + X_W + I_W // 2 for i in range(n - 1)]

    # ── Positions verticales dans la rangée f ─────────────────
    f_top = ROW0_H + ROW1_H + 12
    f_bot = total_h - 12

    def _fy(i: int) -> int:
        if i == 0:
            return f_bot if signs[0] == '+' else f_top
        if i == n - 1:
            return f_top if signs[-1] == '+' else f_bot
        return f_top if signs[i - 1] == '+' else f_bot

    fy = [_fy(i) for i in range(n)]

    # ── Construction SVG ───────────────────────────────────────
    uid   = random.randint(10000, 99999)
    parts = []

    parts.append(f'''<div style="overflow-x:auto;">
<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg"
     style="font-family:'Times New Roman',serif; display:block; margin:10px 0;">
  <defs>
    <marker id="tv_arr_{uid}" markerWidth="9" markerHeight="7"
            refX="8" refY="3.5" orient="auto">
      <path d="M0,0 L9,3.5 L0,7 Z" fill="#1a1a1a"/>
    </marker>
  </defs>
  <!-- Fond et bordure extérieure -->
  <rect width="{total_w}" height="{total_h}" fill="white"
        stroke="black" stroke-width="1.6"/>''')

    # Lignes horizontales
    parts.append(f'  <line x1="0" y1="{ROW0_H}" x2="{total_w}" y2="{ROW0_H}" stroke="black" stroke-width="1.2"/>')
    parts.append(f'  <line x1="0" y1="{ROW0_H+ROW1_H}" x2="{total_w}" y2="{ROW0_H+ROW1_H}" stroke="black" stroke-width="1.2"/>')

    # Séparateur vertical étiquettes | données
    parts.append(f'  <line x1="{LABEL_W}" y1="0" x2="{LABEL_W}" y2="{total_h}" stroke="black" stroke-width="1.6"/>')

    # Traits verticaux pointillés aux points critiques
    for i, xl in enumerate(x_labels):
        if xl.strip() not in _INFINITY_VALS:
            parts.append(
                f'  <line x1="{xc[i]}" y1="0" x2="{xc[i]}" y2="{total_h}"'
                f' stroke="#777" stroke-width="0.8" stroke-dasharray="5,3"/>'
            )

    # ── Étiquettes de rangée ──────────────────────────────────
    y_lbl_x  = ROW0_H // 2
    y_lbl_fp = ROW0_H + ROW1_H // 2
    y_lbl_f  = ROW0_H + ROW1_H + ROW2_H // 2

    parts.append(f'  <text x="{LABEL_W//2}" y="{y_lbl_x}" text-anchor="middle" dominant-baseline="middle" font-size="14" font-style="italic">x</text>')
    parts.append(f'  <text x="{LABEL_W//2}" y="{y_lbl_fp}" text-anchor="middle" dominant-baseline="middle" font-size="12">f\'(x)</text>')
    parts.append(f'  <text x="{LABEL_W//2}" y="{y_lbl_f}" text-anchor="middle" dominant-baseline="middle" font-size="14" font-style="italic">f</text>')

    # ── Rangée x ─────────────────────────────────────────────
    for i, xl in enumerate(x_labels):
        parts.append(
            f'  <text x="{xc[i]}" y="{y_lbl_x}" text-anchor="middle"'
            f' dominant-baseline="middle" font-size="13">{xl}</text>'
        )

    # ── Rangée f'(x) ─────────────────────────────────────────
    for i, sign in enumerate(signs):
        color = '#1a56db' if sign == '+' else '#c81e1e'
        parts.append(
            f'  <text x="{ic[i]}" y="{y_lbl_fp}" text-anchor="middle"'
            f' dominant-baseline="middle" font-size="14" fill="{color}" font-weight="bold">{sign}</text>'
        )
    for i, xl in enumerate(x_labels):
        if xl.strip() not in _INFINITY_VALS:
            parts.append(
                f'  <text x="{xc[i]}" y="{y_lbl_fp}" text-anchor="middle"'
                f' dominant-baseline="middle" font-size="13">0</text>'
            )

    # ── Rangée f — valeurs ───────────────────────────────────
    FONT_F = 13
    for i, (fv, ypos) in enumerate(zip(f_values, fy)):
        if ypos == f_bot:
            dy, baseline = -4, 'auto'
        else:
            dy, baseline = 4, 'hanging'
        parts.append(
            f'  <text x="{xc[i]}" y="{ypos + dy}" text-anchor="middle"'
            f' dominant-baseline="{baseline}" font-size="{FONT_F}" font-weight="bold">{fv}</text>'
        )

    # ── Rangée f — flèches ───────────────────────────────────
    for i in range(n - 1):
        xa = xc[i]  + 14
        xb = xc[i+1] - 14

        ya = fy[i]   + (-(FONT_F + 4) if fy[i]   == f_bot else (FONT_F + 4))
        yb = fy[i+1] + (-(FONT_F + 4) if fy[i+1] == f_bot else (FONT_F + 4))

        parts.append(
            f'  <line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}"'
            f' stroke="#1a1a1a" stroke-width="1.6"'
            f' marker-end="url(#tv_arr_{uid})"/>'
        )

    parts.append('</svg>\n</div>')
    return '\n'.join(parts)
