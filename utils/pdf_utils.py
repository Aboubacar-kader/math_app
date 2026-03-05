"""
Utilitaire pour générer des PDFs depuis du texte Markdown.
Convertit les réponses du chatbot en documents PDF téléchargeables.
"""

from fpdf import FPDF
import re
from typing import Optional


class MarkdownPDF(FPDF):
    """PDF personnalisé pour IntelliMath avec support du français"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """En-tête personnalisé"""
        self.set_font('Arial', 'B', 12)
        self.set_text_color(79, 129, 189)  # Bleu
        self.cell(0, 10, 'IntelliMath', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Pied de page avec numéro"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def clean_markdown_for_pdf(text: str) -> str:
    """
    Nettoie le markdown pour le PDF.
    Supprime les éléments qui ne s'affichent pas bien en PDF.

    Args:
        text: Texte markdown brut

    Returns:
        Texte nettoyé
    """
    # Supprimer les formules LaTeX (elles ne s'affichent pas en PDF basique)
    text = re.sub(r'\$\$[^$]+\$\$', '[Formule mathematique]', text)
    text = re.sub(r'\$[^$]+\$', '[formule]', text)

    # Supprimer les balises markdown complexes
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Liens → texte

    # Titres markdown → texte simple
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\n\1\n', text, flags=re.MULTILINE)

    # Emphases
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Gras
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italique
    text = re.sub(r'`([^`]+)`', r'\1', text)        # Code inline

    # Blocs de code
    text = re.sub(r'```[\s\S]*?```', '[Code]', text)

    # Listes
    text = re.sub(r'^\s*[-*]\s+', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '- ', text, flags=re.MULTILINE)

    # Lignes horizontales
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

    # Nettoyer espaces multiples
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def markdown_to_pdf(
    content: str,
    title: str = "Document IntelliMath"
) -> bytes:
    """
    Convertit du contenu markdown en PDF.

    Args:
        content: Contenu markdown
        title: Titre du document

    Returns:
        Bytes du PDF généré
    """
    # Créer le PDF
    pdf = MarkdownPDF()
    pdf.add_page()

    # Titre principal
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, title, 0, 1, 'L')
    pdf.ln(5)

    # Nettoyer et ajouter le contenu
    clean_content = clean_markdown_for_pdf(content)

    # Encoder en latin-1 pour éviter erreurs Unicode
    safe_content = clean_content.encode('latin-1', 'replace').decode('latin-1')

    # Ajouter le contenu
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, safe_content)

    # Retourner les bytes (output retourne déjà un bytearray)
    return bytes(pdf.output(dest='S'))


def create_exercise_pdf(exercise: str, solution: str) -> bytes:
    """
    Crée un PDF spécial pour un exercice avec sa solution.

    Args:
        exercise: Énoncé de l'exercice
        solution: Solution de l'exercice

    Returns:
        Bytes du PDF
    """
    pdf = MarkdownPDF()
    pdf.add_page()

    # Énoncé
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, 'Enonce', 0, 1, 'L')
    pdf.ln(3)

    clean_exercise = clean_markdown_for_pdf(exercise)
    safe_exercise = clean_exercise.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, safe_exercise)

    pdf.ln(8)

    # Solution
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, 'Solution', 0, 1, 'L')
    pdf.ln(3)

    clean_solution = clean_markdown_for_pdf(solution)
    safe_solution = clean_solution.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, safe_solution)

    # output(dest='S') retourne déjà un bytearray, on le convertit en bytes
    return bytes(pdf.output(dest='S'))
