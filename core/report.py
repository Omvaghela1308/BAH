"""
PDF Report Generation Module.

Compiles risk assessment metrics, analysis, and recommendations into a formatted
professional PDF document utilizing fpdf2.
"""

import datetime
from fpdf import FPDF


def clean_text(text: str) -> str:
    """
    Sanitize input text to prevent FPDF font encoding crashes.
    Strips emojis and maps common unicode characters to latin-1 equivalents.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Map common emojis and symbols to safe ASCII/Latin-1 text
    replacements = {
        "⚠️": "[!]",
        "🔴": "[CRITICAL]",
        "🟢": "[LOW]",
        "🟡": "[MEDIUM]",
        "🟠": "[HIGH]",
        "🛡️": "[SECURE]",
        "🛰️": "[SAT]",
        "🔍": "[DET]",
        "°C": " deg C",
        "°F": " deg F",
        "°": " deg ",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'"
    }
    
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
        
    # Fallback: replace any remaining non-latin-1 characters with a question mark
    return text.encode('latin-1', 'replace').decode('latin-1')


class RiskReportPDF(FPDF):
    """Custom FPDF layout for AI Thermal Risk Assessment Reports."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Draw futuristic dark-themed header."""
        # Top background panel
        self.set_fill_color(22, 27, 34)  # #161b22 slate color
        self.rect(0, 0, 210, 38, "F")

        # Top border accent line (blue neon representation)
        self.set_fill_color(88, 166, 255)  # #58a6ff blue accent
        self.rect(0, 38, 210, 1.5, "F")

        # Header Title text
        self.set_xy(15, 10)
        self.set_text_color(88, 166, 255)
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "SATELLITE THERMAL RISK REPORT", border=False, ln=True, align="L")

        self.set_x(15)
        self.set_text_color(139, 148, 158)  # #8b96a5 grey text
        self.set_font("helvetica", "I", 9)
        self.cell(0, 5, "Bhartiya Antariksh AI Intelligence Layer", border=False, ln=True, align="L")

        # Reset cursor position below header
        self.set_xy(15, 48)

    def footer(self):
        """Draw footer with page number and timestamp."""
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(139, 148, 158)
        
        # Draw bottom boundary line
        self.set_draw_color(48, 54, 61)  # #30363d border
        self.line(15, self.get_y() - 2, 195, self.get_y() - 2)
        
        # Footer text
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(100, 10, f"Generated: {current_time} | Confidential", border=False, align="L")
        self.cell(0, 10, f"Page {self.page_no()}", border=False, align="R")


def generate_pdf_report(data: dict) -> bytes:
    """
    Generate a structured PDF report in-memory.

    Args:
        data (dict): Report fields including timestamp, risk_score, threat_level,
                    analysis_points, recommendation, objects_detected,
                    processing_time_ms, and image_name.

    Returns:
        bytes: Compiled PDF bytes.
    """
    pdf = RiskReportPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    # Section 1: Document Metadata Card
    pdf.set_fill_color(240, 246, 252)  # Very light grey background
    pdf.set_draw_color(208, 215, 222)  # Light grey border
    pdf.rect(15, 48, 180, 42, "FD")

    pdf.set_text_color(36, 41, 47)  # Dark text
    
    # Metadata Fields
    pdf.set_xy(20, 52)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 6, "Target Image:")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, clean_text(str(data.get("image_name", "N/A"))), ln=True)

    pdf.set_x(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 6, "Analysis Timestamp:")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, clean_text(str(data.get("timestamp", "N/A"))), ln=True)

    pdf.set_x(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 6, "Processing Latency:")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"{data.get('processing_time_ms', 0.0):.2f} ms", ln=True)

    pdf.set_x(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 6, "Objects Detected:")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, str(data.get("objects_detected", 0)), ln=True)

    pdf.set_x(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 6, "AI Confidence:")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"{data.get('confidence', 90.0):.1f}%", ln=True)

    # Section 2: Risk Assessment Panel
    pdf.ln(12)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(88, 166, 255)
    pdf.cell(0, 8, "RISK ASSESSMENT SUMMARY", ln=True)
    pdf.ln(2)

    # Determine colors based on threat level
    level = data.get("threat_level", "LOW")
    score = data.get("risk_score", 0)

    # Color definitions (R, G, B)
    if level == "LOW":
        color_rgb = (46, 160, 67)      # Green
    elif level == "MEDIUM":
        color_rgb = (210, 153, 34)     # Yellow/Gold
    elif level == "HIGH":
        color_rgb = (240, 136, 62)     # Orange
    else:  # CRITICAL
        color_rgb = (248, 81, 73)      # Red

    # Score Box & Threat Badge Table
    pdf.set_fill_color(248, 249, 250)
    pdf.set_draw_color(220, 224, 230)
    pdf.rect(15, 102, 180, 24, "FD")

    pdf.set_xy(20, 105)
    pdf.set_text_color(36, 41, 47)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(50, 18, "Overall Risk Index:")
    
    # Large score text
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(*color_rgb)
    pdf.cell(30, 18, f"{score} / 100")

    # Threat Badge
    pdf.set_xy(110, 109)
    pdf.set_fill_color(*color_rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(75, 10, f"THREAT LEVEL: {level}", border=False, ln=True, align="C", fill=True)

    # Section 3: Detailed AI Analysis
    pdf.set_xy(15, 132)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(88, 166, 255)
    pdf.cell(0, 8, "DETAILED AI ANALYSIS", ln=True)
    pdf.ln(2)

    pdf.set_text_color(36, 41, 47)
    pdf.set_font("helvetica", "", 10)
    
    analysis_points = data.get("analysis_points", [])
    if not analysis_points:
        pdf.cell(0, 6, "- No abnormal thermal patterns identified in the scene.", ln=True)
    else:
        for point in analysis_points:
            pdf.cell(6, 6, ">>")
            pdf.multi_cell(0, 6, clean_text(point))
            pdf.set_x(15)

    # Section 4: Event Classification & Risk Factors
    pdf.ln(6)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(88, 166, 255)
    pdf.cell(0, 8, "EVENT CLASSIFICATION & FACTORS", ln=True)
    pdf.ln(2)

    # Classification
    class_data = data.get("classification", {"name": "Normal", "color": "#39d353"})
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(36, 41, 47)
    pdf.cell(45, 6, "System Classification:")
    pdf.set_font("helvetica", "B", 10)
    
    # Class Color
    hex_color = class_data["color"].lstrip("#")
    c_r = int(hex_color[0:2], 16)
    c_g = int(hex_color[2:4], 16)
    c_b = int(hex_color[4:6], 16)
    pdf.set_text_color(c_r, c_g, c_b)
    pdf.cell(0, 6, clean_text(class_data["name"].upper()), ln=True)

    # Risk Factors
    pdf.set_x(15)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(36, 41, 47)
    pdf.cell(45, 6, "Identified Risk Factors:")
    pdf.set_font("helvetica", "", 10)
    
    factors = data.get("risk_factors", [])
    if not factors:
        pdf.cell(0, 6, "None", ln=True)
    else:
        pdf.cell(0, 6, clean_text(", ".join(factors)), ln=True)

    # Section 5: Recommendations
    pdf.ln(8)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(88, 166, 255)
    pdf.cell(0, 8, "ACTIONABLE RECOMMENDATION", ln=True)
    pdf.ln(2)

    # Recommendation callout box
    pdf.set_fill_color(240, 246, 255)  # Soft blue background
    pdf.set_draw_color(179, 211, 252)  # Light blue border
    pdf.rect(15, pdf.get_y(), 180, 20, "FD")

    pdf.set_xy(18, pdf.get_y() + 3)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(9, 105, 218)  # Deep blue text
    pdf.cell(0, 5, "RECOMMENDED OPERATOR ACTION:", ln=True)
    
    pdf.set_x(18)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(36, 41, 47)
    pdf.cell(0, 5, clean_text(str(data.get("recommendation", "Monitor area according to standard protocols."))))

    # Return PDF bytes
    return bytes(pdf.output())
