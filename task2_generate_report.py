"""
task2_generate_report.py
Generates a 2-page publication-grade PDF report for Task 2:
Feature Engineering, Model Optimization & Performance Comparison
Maincrafts Technology - AI & Machine Learning Internship
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Custom canvas that provides running headers and footers with total page counts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header
        self.drawString(54, 750, "MAINCRAFTS TECHNOLOGY  |  AI & Machine Learning Internship  —  Task 2 Report")
        self.setStrokeColor(colors.HexColor("#007acc"))
        self.setLineWidth(1)
        self.line(54, 744, 612 - 54, 744)

        # Footer
        self.setStrokeColor(colors.HexColor("#d0d0d0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Task 2: Feature Engineering, Model Optimization & Performance Comparison")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_task2_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    pdf_path = os.path.join(reports_dir, "Task2_Model_Optimization_Report.pdf")

    # Load metrics
    metrics_path = os.path.join(models_dir, "task2_comparison_metrics.json")
    with open(metrics_path, "r") as f:
        meta = json.load(f)
    results = meta["models"]

    # Document configuration (margins 54pt = 0.75 in)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0c2340')
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#2b5c8f')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor('#0c2340'),
        spaceBefore=7,
        spaceAfter=4
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#1f4e78'),
        spaceBefore=5,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#222222'),
        spaceBefore=1.5,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceBefore=1,
        spaceAfter=1.5
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#222222')
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0c2340')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Executive Summary, Preprocessing Rationale & Model Table
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("California Housing Price Predictor", title_style))
    story.append(Paragraph("Task 2: Feature Engineering, Model Optimization & Performance Comparison", subtitle_style))
    story.append(Spacer(1, 6))

    # Badge bar
    badge_data = [
        [
            Paragraph("<b>Task:</b> AIML Task 2", table_cell),
            Paragraph("<b>Preprocessing:</b> StandardScaler", table_cell),
            Paragraph(f"<b>Top Core Model:</b> Decision Tree", table_cell),
            Paragraph(f"<b>Best Test R²:</b> {results['Decision Tree']['Test_R2']:.4f}", table_cell),
            Paragraph(f"<b>Best RMSE:</b> ${results['Decision Tree']['Test_RMSE_USD']:,.0f}", table_cell)
        ]
    ]
    t_badge = Table(badge_data, colWidths=[90, 120, 114, 85, 95])
    t_badge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eaf2f8')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#b3cde0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbdbe6')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_badge)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Executive Summary & Objective", h1_style))
    story.append(Paragraph(
        "Building upon the baseline established in Task 1, this task demonstrates how machine learning engineers optimize "
        "predictive models in real-world workflows. We implemented <b>Feature Scaling via StandardScaler</b> and trained multiple "
        "algorithms (<b>Linear Regression</b>, <b>Ridge Regression</b>, and <b>Decision Tree Regressor</b>) to evaluate whether "
        "regularization or non-linear splitting delivers measurable performance improvements on the California Housing dataset. "
        "A structured comparison framework was developed to balance predictive accuracy against model complexity and overfitting risk.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Data Preprocessing: Feature Scaling (StandardScaler)", h1_style))
    story.append(Paragraph(
        "<b>Why Scaling Matters:</b> Features in the California Housing dataset span vastly different numerical scales (e.g., "
        "<i>Population</i> ranges up to 35,000, while <i>AveBedrms</i> centers around 1.1). Without standardization, features with larger "
        "magnitudes disproportionately dominate gradient updates and penalization terms. Applying <i>StandardScaler</i> "
        "(z-score normalization: z = (x - μ) / σ) centers each feature at zero mean and unit variance, ensuring stable optimization "
        "and fair weight penalties across all predictors.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Model Selection Architecture & Hypotheses", h1_style))
    story.append(Paragraph("• <b>Linear Regression (Baseline):</b> Minimizes residual sum of squares; provides linear benchmark.", bullet_style))
    story.append(Paragraph("• <b>Ridge Regression (α = 1.0):</b> Introduces L2 regularization penalty (λ Σ w²) to shrink collinear coefficients (such as AveRooms and AveBedrms) and reduce model variance.", bullet_style))
    story.append(Paragraph("• <b>Decision Tree Regressor (max_depth = 5):</b> Employs recursive binary splits based on variance reduction to model non-linear interactions without requiring parametric assumptions.", bullet_style))
    story.append(Paragraph("• <b>Random Forest Regressor (Ensemble Benchmark):</b> Aggregates 100 decorrelated trees to illustrate state-of-the-art ensemble variance reduction.", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Structured Model Performance Comparison Table", h1_style))
    
    # Table data
    table_data = [
        [
            Paragraph("Algorithm", table_header),
            Paragraph("Test RMSE", table_header),
            Paragraph("Test USD Error", table_header),
            Paragraph("Test R²", table_header),
            Paragraph("Test MAE", table_header),
            Paragraph("Train R²", table_header),
            Paragraph("Overfitting Gap", table_header)
        ],
        [
            Paragraph("<b>Linear Regression</b>", table_cell),
            Paragraph(f"{results['Linear Regression']['Test_RMSE']:.4f}", table_cell),
            Paragraph(f"${results['Linear Regression']['Test_RMSE_USD']:,.0f}", table_cell),
            Paragraph(f"<b>{results['Linear Regression']['Test_R2']:.4f}</b>", table_cell),
            Paragraph(f"${results['Linear Regression']['Test_MAE_USD']:,.0f}", table_cell),
            Paragraph(f"{results['Linear Regression']['Train_R2']:.4f}", table_cell),
            Paragraph(f"+{results['Linear Regression']['Overfitting_R2_Gap']:.4f}", table_cell)
        ],
        [
            Paragraph("<b>Ridge Regression (α=1.0)</b>", table_cell),
            Paragraph(f"{results['Ridge Regression']['Test_RMSE']:.4f}", table_cell),
            Paragraph(f"${results['Ridge Regression']['Test_RMSE_USD']:,.0f}", table_cell),
            Paragraph(f"<b>{results['Ridge Regression']['Test_R2']:.4f}</b>", table_cell),
            Paragraph(f"${results['Ridge Regression']['Test_MAE_USD']:,.0f}", table_cell),
            Paragraph(f"{results['Ridge Regression']['Train_R2']:.4f}", table_cell),
            Paragraph(f"+{results['Ridge Regression']['Overfitting_R2_Gap']:.4f}", table_cell)
        ],
        [
            Paragraph("<b>Decision Tree (depth=5)</b>", table_cell_bold),
            Paragraph(f"<b>{results['Decision Tree']['Test_RMSE']:.4f}</b>", table_cell_bold),
            Paragraph(f"<b>${results['Decision Tree']['Test_RMSE_USD']:,.0f}</b>", table_cell_bold),
            Paragraph(f"<b>{results['Decision Tree']['Test_R2']:.4f}</b>", table_cell_bold),
            Paragraph(f"<b>${results['Decision Tree']['Test_MAE_USD']:,.0f}</b>", table_cell_bold),
            Paragraph(f"{results['Decision Tree']['Train_R2']:.4f}", table_cell_bold),
            Paragraph(f"+{results['Decision Tree']['Overfitting_R2_Gap']:.4f}", table_cell_bold)
        ],
        [
            Paragraph("<b>Random Forest (Benchmark)</b>", table_cell),
            Paragraph(f"{results['Random Forest (Benchmark)']['Test_RMSE']:.4f}", table_cell),
            Paragraph(f"${results['Random Forest (Benchmark)']['Test_RMSE_USD']:,.0f}", table_cell),
            Paragraph(f"<b>{results['Random Forest (Benchmark)']['Test_R2']:.4f}</b>", table_cell),
            Paragraph(f"${results['Random Forest (Benchmark)']['Test_MAE_USD']:,.0f}", table_cell),
            Paragraph(f"{results['Random Forest (Benchmark)']['Train_R2']:.4f}", table_cell),
            Paragraph(f"+{results['Random Forest (Benchmark)']['Overfitting_R2_Gap']:.4f}", table_cell)
        ],
    ]
    t_comp = Table(table_data, colWidths=[114, 60, 75, 55, 70, 60, 70])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0c2340')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#0c2340')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#e8f5e9')), # Highlight Decision tree
        ('ROWBACKGROUNDS', (0,1), (-1,2), [colors.white, colors.HexColor('#f8f9fa')]),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#fdf2e9')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_comp)

    # Page 1 Break
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Visual Validation, Overfitting Analysis & Selection Justification
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("5. Visual Performance Validation & Benchmarks", h1_style))
    
    # Comparison Bar Chart Image
    bar_img = os.path.join(assets_dir, "task2_model_comparison_bar.png")
    if os.path.exists(bar_img):
        story.append(Image(bar_img, width=4.9*inch, height=1.9*inch))
    story.append(Spacer(1, 4))

    # Dual Diagnostics (Actual vs Pred + Feature Importance)
    diag_data = [
        [
            Image(os.path.join(assets_dir, "task2_actual_vs_predicted.png"), width=3.3*inch, height=2.3*inch),
            Image(os.path.join(assets_dir, "task2_decision_tree_feature_importance.png"), width=3.3*inch, height=2.3*inch)
        ]
    ]
    t_diag = Table(diag_data, colWidths=[252, 252])
    t_diag.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_diag)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Analysis of Overfitting vs. Generalization", h1_style))
    story.append(Paragraph(
        "• <b>Linear & Ridge Stability:</b> Both linear models exhibit near-identical test performance (R² = 57.58%, RMSE = $74.5k) "
        "and small train-test divergence (ΔR² = 0.0368), confirming high stability but clear underfitting due to rigid linear assumptions.<br/>"
        "• <b>Decision Tree Regularization:</b> By constraining depth to <i>max_depth = 5</i>, the Decision Tree prevents memorization "
        "while capturing non-linear regional thresholds. The overfitting gap is minimal (ΔR² = 0.0379), achieving <b>R² = 59.97%</b> and "
        "reducing RMSE to <b>$72,423</b> (a ~$2,135 error reduction over linear baselines).<br/>"
        "• <b>Ensemble Supremacy:</b> Random Forest reduces test RMSE to $54,450 (R² = 77.38%), illustrating the advantage of aggregating "
        "multiple decorrelated trees to suppress variance.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("7. Final Model Selection & Technical Justification", h1_style))
    story.append(Paragraph(
        "<b>Selected Model: Decision Tree Regressor (max_depth = 5)</b><br/>"
        "1. <b>Superior Accuracy:</b> Outperforms Linear and Ridge regression across all error metrics (MAE: $52,226 vs $53,320; RMSE: $72,423 vs $74,558).<br/>"
        "2. <b>Non-Linear Partitioning:</b> As seen in the feature importance analysis, the tree partitions on <i>MedInc</i> (64.5% weight) "
        "and geographical splits (<i>Latitude/Longitude</i>), capturing coastal price surges that linear models blur.<br/>"
        "3. <b>Controlled Complexity:</b> The depth-5 constraint preserves high interpretability and prevents overfitting, making it an ideal "
        "production candidate.",
        body_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Task 2 Report compiled successfully: {pdf_path}")
    return pdf_path

if __name__ == '__main__':
    build_task2_report()
