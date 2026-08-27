"""
generate_report.py
Generates a professional 3-page PDF report for AIML Task 1:
California Housing Price Predictor (Linear Regression)
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
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
        self.drawString(54, 750, "MAINCRAFTS TECHNOLOGY  |  AI & Machine Learning Internship  —  Task 1 Report")
        self.setStrokeColor(colors.HexColor("#007acc"))
        self.setLineWidth(1)
        self.line(54, 744, 612 - 54, 744)

        # Footer
        self.setStrokeColor(colors.HexColor("#d0d0d0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "California Housing Price Predictor  •  Linear Regression Model")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    pdf_path = os.path.join(reports_dir, "California_Housing_Linear_Regression_Report.pdf")

    # Load metrics
    metrics_file = os.path.join(models_dir, "metrics.json")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

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
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0c2340')
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2b5c8f')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0c2340'),
        spaceBefore=8,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#1f4e78'),
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#222222'),
        spaceBefore=2,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceBefore=1,
        spaceAfter=2
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#222222')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # ==========================================
    # PAGE 1: Title, Executive Summary & Dataset Overview
    # ==========================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("California Housing Price Predictor", title_style))
    story.append(Paragraph("Task 1: End-to-End Linear Regression Modeling, Diagnostics & Evaluation", subtitle_style))
    story.append(Spacer(1, 8))

    # Metadata badge bar
    badge_data = [
        [
            Paragraph("<b>Framework:</b> scikit-learn", table_cell),
            Paragraph("<b>Dataset:</b> California Housing", table_cell),
            Paragraph("<b>Rows:</b> 20,640", table_cell),
            Paragraph(f"<b>Test R²:</b> {metrics['R2_Test']:.3f}", table_cell),
            Paragraph(f"<b>RMSE:</b> ${metrics['RMSE']*100000:,.0f}", table_cell)
        ]
    ]
    t_badge = Table(badge_data, colWidths=[105, 115, 80, 85, 119])
    t_badge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eaf2f8')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#b3cde0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbdbe6')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_badge)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Summary", h1_style))
    exec_summary_text = (
        "This project introduces the complete machine learning lifecycle by designing, training, and rigorously "
        "evaluating a <b>Multiple Linear Regression</b> baseline to predict median house values across California districts. "
        "Using the 1990 U.S. Census California Housing dataset (20,640 instances, 8 continuous features), the model explains "
        f"<b>{metrics['R2_Test']*100:.1f}% of target variance</b> ($R^2 = {metrics['R2_Test']:.4f}$) with a Mean Absolute Error "
        f"of <b>${metrics['MAE']*100000:,.2f}</b> and Root Mean Squared Error of <b>${metrics['RMSE']*100000:,.2f}</b>. "
        "Median Income (<i>MedInc</i>) emerged as the dominant positive driver, while geographic coordinates capture regional "
        "cost differentials. The study documents key linear assumptions, residual patterns, and a roadmap for non-linear models."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. California Housing Dataset Architecture", h1_style))
    ds_intro = (
        "The dataset represents census block groups (clusters of 600 to 3,000 individuals). "
        "A rigorous data hygiene inspection confirmed <b>zero missing values (0 / 185,760 cells)</b>, uniform float64 data types, "
        "and clean data integrity. Features are defined below:"
    )
    story.append(Paragraph(ds_intro, body_style))
    story.append(Spacer(1, 4))

    # Feature definitions table
    feat_data = [
        [Paragraph("Feature", table_header), Paragraph("Description", table_header), Paragraph("Unit / Scale", table_header), Paragraph("Mean ± Std", table_header)],
        [Paragraph("<b>MedInc</b>", table_cell), Paragraph("Median income in block group", table_cell), Paragraph("$10,000s (e.g. 3.87 = $38.7k)", table_cell), Paragraph("3.87 ± 1.90", table_cell)],
        [Paragraph("<b>HouseAge</b>", table_cell), Paragraph("Median house age in block group", table_cell), Paragraph("Years (1 to 52)", table_cell), Paragraph("28.6 ± 12.6", table_cell)],
        [Paragraph("<b>AveRooms</b>", table_cell), Paragraph("Average rooms per household", table_cell), Paragraph("Rooms / household", table_cell), Paragraph("5.43 ± 2.47", table_cell)],
        [Paragraph("<b>AveBedrms</b>", table_cell), Paragraph("Average bedrooms per household", table_cell), Paragraph("Bedrooms / household", table_cell), Paragraph("1.10 ± 0.47", table_cell)],
        [Paragraph("<b>Population</b>", table_cell), Paragraph("Block group total population", table_cell), Paragraph("Count of residents", table_cell), Paragraph("1,425 ± 1,132", table_cell)],
        [Paragraph("<b>AveOccup</b>", table_cell), Paragraph("Average house occupancy", table_cell), Paragraph("Household members count", table_cell), Paragraph("3.07 ± 10.39", table_cell)],
        [Paragraph("<b>Latitude</b>", table_cell), Paragraph("Block group latitude coordinate", table_cell), Paragraph("Degrees North (32.5° – 41.9°)", table_cell), Paragraph("35.63 ± 2.14", table_cell)],
        [Paragraph("<b>Longitude</b>", table_cell), Paragraph("Block group longitude coordinate", table_cell), Paragraph("Degrees West (-124.3° – -114.3°)", table_cell), Paragraph("-119.57 ± 2.00", table_cell)],
        [Paragraph("<b>MedHouseVal (Y)</b>", table_cell), Paragraph("Target: Median house value", table_cell), Paragraph("$100,000s (Capped at 5.0 = $500k)", table_cell), Paragraph("2.07 ± 1.15", table_cell)],
    ]
    t_feat = Table(feat_data, colWidths=[100, 190, 124, 90])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0c2340')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#0c2340')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Target Distribution Characteristics", h1_style))
    story.append(Paragraph(
        "The target variable <b>MedHouseVal</b> has a mean of <b>$206,855</b> and standard deviation of <b>$115,395</b>. "
        "A critical property of this dataset is an explicit <b>ceiling cap at $500,000 (5.000)</b> representing 965 census blocks (4.67%). "
        "This truncation introduces minor boundary distortion in linear estimations near the ceiling.",
        body_style
    ))
    
    # End of Page 1
    story.append(PageBreak())

    # ==========================================
    # PAGE 2: Exploratory Data Analysis & Visual Insights
    # ==========================================
    story.append(Spacer(1, 5))
    story.append(Paragraph("4. Exploratory Data Analysis (EDA) & Correlations", h1_style))
    story.append(Paragraph(
        "Bivariate correlation analysis identifies linear associations and detects multicollinearity between regressors:",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Heatmap Image
    heatmap_img = os.path.join(assets_dir, "correlation_heatmap.png")
    if os.path.exists(heatmap_img):
        story.append(Image(heatmap_img, width=4.8*inch, height=3.2*inch))
    story.append(Spacer(1, 6))

    # Correlation bullet observations
    story.append(Paragraph("Key Correlation Insights:", h2_style))
    story.append(Paragraph("• <b>Primary Driver:</b> <i>MedInc</i> correlates most strongly with house prices (<b>r = +0.69</b>), confirming that local purchasing power is the foremost predictor.", bullet_style))
    story.append(Paragraph("• <b>Multicollinearity Pair:</b> <i>AveRooms</i> and <i>AveBedrms</i> display heavy collinearity (<b>r = +0.85</b>). In unregularized OLS, this causes inflated standard errors and counter-intuitive opposing coefficients.", bullet_style))
    story.append(Paragraph("• <b>Geographic Orientation:</b> <i>Latitude</i> and <i>Longitude</i> correlate negatively (<b>r = -0.92</b>), tracking California's northwest-to-southeast geographic tilt.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Spatial & Feature Distribution Analysis", h1_style))
    geo_img = os.path.join(assets_dir, "geo_distribution.png")
    if os.path.exists(geo_img):
        story.append(Image(geo_img, width=4.8*inch, height=2.4*inch))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Geographic Premium:</b> As visualized above, maximum median home values ($500k) concentrate heavily along the coastal corridor "
        "(San Francisco Bay Area and coastal Los Angeles / Orange County), while inland Central Valley communities reflect lower valuations. "
        "This nonlinear spatial pattern highlights the necessity for geospatial feature engineering or tree-based splits.",
        body_style
    ))

    # End of Page 2
    story.append(PageBreak())

    # ==========================================
    # PAGE 3: Model Evaluation, Diagnostics & Improvement Roadmap
    # ==========================================
    story.append(Spacer(1, 5))
    story.append(Paragraph("6. Model Training & Evaluation Metrics", h1_style))
    story.append(Paragraph(
        "The model was trained using an 80/20 train-test split (16,512 train samples, 4,128 test samples) with random_state=42. "
        "Standard Ordinary Least Squares (OLS) closed-form optimization was performed:",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Metrics Scorecard Table
    metrics_table_data = [
        [Paragraph("Evaluation Metric", table_header), Paragraph("Formula", table_header), Paragraph("Model Value", table_header), Paragraph("USD Equivalent", table_header), Paragraph("Interpretation", table_header)],
        [Paragraph("<b>MAE</b>", table_cell), Paragraph("Σ|y - ŷ| / n", table_cell), Paragraph(f"<b>{metrics['MAE']:.4f}</b>", table_cell), Paragraph(f"<b>${metrics['MAE']*100000:,.2f}</b>", table_cell), Paragraph("Average absolute prediction deviation", table_cell)],
        [Paragraph("<b>MSE</b>", table_cell), Paragraph("Σ(y - ŷ)² / n", table_cell), Paragraph(f"<b>{metrics['MSE']:.4f}</b>", table_cell), Paragraph(f"—", table_cell), Paragraph("Variance of prediction residuals", table_cell)],
        [Paragraph("<b>RMSE</b>", table_cell), Paragraph("√MSE", table_cell), Paragraph(f"<b>{metrics['RMSE']:.4f}</b>", table_cell), Paragraph(f"<b>${metrics['RMSE']*100000:,.2f}</b>", table_cell), Paragraph("Standard deviation of unexplained variance", table_cell)],
        [Paragraph("<b>R² (Test)</b>", table_cell), Paragraph("1 - (SSres / SStot)", table_cell), Paragraph(f"<b>{metrics['R2_Test']:.4f}</b>", table_cell), Paragraph("<b>57.58%</b>", table_cell), Paragraph("Proportion of variance explained by model", table_cell)],
        [Paragraph("<b>R² (Train)</b>", table_cell), Paragraph("1 - (SSres / SStot)", table_cell), Paragraph(f"<b>{metrics['R2_Train']:.4f}</b>", table_cell), Paragraph("<b>61.26%</b>", table_cell), Paragraph("Training fit; no severe variance/overfitting", table_cell)],
    ]
    t_metrics = Table(metrics_table_data, colWidths=[70, 95, 75, 95, 169])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0c2340')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#0c2340')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 8))

    story.append(Paragraph("7. Model Diagnostics & Error Analysis", h1_style))
    
    # Side-by-side or stacked diagnostic images
    # We can embed actual_vs_predicted and feature_importance side-by-side or stacked
    diag_data = [
        [
            Image(os.path.join(assets_dir, "actual_vs_predicted.png"), width=3.3*inch, height=2.4*inch),
            Image(os.path.join(assets_dir, "feature_importance.png"), width=3.3*inch, height=2.4*inch)
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
    story.append(Spacer(1, 6))

    story.append(Paragraph("8. Diagnostics Summary & Next-Step Improvements", h1_style))
    story.append(Paragraph(
        "<b>Diagnostic Observations:</b><br/>"
        "1. <i>Ceiling Saturation:</i> The horizontal line at y = 5.0 in the Actual vs. Predicted plot represents the dataset cap. "
        "The model underpredicts top-bracket homes because linear functions cannot account for truncated labels.<br/>"
        "2. <i>Standardized Influence:</i> <b>MedInc (+0.854)</b> is by far the highest positive determinant of property value, "
        "followed by positive age effects and negative inland coordinate shifts.",
        body_style
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Actionable Improvement Roadmap:</b><br/>"
        "• <b>Feature Engineering:</b> Compute ratio features (e.g., <i>Bedrooms_per_Room = AveBedrms / AveRooms</i>) and spatial geodesic proximity to coastline / tech hubs.<br/>"
        "• <b>Regularization:</b> Apply <i>Ridge / Lasso / ElasticNet</i> to constrain collinear coefficients between room counts.<br/>"
        "• <b>Non-Linear Ensembles:</b> Upgrade to tree-based algorithms (Random Forest, LightGBM, XGBoost) to capture non-linear regional interactions and lift $R^2$ beyond 0.82.",
        body_style
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report compiled successfully: {pdf_path}")
    return pdf_path

if __name__ == '__main__':
    build_pdf_report()
