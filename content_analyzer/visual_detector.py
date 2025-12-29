"""
Local Chart Detection Module - NO API CALLS

Uses OpenCV and image analysis for chart detection without any LLM cost.
This module provides FREE chart detection as an alternative to expensive Gemini Vision API calls.

Author: SmartDoc AI
License: MIT
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LocalChartDetector:
    """
    Detects charts in images using OpenCV - completely free, no API calls.
    Detection Features:
        - Edge detection (Canny)
        - Line detection (HoughLinesP)
        - Circle detection (HoughCircles)
        - Contour analysis for shapes
        - Axis pattern recognition
    Detectable Chart Types:
        - Line charts (multiple organized lines)
        - Bar charts (rectangular shapes)
        - Pie charts (circular patterns)
        - Scatter plots (lines + circles)
        - Charts with axes (H/V line patterns)
        - Bubble charts (circles with variable size)
        - Zone diagrams (areas with color coding)
    """

    @staticmethod
    def detect_charts(image) -> Dict[str, Any]:
        """
        Detects complex charts and visualizations only - rejects tables, maps, and simple graphics.
        Returns a dictionary with detection results and features.
        """
        import time
        start_time = time.time()
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage

            # --- Image Preparation ---
            # Convert PIL image to OpenCV format if needed
            if isinstance(image, PILImage.Image):
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image_cv = image
            height, width = image_cv.shape[:2]
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # --- Edge Detection ---
            edges = cv2.Canny(gray, 50, 150)

            # --- Edge Density Calculation ---
            w_half = width // 2
            left_region = edges[:, :w_half]
            right_region = edges[:, w_half:]
            left_edge_density = np.sum(left_region > 0) / (left_region.shape[0] * left_region.shape[1])
            right_edge_density = np.sum(right_region > 0) / (right_region.shape[0] * right_region.shape[1])
            overall_edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            has_text_region = (
                (left_edge_density > 0.08 and right_edge_density > 0.08) or
                overall_edge_density > 0.15
            )

            # --- Line Detection ---
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10
            )
            line_count = len(lines) if lines is not None else 0
            diagonal_lines = 0
            line_angles = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if 10 < angle < 80 or 100 < angle < 170:
                        diagonal_lines += 1
                        line_angles.append(angle)

            # --- Circle Detection (Optimized) ---
            run_circles = diagonal_lines >= 1 or line_count >= 6 or overall_edge_density > 0.08
            circle_count = 0
            circles = None
            if run_circles:
                scale = 0.5 if max(height, width) > 800 else 1.0
                small_gray = cv2.resize(gray, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
                circles = cv2.HoughCircles(
                    small_gray,
                    cv2.HOUGH_GRADIENT,
                    dp=2.5,
                    minDist=60,
                    param1=50,
                    param2=55,
                    minRadius=18,
                    maxRadius=100
                )
                if circles is not None:
                    circle_count = circles.shape[2]

            # --- Color Diversity Analysis ---
            hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            color_peaks = np.sum(hist > np.mean(hist) * 2)

            # --- Contour Detection ---
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            significant_contours = 0
            rectangle_contours = 0
            similar_rectangles = []
            large_contours = 0
            small_scattered_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:
                    small_scattered_contours += 1
                elif 1500 < area < 40000:
                    significant_contours += 1
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    if len(approx) == 4:
                        rectangle_contours += 1
                        x, y, w, h = cv2.boundingRect(contour)
                        similar_rectangles.append((w, h, area))
                elif 40000 < area < 500000:
                    large_contours += 1

            # --- Bar Chart Pattern Detection ---
            bar_pattern = False
            if len(similar_rectangles) >= 6:
                widths = [r[0] for r in similar_rectangles]
                heights = [r[1] for r in similar_rectangles]
                width_std = np.std(widths)
                height_std = np.std(heights)
                avg_width = np.mean(widths)
                avg_height = np.mean(heights)
                if (width_std < avg_width * 0.3 or height_std < avg_height * 0.3):
                    bar_pattern = True

            # --- Line Classification ---
            horizontal_lines = 0
            vertical_lines = 0
            diagonal_lines = 0
            line_angles = []
            very_short_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if length < 50:
                        very_short_lines += 1
                        continue
                    if length < 80:
                        continue
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    line_angles.append(angle)
                    if angle < 10 or angle > 170:
                        horizontal_lines += 1
                    elif 80 < angle < 100:
                        vertical_lines += 1
                    else:
                        diagonal_lines += 1
            angle_variance = np.var(line_angles) if len(line_angles) > 2 else 0

            # --- Debug Logging ---
            logger.debug(f"Chart detection features: lines={line_count}, diagonal_lines={diagonal_lines}, circles={circle_count}, horizontal_lines={horizontal_lines}, vertical_lines={vertical_lines}, color_peaks={color_peaks}, angle_variance={angle_variance}")

            # --- Chart Heuristics and Classification ---
            chart_types = []
            confidence = 0.0
            description = ""
            rejection_reason = ""

            # Negative checks (text slides, decorative backgrounds, tables)
            if has_text_region and circle_count < 2 and diagonal_lines < 2 and not bar_pattern:
                if small_scattered_contours > 100 or very_short_lines > 50:
                    rejection_reason = f"Text slide with decorative background (overall density: {overall_edge_density:.2%})"
                    logger.debug(f"Rejected: {rejection_reason}")
                    return _chart_result(False, 0.0, [], rejection_reason, line_count, circle_count, overall_edge_density)
            if very_short_lines > 50 and circle_count < 2 and diagonal_lines < 3 and line_count < 10:
                rejection_reason = f"Decorative network background ({very_short_lines} tiny lines, no data elements)"
                logger.debug(f"Rejected: {rejection_reason}")
                return _chart_result(False, 0.0, [], rejection_reason, line_count, circle_count, overall_edge_density)
            if horizontal_lines > 12 and vertical_lines > 12 and circle_count == 0 and diagonal_lines < 2:
                grid_lines = horizontal_lines + vertical_lines
                total_lines = line_count
                grid_ratio = grid_lines / max(total_lines, 1)
                if grid_ratio > 0.75:
                    rejection_reason = f"Simple table pattern (H:{horizontal_lines}, V:{vertical_lines})"
                    logger.debug(f"Rejected: {rejection_reason}")
                    return _chart_result(False, 0.0, [], rejection_reason, line_count, circle_count, overall_edge_density)

            # Positive chart heuristics (bubble, scatter, line, pie, bar, complex)
            # RELAXED: Detect as line chart if 2+ diagonal lines and angle variance > 40, or 1+ diagonal line and 1+ axis
            if (
                (diagonal_lines >= 2 and angle_variance > 40) or
                (diagonal_lines >= 1 and (horizontal_lines >= 1 or vertical_lines >= 1))
            ):
                chart_types.append("line_chart")
                confidence = max(confidence, min(0.88, 0.6 + (diagonal_lines / 40)))
                if (horizontal_lines >= 1 or vertical_lines >= 1):
                    confidence = min(0.95, confidence + 0.08)
                if not description:
                    description = f"Line chart: {diagonal_lines} diagonal lines, axes: {horizontal_lines+vertical_lines}, variance: {angle_variance:.0f}"
            if circle_count >= 5:
                chart_types.append("bubble_chart")
                confidence = min(0.92, 0.70 + (min(circle_count, 20) * 0.01))
                description = f"Bubble chart: {circle_count} circles"
                if color_peaks > 5:
                    confidence = min(0.95, confidence + 0.1)
                    description += f", {int(color_peaks)} color zones"
                if large_contours > 2:
                    confidence = min(0.97, confidence + 0.05)
                    chart_types.append("zone_diagram")
                    description += f", {large_contours} colored regions"
            elif circle_count >= 3 and diagonal_lines > 2:
                chart_types.append("scatter_plot")
                confidence = max(confidence, 0.75)
                description = f"Scatter plot: {circle_count} data points"
            if circle_count > 0 and circle_count < 5:
                if "bubble_chart" not in chart_types:
                    chart_types.append("pie_chart")
                    confidence = max(confidence, 0.80)
                    if not description:
                        description = f"Pie chart: {circle_count} circular pattern(s)"
            if bar_pattern and rectangle_contours >= 6:
                chart_types.append("bar_chart")
                confidence = max(confidence, 0.75 + (min(rectangle_contours, 12) / 40))
                if not description:
                    description = f"Bar chart: {rectangle_contours} bars"
            if circle_count >= 3 and large_contours >= 2 and color_peaks > 5:
                chart_types.append("complex_visualization")
                confidence = max(confidence, 0.85)
                if not description:
                    description = "Complex visualization with zones and data points"
            has_moderate_axes = (1 <= horizontal_lines <= 6 or 1 <= vertical_lines <= 6)
            has_real_data = (circle_count >= 3 or diagonal_lines >= 2 or bar_pattern)
            if has_moderate_axes and has_real_data and confidence > 0.3:
                confidence = min(0.90, confidence + 0.10)
                if not description:
                    description = f"Chart with axes and data elements"

            # Final chart determination
            strong_indicator = (
                (diagonal_lines >= 2 and angle_variance > 40) or
                (diagonal_lines >= 1 and (horizontal_lines >= 1 or vertical_lines >= 1)) or
                circle_count >= 5 or
                (circle_count >= 3 and large_contours >= 2) or
                bar_pattern or
                (circle_count >= 3 and color_peaks > 5)
            )
            has_chart = (
                len(chart_types) > 0 and
                confidence > 0.4 and
                strong_indicator
            )
            total_time = time.time() - start_time
            if has_chart:
                logger.info(f"?? OpenCV detection: {total_time*1000:.0f}ms (lines:{line_count}, diagonal_lines:{diagonal_lines}, circles:{circle_count}, axes:{horizontal_lines+vertical_lines}, angle_variance:{angle_variance})")
            else:
                logger.debug(f"?? OpenCV detection: {total_time*1000:.0f}ms (rejected)")
            return {
                'has_chart': has_chart,
                'confidence': float(confidence),
                'chart_types': list(set(chart_types)),
                'description': description or "Potential chart detected",
                'features': {
                    'lines': line_count,
                    'diagonal_lines': diagonal_lines,
                    'circles': circle_count,
                    'contours': significant_contours,
                    'rectangles': rectangle_contours,
                    'horizontal_lines': horizontal_lines,
                    'vertical_lines': vertical_lines,
                    'angle_variance': float(angle_variance),
                    'bar_pattern': bar_pattern,
                    'large_contours': large_contours,
                    'color_peaks': int(color_peaks),
                    'text_region': has_text_region,
                    'very_short_lines': very_short_lines,
                    'overall_edge_density': float(overall_edge_density),
                    'detection_time_ms': float(total_time * 1000)
                }
            }
        except ImportError as e:
            logger.warning(f"OpenCV not installed: {e}")
            logger.info("Install with: pip install opencv-python")
            return {
                'has_chart': False,
                'confidence': 0.0,
                'chart_types': [],
                'description': 'OpenCV required for local detection',
                'features': {},
                'error': 'opencv_not_installed'
            }
        except Exception as e:
            logger.error(f"Chart detection error: {e}")
            return {
                'has_chart': False,
                'confidence': 0.0,
                'chart_types': [],
                'description': f'Detection error: {str(e)}',
                'features': {},
                'error': str(e)
            }

def _chart_result(has_chart, confidence, chart_types, description, line_count, circle_count, overall_edge_density):
    """Helper to return a standard chart detection result dict."""
    return {
        'has_chart': has_chart,
        'confidence': confidence,
        'chart_types': chart_types,
        'description': description,
        'features': {
            'lines': line_count,
            'circles': circle_count,
            'overall_edge_density': float(overall_edge_density)
        }
    }

# Detection configuration thresholds (BALANCED - detect real charts, reject pure text)
DETECTION_CONFIG = {
    'min_circles_bubble_chart': 5,
    'min_circles_scatter': 3,
    'min_diagonal_lines': 5,           # Lowered from 8 for line charts
    'min_angle_variance': 150,         # Lowered from 200 for line charts
    'min_rectangle_contours': 6,
    'min_confidence_threshold': 0.4,   # Lowered from 0.5
    'max_grid_ratio': 0.75,
    'max_text_edge_density_both': 0.08,  # Both sides text
    'max_text_edge_density_overall': 0.15,  # Entire page text
    'min_very_short_lines_mesh': 50,
    'axis_confidence_bonus': 0.10,
    'min_line_length': 80,
    'contour_area_min': 1500,
    'contour_area_max': 40000,
    'large_contour_min': 40000,
    'large_contour_max': 500000,
    'circle_radius_min': 15,
    'circle_radius_max': 300,
    'min_bar_chart_bars': 6,
    'min_color_peaks': 5
}
