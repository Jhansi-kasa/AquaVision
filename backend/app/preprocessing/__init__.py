"""
Preprocessing module for Side-Scan Sonar (SSS) imagery.
Provides modular, deterministic contrast enhancement and speckle reduction
for YOLO object detection and acoustic analysis.
"""

from .sonar_preprocessor import SonarPreprocessor, preprocess_sonar_image

__all__ = ["SonarPreprocessor", "preprocess_sonar_image"]
