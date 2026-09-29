# Aqua Vision Documentation

Welcome to the comprehensive documentation repository for the **Aqua Vision AI-Powered Marine Debris & Underwater Anomaly Detection System**.

---

## Directory Organization

```
docs/
├── architecture/         # System architecture blueprints, slides, and flow diagrams
│   ├── architecture_diagram.html
│   ├── architecture_diagram.png
│   ├── architecture_diagram.svg
│   ├── sih_architecture_flow_corrected.drawio
│   ├── sih_architecture_flow_corrected.html
│   ├── sih_system_architecture.html
│   └── simple_architecture_slide.html
├── benchmarks/           # Quantitative benchmark curves, comparison charts, and metrics
│   ├── 100ep_full_BoxF1_curve.png
│   ├── 100ep_full_BoxPR_curve.png
│   ├── 100ep_full_confusion_matrix_normalized.png
│   ├── 100ep_full_results.png
│   ├── Comparative_Methods_Benchmark.csv
│   ├── final_100_epochs_summary.txt
│   └── performance_comparison_chart.png
├── integration/          # Subsystem and cross-module integration guides
│   └── README_MEMBER3_INTEGRATION.md
└── reports/              # Technical diagnostics, audit reports, and evaluation summaries
    ├── 4CLASS_TRAINING_READINESS_REPORT.md
    ├── Comparative_Methods_Benchmark.md
    ├── final_dataset_builder_report.md
    ├── FISHING_GEAR_DIAGNOSTIC_REPORT.md
    ├── FISHING_TARGETED_DATASET_AUDIT.md
    ├── SIH_Final_Evaluation_Report.md
    ├── SIH_MODEL_EVALUATION_AND_INTEGRATION_REPORT.md
    └── TILED_INFERENCE_DIAGNOSTIC_REPORT.md
```

---

## Key Highlights

- **Architecture**: End-to-end data pipeline from raw sonar acquisition, multi-stage physics-constrained pre-processing, YOLO detection, risk & priority ranking, to GIS trajectory generation.
- **Model Evaluation**: 100-epoch curriculum benchmarks across Shipwreck, Aircraft, Mine, and Fishing Gear targets.
- **Diagnostic Reports**: In-depth audits on small object detection, speckle filter trade-offs, and contrast enhancement.
