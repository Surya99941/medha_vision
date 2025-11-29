# MedhaVision  
**AI-Powered Medical Imaging Triage Assistant**

MedhaVision is an AI-driven medical imaging triage system designed to accelerate anomaly detection in X-ray and CT scans. Built for healthcare professionals, it combines computer vision and conversational AI to deliver instant, intelligent, and interactive diagnostic support.

---

## Overview

Medical imaging analysis is often delayed in smaller clinics due to limited radiology expertise and time-consuming report generation. MedhaVision addresses these challenges by providing:

- Real-time anomaly detection
- AI-assisted reporting
- Interactive medical Q&A
- Seamless integration as a SaaS application for clinics and diagnostic centers

---

## Key Features

### Automatic Scan Type Identification

The system automatically detects input scan type and routes it to the appropriate model.

| Input Type | Identified As | Model Used |
|------------|---------------|------------|
| X-ray      | Bone fracture detection | YOLOv8-F |
| Brain CT   | Tumor and mass detection | YOLOv8-T |

---

### AI-Generated Medical Reports

Translates detection outputs into clear, clinically relevant summaries.

---

### Visual Annotations

- Overlays bounding boxes on scan images
- Displays confidence scores and precise localization
- Helps doctors quickly interpret AI findings

---

### Interactive AI-Driven Q&A

Doctors can ask context-based questions about the scan, such as:

- "How many anomalies were detected?"
- "What is the confidence score for each lesion?"
- "Is the detected tumor likely malignant based on features?"

The system responds using real analysis data, not generic AI responses.

---


## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend   | Python, FastAPI |
| AI/ML Models | YOLO, LangChain, OpenAI GPT-4 |
| Database  | PostgreSQL |

---

## Business Model  
**SaaS solution for medical facilities**

| Target | Value |
|--------|-------|
| Small Clinics | Instant AI triaging without specialist radiologists |
| Diagnostic Centers | Faster reporting and documentation |
| Hospitals | PACS/EMR integration and AI-assisted workflows |

Integrates with existing imaging hardware and can act as an AI assistant in radiology workflows.

---

## Future Enhancements

- Support for MRI, Ultrasound, and Retina scans  
- Treatment recommendation suggestions  
- Severity grading using clinical guidelines  
- Physician collaboration dashboard  

---
