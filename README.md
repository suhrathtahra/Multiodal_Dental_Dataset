# MMDental: A Multimodal Dental X-ray Dataset

## Reproducible Dataset Repository for Predictive and Assistive Dentistry AI

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Repository Structure](#repository-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Workflow & Usage Guide](#workflow--usage-guide)
- [Technical Specifications](#technical-specifications)
- [Quality Assurance](#quality-assurance)
- [Dataset Metadata](#dataset-metadata)
- [Publication & Reproducibility](#publication--reproducibility)
- [Contact & Citation](#contact--citation)

---

## 🎯 Overview

This repository contains the **MMDental dataset** and supporting code — a multimodal collection of dental radiographs annotated for **predictive and assistive dentistry AI**. It combines periapical (RVG) X-rays collected from three local dental clinics with panoramic (OPG) X-rays, annotated across a **21-class clinical schema** spanning anatomical structures, pathological findings, and clinical/developmental status.

The dataset and accompanying scripts implement **structured annotation handling, dual-reviewer clinical validation, and format conversion (YOLO/COCO)** to support object detection and semantic segmentation research in dental radiography.

**Key Features:**

- ✅ Real clinical data from three dentists plus a public OPG source
- ✅ 21-class annotation schema (anatomical + pathological + clinical status)
- ✅ Dual-reviewer quality assurance (primary investigator + board-certified dentist)
- ✅ YOLO `.txt` and COCO-compatible JSON annotation formats
- ✅ Full metadata and documentation for reproducibility

---

## 🎓 Project Objectives

This project addresses the need for:

1. **Open, clinically validated dental imaging data** — most existing public dental datasets label only a handful of broad categories, limiting downstream model utility.
2. **A unified annotation framework** — combining anatomical ground truth, pathology, and clinical/developmental status in a single schema, enabling multi-task learning.
3. **Reproducible data preparation** — documented, scriptable steps from raw clinical archives to a training-ready dataset.
4. **Predictive dentistry research** — a foundation for models that go beyond single-lesion detection toward full automated dental charting and risk stratification.

---

## 🦷 Dataset Description

### Modalities and Sources

| Modality | Images | Source |
|---|---|---|
| Periapical (RVG) X-ray | 849 | Collected from three local dentists (Dr. Homaira: 155, Dr. Tasmia: 215, Dr. Tashfia: 479) |
| Panoramic (OPG) X-ray | 301 | Sourced from Zannah et al. (2024), IQBAL'S Dental Clinic, Bogura, Bangladesh |
| **Total** | **1,150** | — |

### 21-Class Annotation Schema

| Category | Classes |
|---|---|
| **Anatomical Foundations (1–7)** | Enamel, Dentine, Cementum, Pulp, Periodontal Ligament (PDL), Alveolar Bone, Apical Foramen |
| **Pathological Indicators** | Caries, Periapical Lesion, Bone Loss, Fracture Tooth, Attrition, Failed Restoration, Root Resorption |
| **Clinical Status & Developmental Markers** | Healthy Teeth, Missing Tooth, Endodontically Treated, Impacted Tooth, Primary Tooth, Permanent Tooth, Implant |

Every bounding box was drawn by a primary investigator using **CVAT (Computer Vision Annotation Tool)** and subsequently reviewed, corrected, and confirmed by a **board-certified dentist**, producing over **12,000 individual annotations** with clinical diagnostic consensus.

### Repository Structure

```
MMDental/
├── Images/
│   ├── Pa-X-Ray (Dr Homaira)/      # 155 periapical images
│   ├── Pa-X-Ray (Dr Tasmia)/       # 215 periapical images
│   ├── Pa-X-Ray (Dr Tashfia)/      # 479 periapical images
│   └── OPG/                        # 301 panoramic images
│
├── Annotations/
│   ├── Pa-X-Ray (Dr Homaira)/      # YOLO .txt / COCO JSON annotations
│   ├── Pa-X-Ray (Dr Tasmia)/
│   ├── Pa-X-Ray (Dr Tashfia)/
│   └── OPG/                        # Segmentation masks
│
├── metadata.csv                    # Complete dataset inventory
├── classes.txt                     # 21-class label map
└── README.md
```

### Data Specifications

- **Image Format:** 8-bit RGB JPEG, native periapical digital sensor resolution
- **Annotation Format:** Individual JSON files (COCO-compatible structure), exportable to YOLO `.txt` via CVAT
- **Annotation Type:** Rectangular bounding boxes
- **Annotation Tool:** CVAT v2.4

---

## 📁 Repository Contents

### Data Files

- `Images/` — Periapical and panoramic radiographs, organized by contributing dentist/source
- `Annotations/` — Per-image bounding box files (JSON/YOLO), one set per image directory
- `metadata.csv` — Full inventory with `image_id`, `class_name`, `image_path`, `annotation_path`, dimensions, and file size
- `classes.txt` — The 21 class names, in numerical ID order

### Example Annotation (COCO-style JSON)

```json
{
  "image_id": "0000247_MD_Nasrin_Ahmed_20241113_134249_3_1",
  "class_id": 8,
  "class_name": "Caries",
  "bbox": [x_min, y_min, x_max, y_max],
  "area": 88254,
  "iscrowd": 0
}
```

---

## 🔧 Installation & Dependencies

### System Requirements

- **OS:** Linux/Windows/macOS
- **Python:** 3.8+
- **Storage:** ~1 GB for the full uncompressed dataset

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/suhrathtahra/Multiodal_Dental_Dataset.git
cd Multiodal_Dental_Dataset

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.3.0
numpy>=1.21.0
pillow>=9.0.0
opencv-python>=4.5.0
matplotlib>=3.4.0
```

---

## 🚀 Workflow & Usage Guide

### Loading the Dataset

```python
import pandas as pd
import cv2
import json

# Load metadata
meta = pd.read_csv("metadata.csv")
print(f"Total samples: {len(meta)}")
print(f"Classes: {meta['class_name'].unique()}")

# Load a sample image and its annotation
row = meta.sample(1).iloc[0]
img = cv2.imread(row['image_path'])

with open(row['annotation_path'], 'r') as f:
    ann = json.load(f)
    bbox = ann['bbox']
```

### PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd

class MMDentalDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        self.meta = pd.read_csv(metadata_path)
        self.transform = transform
        self.classes = sorted(self.meta['class_name'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')

        with open(row['annotation_path'], 'r') as f:
            ann = json.load(f)

        label = self.class_to_idx[row['class_name']]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'bbox': torch.tensor(ann['bbox'], dtype=torch.float32),
            'label': label,
            'class_name': row['class_name']
        }
```

### Recommended Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

meta = pd.read_csv("metadata.csv")

train_df, temp_df = train_test_split(
    meta, test_size=0.3, stratify=meta['class_name'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['class_name'], random_state=42
)
```

---

## 📊 Technical Specifications

### Image Specifications

- **Format:** JPEG (RGB, 8-bit)
- **Source Hardware:** Digital intraoral X-ray sensor (Carestream RVG 6200 and equivalent systems)
- **Resolution:** Native periapical sensor dimensions (varies by device)

### Annotation Specifications

- **Format:** COCO-compatible JSON, YOLO `.txt` export available via CVAT
- **Bounding Boxes:** `[x_min, y_min, x_max, y_max]` in pixel coordinates
- **Classes:** 21, each with a unique numerical ID and hexadecimal color code

---

## ✅ Quality Assurance

### Two-Phase Annotation Review

1. **Initial Annotation** — performed by a primary investigator trained in dental radiograph interpretation and CVAT operation.
2. **Expert Review** — all annotations examined by a board-certified dentist, who performed:
   - **Additions** — labeling missed pathological findings
   - **Deletions** — removing erroneous annotations
   - **Confirmations** — approving clinically accurate labels

Ambiguous or borderline cases were resolved through consultation with an assisting dentist, ensuring the final annotations reflect **robust clinical diagnostic consensus** rather than a single annotator's judgment.

### Automated Checks

- ✅ File integrity (all images/annotations readable)
- ✅ 1:1 correspondence between images and annotation files
- ✅ Class ID / label map consistency across all annotation tasks
- ✅ Bounding box coordinates within image bounds

---

## 📝 Dataset Metadata

### metadata.csv Schema

| Column | Type | Description | Example |
|---|---|---|---|
| `image_id` | str | Unique identifier | `0000247_MD_Nasrin_Ahmed_...` |
| `class_name` | str | One of 21 dental classes | `Caries` |
| `image_path` | str | Relative path to image | `Images/Pa-X-Ray (Dr Homaira)/...jpg` |
| `annotation_path` | str | Path to annotation file | `Annotations/Pa-X-Ray (Dr Homaira)/...json` |
| `width`, `height` | int | Image dimensions (pixels) | native sensor resolution |
| `file_size_kb` | float | File size | varies |

---

## 🔬 Publication & Reproducibility

This dataset accompanies the manuscript:

> *A Multimodal Dental Dataset for Predictive and Assistive Dentistry AI Models*
> Tahra, S., Chowdhury, M.A., Kohinoor, M.S.R., Shorfuzzaman, M., Rahman, M.M.

### Reproducibility Features

- Version-controlled repository with full commit history
- Documented annotation workflow (CVAT project/task setup, class ordering, export steps)
- Dual-reviewer clinical validation process fully described
- Metadata file enabling exact reconstruction of train/val/test splits

### Data Availability

- **Repository:** [github.com/suhrathtahra/Multiodal_Dental_Dataset](https://github.com/suhrathtahra/Multiodal_Dental_Dataset)
- **Kaggle Mirror:** [kaggle.com/datasets/suhrathtahra/dental-dataset](https://www.kaggle.com/datasets/suhrathtahra/dental-dataset)
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Size:** ~956 MB (uncompressed), ~376 MB (compressed .zip)

---

## 📖 How to Cite

If you use this dataset in your research, please cite:

```bibtex
@article{tahra2026mmdental,
  author  = {Tahra, Suhrath and Chowdhury, Mehzabeen Azad and Kohinoor, Md. Saidur Rahman and Shorfuzzaman, Mohammad and Rahman, Md. Mahfuzur},
  title   = {A Multimodal Dental Dataset for Predictive and Assistive Dentistry AI Models},
  year    = {2026},
  howpublished = {\url{https://github.com/suhrathtahra/Multiodal_Dental_Dataset}}
}
```

---

## 📧 Contact & Support

**Repository Owner:** Suhrath Tahra
**Institution:** InteX Research Lab, Sylhet, Bangladesh
**Dataset:** [GitHub](https://github.com/suhrathtahra/Multiodal_Dental_Dataset) · [Kaggle](https://www.kaggle.com/datasets/suhrathtahra/dental-dataset)

### Contributing

Contributions, bug reports, and suggestions are welcome via GitHub Issues or Pull Requests.

### FAQ

**Q: Can I use a subset of the 21 classes for a simpler classification task?**
A: Yes — the `classes.txt` label map and per-annotation `class_id` make it straightforward to filter to any subset (e.g., pathology-only or anatomy-only).

**Q: Are the OPG images annotated with the same 21-class schema as the periapical images?**
A: The OPG subset (sourced from Zannah et al., 2024) uses segmentation masks rather than the full 21-class bounding box schema — see the original source for its annotation details.

**Q: What license is this dataset released under?**
A: CC BY 4.0 — free to share and adapt, with attribution to the associated publication.

---

## 📄 License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Radiograph collection:** Dr. Syeda Tasmia Kawser and Dr. Syeda Tasfia Kawser (North East Medical College Hospital), and Dr. Samia for clinical guidance during annotation.
- **OPG subset:** Sourced from Zannah et al. (2024), collected at IQBAL'S Dental Clinic, Bogura, Bangladesh.
- **Institutional support:** InteX Research Lab.

---

**Repository:** <https://github.com/suhrathtahra/Multiodal_Dental_Dataset>
**Status:** ✅ Publicly available
