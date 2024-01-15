import numpy as np

RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)

PANNUKE_FOLDS = [[1, 2], [0, 2], [1, 0]]
PANNUKE_TISSUES = [
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
]
CLASS_NAMES = [
    "neutrophil",
    "epithelial-cell",
    "lymphocyte",
    "plasma-cell",
    "eosinophil",
    "connective-tissue-cell",
    "mitosis",
]
CLASS_NAMES_PANNUKE = [
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
]

BEST_MIN_THRESHS = [30, 30, 20, 20, 30, 30, 15]  # stick to best conic
BEST_MAX_THRESHS = [5000, 5000, 5000, 5000, 5000, 5000, 5000]  # stick to best conic

MIN_THRESHS_PANNUKE = [10, 10, 10, 10, 10]
MAX_THRESHS_PANNUKE = [20000, 20000, 20000, 3000, 10000]
