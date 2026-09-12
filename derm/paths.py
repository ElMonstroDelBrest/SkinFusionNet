"""Project paths resolved relative to the repository root.

Import these constants instead of constructing working-directory-dependent paths:
    from derm import paths
    df = pd.read_csv(paths.FEATURES_CSV)
    mask_root = paths.UNET_MASKS

Run entry points as modules, for example:
    python -m derm.pipeline.cnn_v2s
    python -m derm.fusion.train_lgbm_v2s

Dataset, checkpoint and result paths refer to local files excluded from Git."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../Stage


def relative(path: Path) -> str:
    """Return a repository-relative POSIX path for manifests and reports."""
    return path.resolve().relative_to(ROOT).as_posix()


# Top-level directories
DATA         = ROOT / "data"
MODELS       = ROOT / "models"
ARTIFACTS    = ROOT / "artifacts"
RESULTS      = ROOT / "results"
DOCS         = ROOT / "docs"
EXTERNAL_VAL = ROOT / "external_val"
APP          = ROOT / "app"
APP_MODELS   = APP / "assets" / "models"               # Deployed ONNX assets bundled with the app.
LEGACY_MODELS    = MODELS / "legacy"
LEGACY_ARTIFACTS = ARTIFACTS / "legacy"
LEGACY_RESULTS   = RESULTS / "legacy"

# Result groups
RESULTS_INTERNAL           = RESULTS / "internal"
RESULTS_EXTERNAL           = RESULTS / "external"
RESULTS_ISIC2019           = RESULTS / "isic2019"
RESULTS_ABLATIONS          = RESULTS / "ablations"
RESULTS_ABLATION_ATYPICAL  = RESULTS_ABLATIONS / "atypical"
RESULTS_ABLATION_HANDCRAFT = RESULTS_ABLATIONS / "handcraft"
RESULTS_FEATURES           = RESULTS / "features"

# Document groups
DOCS_PAPER      = DOCS / "paper"
DOCS_PAPER_GENERATED = DOCS_PAPER / "generated"
DOCS_PAPER_FIGURES   = DOCS_PAPER / "figures"
DOCS_MODELS     = DOCS / "models"
DOCS_CLINICAL   = DOCS / "clinical_report"
DOCS_BACKSTORY  = DOCS / "backstory"
DOCS_PROTOCOLS  = DOCS / "protocols"
DOCS_ANALYSES   = DOCS / "analyses"
DOCS_SLIDES     = DOCS / "slides"
DOCS_HANDOFF    = DOCS / "handoff"

# Local datasets (excluded from Git)
MERGED     = DATA / "Skin_Cancer_Merged"               # Merged raw images.
LESION     = DATA / "Skin_Cancer_Merged_lesion"        # Masked lesion images from U-Net.
DEHAIR     = DATA / "Skin_Cancer_Merged_dehair"        # DullRazor-processed images.
UNET_MASKS = DATA / "Skin_Cancer_Merged_unet"          # Binary masks.
ISIC2018   = DATA / "isic2018"                          # U-Net training data (ISIC 2018).
OOD_ISIC2020      = DATA / "ood_isic2020"               # External dermoscopy cohort excluded from the merged dataset.
OOD_ISIC2020_RAW  = OOD_ISIC2020 / "raw"
OOD_ISIC2020_EVAL = OOD_ISIC2020 / "eval_root"          # Melanoma/ + NonMelanoma/

# Local checkpoints (excluded from Git)
CNN_BEST            = LEGACY_MODELS / "cnn_best.pth"           # ResNet18+CBAM (archive 512-d)
CNN_DEHAIR_BEST     = LEGACY_MODELS / "cnn_dehair_best.pth"
CNN_V2S_BEST        = MODELS / "cnn_v2s_best.pth"       # EfficientNetV2-S+CBAM (1280-D deployed feature architecture).
CNN_V2S_DEHAIR_BEST = MODELS / "cnn_v2s_dehair_best.pth"
UNET_BEST           = MODELS / "unet_best.pth"
CNN_V2S_BINARY_LESION = MODELS / "binary" / "cnn_v2s_binary_lesion.pth"   # Experimental binary CNN branch.
CNN_V2S_BINARY_DEHAIR = MODELS / "binary" / "cnn_v2s_binary_dehair.pth"

# Local tabular features and CNN embeddings
FEATURES_CSV           = ARTIFACTS / "features.csv"               # 18 handcraft
CNN_FEATS_CSV          = LEGACY_ARTIFACTS / "cnn_features.csv"           # Historical 512-D lesion features.
CNN_DEHAIR_CSV         = LEGACY_ARTIFACTS / "cnn_dehair_features.csv"    # 512-d dehair
CNN_V2S_FEATS_CSV      = ARTIFACTS / "cnn_v2s_features.csv"       # 1280-D lesion features.
CNN_V2S_DEHAIR_CSV     = ARTIFACTS / "cnn_v2s_dehair_features.csv"
CNN_V2S_TTA_CSV        = ARTIFACTS / "cnn_v2s_tta_features.csv"   # Historical 1280-D lesion features with eight-pass TTA.
CNN_V2S_DEHAIR_TTA_CSV = ARTIFACTS / "cnn_v2s_dehair_tta_features.csv"
INTERNAL_METADATA      = ARTIFACTS / "image_metadata.csv"
INTERNAL_METADATA_JOINED = ARTIFACTS / "image_metadata_joined.csv"
INTERNAL_RAW_MANIFEST  = ARTIFACTS / "internal_dataset_manifest.csv"
ARTIFACTS_BINARY              = ARTIFACTS / "binary"
CNN_V2S_BINARY_LESION_CSV     = ARTIFACTS_BINARY / "cnn_v2s_binary_lesion_features.csv"
CNN_V2S_BINARY_DEHAIR_CSV     = ARTIFACTS_BINARY / "cnn_v2s_binary_dehair_features.csv"

# Internal evaluation
INTERNAL_PREDICTIONS         = RESULTS_INTERNAL / "internal_validity_predictions.csv"
INTERNAL_DECOMPOSITION       = RESULTS_INTERNAL / "internal_validity_decomposition.csv"
INTERNAL_EXPERIMENT_REGISTRY = RESULTS_INTERNAL / "internal_experiment_registry.csv"
INTERNAL_VALIDITY_REPORT     = RESULTS_INTERNAL / "internal_validity_m1_report.md"
INTERNAL_VALIDITY_LOG        = RESULTS_INTERNAL / "internal_validity_m1.log"
INTERNAL_CV_LOG              = RESULTS_INTERNAL / "lgbm_v2s_singlepass1280_gpu_log.txt"

# External evaluation and bundle parity
EXTERNAL_RAW_MANIFEST     = RESULTS_EXTERNAL / "external_raw_file_manifest.csv"
EXTERNAL_PARITY_CSV       = RESULTS_EXTERNAL / "external_deployed_parity_per_image.csv"
EXTERNAL_PARITY_GATE_CSV  = RESULTS_EXTERNAL / "external_deployed_parity_gate.csv"
EXTERNAL_PARITY_GATE_MD   = RESULTS_EXTERNAL / "external_deployed_parity_gate.md"
EXTERNAL_PARITY_GATE_LOG  = RESULTS_EXTERNAL / "external_deployed_parity_gate.log"
EXTERNAL_PARITY_SUMMARY   = RESULTS_EXTERNAL / "external_deployed_parity_summary.txt"
EXTERNAL_PARITY_PREVIEWS  = RESULTS_EXTERNAL / "external_deployed_parity_previews"
EXTERNAL_VAL_LOG          = RESULTS_EXTERNAL / "external_val_singlepass1280.log"
RECALIBRATION_EXTERNAL_CSV = RESULTS_EXTERNAL / "recalibration_external.csv"

# ISIC 2019 held-out evaluation
ISIC2019_PREDICTIONS = RESULTS_ISIC2019 / "isic2019_heldout_predictions.csv"
ISIC2019_METRICS     = RESULTS_ISIC2019 / "isic2019_heldout_metrics.csv"
ISIC2019_AUDIT       = RESULTS_ISIC2019 / "isic2019_heldout_audit.csv"
ISIC2019_WORKBOOK    = DOCS_HANDOFF / "ISIC2019_Heldout_Evaluation.xlsx"

# ISIC 2020 distribution-shift evaluation
RESULTS_OOD_ISIC2020      = RESULTS / "ood_isic2020"
OOD_ISIC2020_GT           = OOD_ISIC2020 / "ISIC_2020_Training_GroundTruth.csv"
OOD_ISIC2020_DUPLICATES   = OOD_ISIC2020 / "ISIC_2020_Training_Duplicates.csv"
OOD_ISIC2020_KEPT         = OOD_ISIC2020 / "kept_manifest.csv"
OOD_ISIC2020_OVERLAP      = RESULTS_OOD_ISIC2020 / "ood_isic2020_overlap_audit.csv"
OOD_ISIC2020_PREDICTIONS  = RESULTS_OOD_ISIC2020 / "ood_isic2020_predictions.csv"
OOD_ISIC2020_METRICS      = RESULTS_OOD_ISIC2020 / "ood_isic2020_metrics.csv"
OOD_ISIC2020_SUMMARY      = RESULTS_OOD_ISIC2020 / "ood_isic2020_summary.txt"
OOD_ISIC2020_LOG          = RESULTS_OOD_ISIC2020 / "ood_isic2020_eval.log"

# Ablation studies
ABLATION_ATYPICAL_CSV           = RESULTS_ABLATION_ATYPICAL / "ablation_atypical.csv"
ABLATION_ATYPICAL_CASES         = RESULTS_ABLATION_ATYPICAL / "ablation_atypical_cases.csv"
ABLATION_ATYPICAL_REPORT        = RESULTS_ABLATION_ATYPICAL / "ablation_atypical_report.md"
ABLATION_ATYPICAL_TRUE_INTERNAL = RESULTS_ABLATION_ATYPICAL / "ablation_atypical_true_atypical_internal.csv"
RESULTS_BINARY_RETRAIN          = RESULTS_ABLATIONS / "binary_retrain"
BINARY_RETRAIN_METRICS          = RESULTS_BINARY_RETRAIN / "binary_retrain_metrics.csv"
BINARY_RETRAIN_REPORT           = RESULTS_BINARY_RETRAIN / "binary_retrain_report.md"
RESULTS_BINARY_CNN              = RESULTS_ABLATIONS / "binary_cnn"
BINARY_CNN_METRICS              = RESULTS_BINARY_CNN / "binary_cnn_metrics.csv"
BINARY_CNN_REPORT               = RESULTS_BINARY_CNN / "binary_cnn_report.md"
RESULTS_SUITE                   = RESULTS / "suite"

ABLATION_HANDCRAFT_FACTORIAL = RESULTS_ABLATION_HANDCRAFT / "ablation_handcraft_factorial.csv"
ABLATION_HANDCRAFT_EFFECTS   = RESULTS_ABLATION_HANDCRAFT / "ablation_handcraft_effects.csv"
ABLATION_HANDCRAFT_CONFOUNDB = RESULTS_ABLATION_HANDCRAFT / "ablation_handcraft_confoundB.csv"
ABLATION_HANDCRAFT_LOG       = RESULTS_ABLATION_HANDCRAFT / "ablation_handcraft_factorial.log"
ABLATION_HANDCRAFT_CACHE     = RESULTS_ABLATION_HANDCRAFT / "cache"
ABLATION_HANDCRAFT_DOC       = DOCS_ANALYSES / "ablation_handcraft_factorial.md"

# Feature analysis and probing
IMAGE_FEATS_CSV            = RESULTS_FEATURES / "image_feats.csv"
FEATURE_IMPORTANCE_CSV     = RESULTS_FEATURES / "feature_importance.csv"
FEATURE_IMPORTANCE_XLSX    = RESULTS_FEATURES / "feature_importance.xlsx"
CURRENT_ONNX_IMPORTANCE    = RESULTS_FEATURES / "current_onnx_feature_split_importance.csv"
PROBING_HANDCRAFT_CSV      = RESULTS_FEATURES / "probing_handcraft.csv"
PROBING_HANDCRAFT_LOG      = RESULTS_FEATURES / "probing_handcraft.log"
GROUPCV_HANDCRAFT_LOG      = RESULTS_FEATURES / "groupcv_handcraft.log"

# Historical results
TTA8_EXTERNAL_CSV       = LEGACY_RESULTS / "external_val_per_image_tta8.csv"
PROBING_IMAGE_FEATS_CSV = LEGACY_RESULTS / "probing_image_feats.csv"

# Local handoff and paper artifacts
HANDOFF_MANIFEST = RESULTS / "handoff_manifest.json"
PAPER            = DOCS_PAPER
PAPER_WORKBOOK   = DOCS_PAPER / "SkinFusionNet_Paper_Data.xlsx"
PAPER_NEAR_DUPLICATE_AUDIT = DOCS_PAPER_GENERATED / "external_near_duplicate_audit.csv"
PAPER_VERIFICATION_AUDIT   = DOCS_PAPER_GENERATED / "verification_audit.csv"
PAPER_INTERNAL_REPRODUCTION = DOCS_PAPER_GENERATED / "internal_reproduction_check.csv"
PAPER_EXTERNAL_PERFORMANCE  = DOCS_PAPER_GENERATED / "external_performance.csv"
PAPER_EXTERNAL_PAIRED       = DOCS_PAPER_GENERATED / "external_paired_comparison.csv"
PAPER_PERFORMANCE_REGISTRY  = DOCS_PAPER_GENERATED / "performance_registry.csv"
PAPER_DEDUPLICATED_SENS     = DOCS_PAPER_GENERATED / "external_deduplicated_sensitivity.csv"
PAPER_POSTHOC_READOUT       = DOCS_PAPER_GENERATED / "external_posthoc_two_class_readout.csv"
