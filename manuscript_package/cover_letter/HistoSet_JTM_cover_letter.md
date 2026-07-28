# Cover Letter

Dear Editors,

We are pleased to submit our manuscript entitled "Hierarchical conformal concept-set segmentation for uncertainty-aware Gleason grading in prostate cancer histopathology" for consideration as a Research Article in Journal of Translational Medicine.

Gleason grading remains central to prostate cancer diagnosis and risk stratification, but pixel-level histological interpretation can be ambiguous, especially in regions containing mixed patterns, rare morphologies, or low reader agreement. Most computational pathology segmentation systems return a single label per pixel, which is useful for benchmarking but can hide the uncertainty that pathologists encounter during review.

In this study, we present HistoSet, a hierarchical conformal concept-set segmentation framework for uncertainty-aware prostate cancer histopathology. HistoSet combines three methodological components in a single pixel-level pipeline: a pathology-facing hierarchy linking histological concepts to Gleason patterns, soft-label learning from available pathologist vote distributions, and conformal prediction that converts model probabilities into calibrated sets of plausible Gleason patterns or morphology concepts.

Using 1,527 supervised prepared samples from public prostate pathology resources, HistoSet generated both deterministic segmentation outputs and calibrated set-valued outputs. At alpha = 0.10, direct Gleason-pattern conformal sets achieved empirical coverage of 0.896 across 14,789,958 test pixels, while concept-level sets achieved empirical coverage of 0.920 across 2,016,949 concept-supervised test pixels. The manuscript includes source-backed main figures, supplementary analyses of model configuration, replicate stability, conformal alpha sensitivity, rare-concept behavior, and data-cleanliness checks.

We believe this work fits Journal of Translational Medicine because it addresses a translational pathology problem at the interface of medical bioinformatics, molecular pathology-adjacent histopathology, translational imaging, and data-driven clinical decision processes. The contribution is not a claim of replacing pathologists or outperforming all deterministic grading systems. Rather, HistoSet provides a transparent framework for making diagnostic ambiguity visible, auditable, and quantitatively calibrated in the same pixel-level space where histological morphology is reviewed.

All authors have approved the submitted version of the manuscript. The manuscript has not been published previously, is not under consideration elsewhere, and does not contain identifiable patient information. The study uses public, de-identified data resources. Data availability, code availability, competing interests, funding, and author contribution statements are provided in the manuscript declarations.

Thank you for considering our manuscript. We look forward to your response.

Sincerely,

The authors
