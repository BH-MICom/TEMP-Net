# Evaluation metrics

Temperature metrics are computed inside the tumor mask. The implementation supports MAE, RMSE, signed bias, median absolute error, Pearson correlation, and SSIM.

CEM43 agreement is evaluated at inclusive thresholds of CEM43 >= 2, 10, and 60 minutes using Dice, HD95, ASSD, precision, recall, and tumor coverage. HD95 and ASSD use the physical in-plane spacing.

Patient-level aggregation is performed by averaging segment-plane metrics within each patient and then averaging patients equally. Statistical testing should use patients as the paired unit. The recommended test is a two-sided paired Wilcoxon signed-rank test with Holm correction when comparing multiple models.
