# Center provenance

Probe/fiber centers are treated as reviewed inputs throughout the paper path. The training preprocessor requires `--center-manifest` and stores the supplied center as `probe_center_rc=[row, col]` in each NPZ. The Gaussian map and normalized coordinate conditioning are generated from that center.

The 3D paper runner reads the same NPZ center for each monitored plane and uses it to establish the DICOM physical point, the virtual-plane coordinate system, and the reconstructed fiber-centered field. No temperature-hotspot `argmax` calibration is performed.
