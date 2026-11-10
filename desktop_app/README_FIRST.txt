PROJECT PROTEUS / PROTEUSSEGNET — WINDOWS LOCAL APPLICATION
Team: Mariolino Daniel III Julabar, AlReem AlRowahi, Sultan AlShehhi

QUICK START
1. Extract the ZIP completely. Do not run the BAT files from inside the ZIP.
2. Open the ProteusSegNet folder.
3. Copy your trained proteus_model.pt into models. Weights are NOT included.
4. Copy ESA WorldCover GeoTIFFs into worldcover_cache, retaining original filenames.
5. Double-click Setup_Windows.bat once. It creates .venv and installs dependencies.
6. Double-click Start_Proteus.bat. Keep its terminal window open.
7. Your browser opens the app. If not, open http://127.0.0.1:7860.
8. Upload your SAR image, review the Input check panel, and execute analysis.
   (Sign-in is requested only if login.json or PROTEUS_USERNAME/PROTEUS_PASSWORD is set.)

WHAT IS NEW (v2)
- Tiled processing: SAR images above 16 million pixels are processed in tiles
  (default 2048 px, 32 px halo) with seam-aware region merging, up to 1.5 billion px.
  Results are pixel-identical to full-scene processing; peak RAM depends on tile size.
- Refined interface: input pre-flight check (size, CRS, dB sanity, WorldCover coverage,
  processing plan), live effective thresholds, progress bar, KPI cards, zoomable figures,
  largest-object table, individual file downloads, User Guide tab.

PREREQUISITES
64-bit Windows and 64-bit Python 3.11 or 3.12, including the Python launcher.
If Python is missing, setup stops with instructions. It does not install Python itself.
Initial installation needs internet and may download large packages, especially PyTorch.
This is a Python application with a double-click launcher, NOT a self-contained EXE.
CPU is the default. No CUDA GPU is required for CPU inference.

MODEL AND TRAINING
Place the existing checkpoint at models/proteus_model.pt. No retraining is needed to rename a compatible checkpoint.
Optional training saves to this same name and overwrites the default checkpoint. Back it up first.
Model download appears only in Model Training; Analysis Workspace downloads only the analysis ZIP.
Startup never trains. Training requires explicit confirmation and a dataset prepared in the same app session.
Only load checkpoints you trust.

WORLDCOVER
The local cache is reused across restarts. Reference upload is optional in Analysis Workspace.
Uploaded analysis references are cached for reuse. Missing tiles are NOT automatically downloaded.
Unknown coverage stays excluded. WorldCover water classes are not specifically marine classes.

OUTPUTS
Analysis results and ZIPs are retained in outputs. Download appears inside the app below all figures.
This local application uses native Gradio file downloads; it does not rely on Colab file handling.
Figures are embedded in the app; downloading the ZIP is not necessary to view them.
Remove old output folders manually when no longer needed.

LOGIN AND PRIVACY
The requested credentials are in login.json. Keep it private and do not publish the package unchanged.
Edit login.json and restart to change credentials. PROTEUS_USERNAME and PROTEUS_PASSWORD environment variables override its values.
The developer credit appears on sign-in and inside the app.
The server binds to 127.0.0.1 only; it does not create a public sharing link.
This is not a hardened multiuser deployment. Do not expose it publicly without additional security work.

LIMITATIONS
Scenes up to 16 million pixels run in memory; larger scenes (up to 1.5 billion pixels) are tiled
automatically. Tiling writes temporary files (about 5 bytes per pixel) to the system temp folder
(override with PROTEUS_SCRATCH_DIR) and
removes them afterwards, so keep enough free disk space. Use a smaller tile size on low-RAM PCs.
Figure display percentiles for tiled scenes come from the downsampled preview.
Labels are physics pseudo-labels, not independently verified oil annotations.
Independent validation and comparison with a physics-only baseline are outstanding.
Training-image predictions demonstrate execution, not accuracy.
Scores and heuristic confidence classes are not calibrated oil probabilities.
Area estimates are CRS-dependent approximations, not confirmed spill areas. Connected objects are not necessarily separate spills.
No cloud AI APIs, pretrained weights or Colab services are used in local processing.

STOPPING
Press Ctrl+C in the terminal. Start_Proteus.bat starts it again.
Do not start a second copy while the first is running: port 7860 may be occupied.

VERIFICATION STATUS
v2 was executed end to end on Windows 11 (Python 3.13, torch 2.15 CPU, gradio 5.50) with
models/proteus_model.pt and SAR_20261021_Subset-2.tif: full-scene results reproduce the original
report exactly (120.1076 km2, 558 objects), tiled results are pixel-identical, and a 45.7 M-pixel
scene completed in tiled mode. Dependency versions are bounded ranges, not a lockfile.
Keep your original notebook and model backup until you verify this application.
