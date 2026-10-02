# nnU-Net Dashboard — 3D Slicer plugin

Scripted module for [3D Slicer](https://www.slicer.org/) that connects to an
nnU-Net labeling server: download cases, edit labels in Segment Editor, save
them back, and run approved auto-segmentation models.

> This plugin is **not** on the Slicer Extensions Manager yet. Install it from
> this GitHub repository using **Additional module paths** (steps below).

## Features

1. **Login** to an nnU-Net server (`POST /auth/login`)
2. **Select** a dataset and train/test case
3. **Fetch** image (+ label if present) into Slicer
4. **Edit** labels with Segment Editor
5. **Save** packed multi-class `.mha` labels back to the server
6. **Predict** with the **Prediction Tool** (approved models → new segmentation)

## Requirements

- 3D Slicer 5.x (tested with recent Stable / Preview builds)
- Network access to your nnU-Net API (`…/api/v3`) and Keycloak login
- Python package **`requests`** inside Slicer’s Python environment

## Install from this Git repository

### 1. Clone the repo

```bash
git clone https://github.com/jinkoo2/nnunet-slicer-plugin.git
```

Or with SSH (if your key is set up):

```bash
git clone git@github.com:jinkoo2/nnunet-slicer-plugin.git
```

Note the absolute path to the cloned folder, for example:

```text
C:\Users\<you>\nnunet-slicer-plugin
```

### 2. Add the module path in Slicer

1. Start **3D Slicer**.
2. Open **Edit → Application Settings → Modules**.
3. Under **Additional module paths**, click **Add** and select the
   **`NnUNetDashboard`** folder inside the clone — the directory that contains
   `NnUNetDashboard.py`, **not** the repo root:

   ```text
   <clone>/NnUNetDashboard
   ```

   Example:

   ```text
   C:\Users\<you>\nnunet-slicer-plugin\NnUNetDashboard
   ```

   Slicer only discovers `*.py` modules in that folder. Pointing at the repo
   root alone will not load the module.

4. Click **OK**. Restart Slicer when prompted (or restart manually).

### 3. Install the `requests` dependency

In Slicer’s **Python Interactor** (View → Python Interactor):

```python
import slicer.util
slicer.util.pip_install("requests")
```

Restart Slicer once after installing.

### 4. Open the module

In the module search box, find **nnUNet Dashboard** (category **Segmentation**),
or browse **Modules → Segmentation → nnUNet Dashboard**.

### Updating

```bash
cd nnunet-slicer-plugin
git pull
```

Then in Slicer use **Developer Tools → Reload & Test** (or restart Slicer) so
Python picks up the changes.

## Usage

### Connection

1. Choose a server URL ending in `…/api/v3` (or type one).
2. Enter email / password → **Connect / Login**.
3. Optional: **Register** opens the Keycloak account page in your browser;
   create an account, then return and log in.
4. **Settings…** edits the server URL list, Keycloak URL/realm, and optional
   registration URL. Values are stored in Slicer’s user settings
   (`NnUNetDashboard/…` in QSettings), not in a project `.env` file.

### Dataset & case

1. Pick a **Dataset**, **train** or **test**, then a **Case** in the table.
2. Case **Status** can be changed from the dropdown (saved to the server).
3. **Fetch Image + Labels** downloads the volume and label (if any) and creates
   a segmentation node. Segment Editor opens automatically.

### Edit & save

1. Paint in **Segment Editor**. Segment names should match `dataset.json`
   label names.
2. **Save Labels to Server** packs segments into a multi-class `.mha` using
   dataset.json class IDs and uploads image + label.

### Prediction Tool

1. Expand **Prediction Tool** (requires login; a volume should be loaded —
   typically after **Fetch**).
2. Optionally filter models, then select an **approved** model.
3. Choose **Mode**:
   - **Fast – Single Model** (`fold_all`) when the model reports
     `fold_all_available` (default when both modes exist)
   - **Accurate – 5-Fold Ensemble** (CV ensemble of folds 0–4)
4. Choose **Labels to Import** and a prediction **Server** (or
   **Next Available Server** for load balancing).
5. **Run Auto Segment**. Job status appears in tabs; when finished, a **new**
   segmentation node is created (existing case labels are left alone) and
   Segment Editor opens on the result.
6. Multi-channel models prompt to download the full case image set from the
   logged-in server (the viewer holds only channel 0).

## Repository layout

```text
nnunet-slicer-plugin/
  README.md
  NnUNetDashboard/
    NnUNetDashboard.py              # Slicer module (UI + logic)
    NnUNetDashboardLib/
      __init__.py
      NnUNetClient.py               # nnU-Net v3 HTTP client
      ModuleSettings.py             # QSettings persistence
      PredictionTool.py             # Prediction Tool panel
    CMakeLists.txt
    Resources/
      Icons/NnUNetDashboard.png
      requirements.txt
```

## API notes

- Auth: Bearer JWT from `/auth/login` (Keycloak via the nnU-Net server).
- Download: `GET /datasets/download_image`, `GET /datasets/download_label`.
- Save: `PUT /datasets/update_image|update_label` (404 → `add_image|add_label`).
- Prediction: approved models, `/predictions/predict`, status poll, result ZIP.
- Image format: MetaImage **`.mha`**.

## Optional: build inside a Slicer source tree

If you maintain a Slicer source tree, you can copy or symlink `NnUNetDashboard/`
under `Modules/Scripted/`, list it in that folder’s `CMakeLists.txt`, and
rebuild. For normal use, **Additional module paths** + this Git clone is enough.

## Related

Companion desktop labeler: [vtk_image_labeler_3d](https://github.com/jinkoo2/vtk_image_labeler_3d).
