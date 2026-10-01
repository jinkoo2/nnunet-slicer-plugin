"""3D Slicer scripted module: nnU-Net Dashboard.

Login to an nnU-Net server, select a dataset/case, load image+labels into
Slicer for editing (Segment Editor), and save labels back to the server.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid

import qt
import ctk
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

# Allow importing sibling package when loaded as a scripted module.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

# Slicer Developer Reload re-executes this file but keeps Lib packages in
# sys.modules. Force-reload so prediction APIs / UI changes take effect.
import importlib

import NnUNetDashboardLib
import NnUNetDashboardLib.NnUNetClient
import NnUNetDashboardLib.ModuleSettings
import NnUNetDashboardLib.PredictionTool

importlib.reload(NnUNetDashboardLib.NnUNetClient)
importlib.reload(NnUNetDashboardLib.ModuleSettings)
importlib.reload(NnUNetDashboardLib.PredictionTool)
importlib.reload(NnUNetDashboardLib)

from NnUNetDashboardLib import NnUNetClient as nnunet_client
from NnUNetDashboardLib import ModuleSettings as module_settings
from NnUNetDashboardLib.PredictionTool import PredictionToolPanel


#
# NnUNetDashboard
#


class NnUNetDashboard(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "nnUNet Dashboard"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jinkoo Kim"]
        self.parent.helpText = (
            "Connect to an nnU-Net labeling server, download a case, edit "
            "labels with Segment Editor, and upload the result.\n\n"
            "Use Prediction Tool to run approved nnU-Net models on the loaded "
            "volume and import result segments.\n\n"
            "Use Settings to configure server URLs and Keycloak registration."
        )
        self.parent.acknowledgementText = (
            "Companion to vtk_image_labeler_3d. Uses the nnU-Net server v3 API."
        )


#
# NnUNetDashboardSettingsDialog
#


class NnUNetDashboardSettingsDialog(qt.QDialog):
    """Edit persisted module settings (servers + Keycloak)."""

    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("nnUNet Dashboard Settings")
        self.setMinimumWidth(520)
        self.resize(560, 420)

        layout = qt.QVBoxLayout(self)
        form = qt.QFormLayout()

        self.serverUrlsEdit = qt.QPlainTextEdit()
        self.serverUrlsEdit.setPlaceholderText(
            "One nnU-Net server URL per line\n"
            "https://nnunet-server-01.apps.myphysics.net/api/v3"
        )
        self.serverUrlsEdit.setMinimumHeight(110)
        form.addRow("Server URLs:", self.serverUrlsEdit)

        self.keycloakUrlEdit = qt.QLineEdit()
        form.addRow("Keycloak URL:", self.keycloakUrlEdit)

        self.keycloakRealmEdit = qt.QLineEdit()
        form.addRow("Keycloak realm:", self.keycloakRealmEdit)

        self.registrationUrlEdit = qt.QLineEdit()
        self.registrationUrlEdit.setPlaceholderText(
            "Optional — leave blank to use Keycloak Account Console"
        )
        form.addRow("Registration URL:", self.registrationUrlEdit)

        layout.addLayout(form)

        note = qt.QLabel(
            "Settings are saved in Slicer's user settings (QSettings). "
            "Leave Registration URL blank to derive it from Keycloak URL + realm "
            "(…/realms/<realm>/account/)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)

        # Explicit buttons — StandardButton enums are unreliable in Slicer PythonQt.
        btnRow = qt.QHBoxLayout()
        self.resetButton = qt.QPushButton("Reset to defaults")
        self.resetButton.clicked.connect(self._resetDefaults)
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.reject)
        self.saveButton = qt.QPushButton("Save")
        self.saveButton.setDefault(True)
        self.saveButton.clicked.connect(self._onSaveClicked)
        btnRow.addWidget(self.resetButton)
        btnRow.addStretch(1)
        btnRow.addWidget(self.cancelButton)
        btnRow.addWidget(self.saveButton)
        layout.addLayout(btnRow)

        self._saved_cfg = None
        self._load()

    def _load(self):
        cfg = module_settings.load_settings()
        self.serverUrlsEdit.setPlainText("\n".join(cfg.get("server_urls") or []))
        self.keycloakUrlEdit.setText(cfg.get("keycloak_url") or "")
        self.keycloakRealmEdit.setText(cfg.get("keycloak_realm") or "")
        # Show blank when using derived default so Save keeps "auto" behavior.
        stored_reg = module_settings._get(module_settings.KEY_REGISTRATION_URL, "")
        self.registrationUrlEdit.setText(str(stored_reg or ""))

    def _widgetText(self, widget, plain=False):
        if plain:
            # Prefer method form used by Qt; fall back to property.
            if hasattr(widget, "toPlainText"):
                text = widget.toPlainText
                text = text() if callable(text) else text
            else:
                text = ""
        else:
            text = widget.text
            text = text() if callable(text) else text
        return str(text or "")

    def _resetDefaults(self):
        self.serverUrlsEdit.setPlainText("\n".join(nnunet_client.DEFAULT_SERVER_URLS))
        self.keycloakUrlEdit.setText(nnunet_client.DEFAULT_KEYCLOAK_URL)
        self.keycloakRealmEdit.setText(nnunet_client.DEFAULT_KEYCLOAK_REALM)
        self.registrationUrlEdit.setText("")

    def _onSaveClicked(self):
        self._saved_cfg = self.save()
        self.accept()

    def save(self):
        return module_settings.save_settings(
            server_urls=self._widgetText(self.serverUrlsEdit, plain=True),
            keycloak_url=self._widgetText(self.keycloakUrlEdit),
            keycloak_realm=self._widgetText(self.keycloakRealmEdit),
            registration_url=self._widgetText(self.registrationUrlEdit),
        )

    def saved_settings(self):
        return self._saved_cfg


#
# NnUNetDashboardWidget
#


class NnUNetDashboardWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._datasets = []
        self._selected_dataset = None
        self._case_nums = []
        self._case_status_by_num = {}
        self._loaded_case = None  # dict with dataset_id, images_for, num, ...
        self._volume_node = None
        self._segmentation_node = None
        self._temp_dir = None
        self.predictionPanel = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = NnUNetDashboardLogic()

        # --- Connection ---
        connCollapsible = ctk.ctkCollapsibleButton()
        connCollapsible.text = "Connection"
        connCollapsible.collapsed = False
        self.layout.addWidget(connCollapsible)
        connLayout = qt.QFormLayout(connCollapsible)

        serverRow = qt.QHBoxLayout()
        self.serverCombo = qt.QComboBox()
        self.serverCombo.setEditable(True)
        self.serverCombo.setToolTip("nnU-Net API root (…/api/v3)")
        self.serverCombo.activated.connect(self._onServerActivated)
        self.settingsButton = qt.QPushButton("Settings…")
        self.settingsButton.setToolTip(
            "Edit server URL list and Keycloak / registration settings"
        )
        self.settingsButton.clicked.connect(self.onSettings)
        serverRow.addWidget(self.serverCombo, 1)
        serverRow.addWidget(self.settingsButton)
        connLayout.addRow("Server:", serverRow)

        self.emailEdit = qt.QLineEdit()
        self.emailEdit.setPlaceholderText("you@example.com")
        connLayout.addRow("Email:", self.emailEdit)

        self.passwordEdit = qt.QLineEdit()
        self.passwordEdit.setEchoMode(qt.QLineEdit.Password)
        self.passwordEdit.setPlaceholderText("Password")
        connLayout.addRow("Password:", self.passwordEdit)

        btnRow = qt.QHBoxLayout()
        self.connectButton = qt.QPushButton("Connect / Login")
        self.connectButton.clicked.connect(self.onConnect)
        self.registerButton = qt.QPushButton("Register")
        self.registerButton.setToolTip(
            "Open the myphysics sign-in page in your browser, then choose Register"
        )
        self.registerButton.clicked.connect(self.onRegister)
        self.disconnectButton = qt.QPushButton("Disconnect")
        self.disconnectButton.clicked.connect(self.onDisconnect)
        self.disconnectButton.enabled = False
        btnRow.addWidget(self.connectButton)
        btnRow.addWidget(self.registerButton)
        btnRow.addWidget(self.disconnectButton)
        connLayout.addRow(btnRow)

        self.authStatusLabel = qt.QLabel("Not signed in")
        self.authStatusLabel.setStyleSheet("color: #666;")
        self.authStatusLabel.setWordWrap(True)
        connLayout.addRow(self.authStatusLabel)

        self._applySettingsToUi(module_settings.load_settings())

        # --- Dataset / case ---
        dataCollapsible = ctk.ctkCollapsibleButton()
        dataCollapsible.text = "Dataset & Case"
        dataCollapsible.collapsed = False
        self.layout.addWidget(dataCollapsible)
        dataLayout = qt.QFormLayout(dataCollapsible)

        self.datasetCombo = qt.QComboBox()
        self.datasetCombo.setEnabled(False)
        self.datasetCombo.currentIndexChanged.connect(self.onDatasetChanged)
        dataLayout.addRow("Dataset:", self.datasetCombo)

        self.splitCombo = qt.QComboBox()
        self.splitCombo.addItem("train", "train")
        self.splitCombo.addItem("test", "test")
        self.splitCombo.setEnabled(False)
        self.splitCombo.currentIndexChanged.connect(self.onSplitChanged)
        dataLayout.addRow("Split:", self.splitCombo)

        self.caseTable = qt.QTableWidget()
        self.caseTable.setColumnCount(2)
        self.caseTable.setHorizontalHeaderLabels(["Case", "Status"])
        self.caseTable.horizontalHeader().setStretchLastSection(True)
        self.caseTable.verticalHeader().setVisible(False)
        self.caseTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.caseTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.caseTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.caseTable.setMinimumHeight(160)
        self.caseTable.setEnabled(False)
        self.caseTable.itemSelectionChanged.connect(self._onCaseSelectionChanged)
        dataLayout.addRow("Cases:", self.caseTable)

        caseBtnRow = qt.QHBoxLayout()
        self.refreshCasesButton = qt.QPushButton("Refresh Cases")
        self.refreshCasesButton.setEnabled(False)
        self.refreshCasesButton.clicked.connect(self.onRefreshCases)
        self.fetchButton = qt.QPushButton("Fetch Image + Labels")
        self.fetchButton.setEnabled(False)
        self.fetchButton.clicked.connect(self.onFetchCase)
        caseBtnRow.addWidget(self.refreshCasesButton)
        caseBtnRow.addWidget(self.fetchButton)
        dataLayout.addRow(caseBtnRow)

        self.caseStatusLabel = qt.QLabel("No case loaded")
        self.caseStatusLabel.setWordWrap(True)
        dataLayout.addRow(self.caseStatusLabel)

        # --- Edit / save ---
        editCollapsible = ctk.ctkCollapsibleButton()
        editCollapsible.text = "Edit & Save"
        editCollapsible.collapsed = False
        self.layout.addWidget(editCollapsible)
        editLayout = qt.QVBoxLayout(editCollapsible)

        note = qt.QLabel(
            "After Fetch, use Segment Editor to paint labels. "
            "Segment names must match dataset.json label names. "
            "Save uploads the packed multi-class label (.mha) to the server."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        editLayout.addWidget(note)

        editBtnRow = qt.QHBoxLayout()
        self.openSegmentEditorButton = qt.QPushButton("Open Segment Editor")
        self.openSegmentEditorButton.setEnabled(False)
        self.openSegmentEditorButton.clicked.connect(self.onOpenSegmentEditor)
        self.saveButton = qt.QPushButton("Save Labels to Server")
        self.saveButton.setEnabled(False)
        self.saveButton.clicked.connect(self.onSaveCase)
        editBtnRow.addWidget(self.openSegmentEditorButton)
        editBtnRow.addWidget(self.saveButton)
        editLayout.addLayout(editBtnRow)

        self.statusText = qt.QTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumHeight(160)
        self.statusText.setPlaceholderText("Status messages appear here.")
        editLayout.addWidget(self.statusText)

        # --- Prediction Tool ---
        self.predictionPanel = PredictionToolPanel(self)
        self.predictionPanel.build(self.layout)

        self.layout.addStretch(1)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)

    def enter(self):
        # Re-check volume / model enable state when the module panel is shown.
        if self.predictionPanel:
            self.predictionPanel.refresh_context()

    def cleanup(self):
        self.removeObservers()

    def onSceneStartClose(self, caller, event):
        self._volume_node = None
        self._segmentation_node = None
        self._loaded_case = None
        self._updateCaseUi()
        if self.predictionPanel:
            self.predictionPanel.refresh_context()

    # ------------------------------------------------------------------ UI helpers

    def _log(self, msg):
        logging.info(msg)
        self.statusText.append(str(msg))
        sb = self.statusText.verticalScrollBar()
        sb.setValue(sb.maximum)
        slicer.app.processEvents()

    def _setConnected(self, connected, email=None):
        self.connectButton.enabled = not connected
        self.disconnectButton.enabled = connected
        self.datasetCombo.setEnabled(connected)
        self.splitCombo.setEnabled(connected)
        self.caseTable.setEnabled(connected)
        self.refreshCasesButton.setEnabled(connected)
        self.fetchButton.setEnabled(connected)
        if connected:
            self.authStatusLabel.setText(f"Signed in as {email or '?'}")
            self.authStatusLabel.setStyleSheet("color: #0a0;")
        else:
            self.authStatusLabel.setText("Not signed in")
            self.authStatusLabel.setStyleSheet("color: #666;")

    def _updateCaseUi(self):
        loaded = self._loaded_case is not None and self._segmentation_node is not None
        self.openSegmentEditorButton.setEnabled(loaded)
        self.saveButton.setEnabled(loaded)
        if self._loaded_case:
            c = self._loaded_case
            self.caseStatusLabel.setText(
                f"Loaded: {c['dataset_id']} / {c['images_for']} / case {c['num']}"
            )
        else:
            self.caseStatusLabel.setText("No case loaded")
        if self.predictionPanel:
            self.predictionPanel.refresh_context()

    def _currentServerUrl(self):
        text = self.serverCombo.currentText
        text = text() if callable(text) else text
        return (text or "").strip().rstrip("/")

    def _applySettingsToUi(self, cfg):
        urls = cfg.get("server_urls") or list(nnunet_client.DEFAULT_SERVER_URLS)
        selected = (cfg.get("selected_server_url") or "").strip().rstrip("/")
        self.serverCombo.blockSignals(True)
        self.serverCombo.clear()
        for url in urls:
            self.serverCombo.addItem(url)
        if selected:
            idx = self.serverCombo.findText(selected)
            if idx < 0:
                self.serverCombo.insertItem(0, selected)
                idx = 0
            self.serverCombo.setCurrentIndex(idx)
        elif self.serverCombo.count > 0:
            self.serverCombo.setCurrentIndex(0)
        self.serverCombo.blockSignals(False)

        email = (cfg.get("last_email") or "").strip()
        existing = self.emailEdit.text
        existing = existing() if callable(existing) else existing
        if email and not str(existing or "").strip():
            self.emailEdit.setText(email)

    def _onServerActivated(self, index):
        url = self._currentServerUrl()
        if url:
            module_settings.save_settings(selected_server_url=url)

    def onSettings(self):
        dlg = NnUNetDashboardSettingsDialog(self.parent)
        result = dlg.exec_()
        accepted = bool(result) or result == qt.QDialog.Accepted
        if not accepted:
            return
        cfg = dlg.saved_settings() or module_settings.load_settings()
        self._applySettingsToUi(cfg)
        if self.predictionPanel:
            self.predictionPanel.on_servers_changed()
        self._log(
            "Settings saved. Servers: "
            + (", ".join(cfg.get("server_urls") or []) or "(none)")
        )

    def _selectedDataset(self):
        data = self.datasetCombo.itemData(self.datasetCombo.currentIndex)
        return data if isinstance(data, dict) else None

    def _selectedImagesFor(self):
        return self.splitCombo.itemData(self.splitCombo.currentIndex) or "train"

    def _selectedCaseNum(self):
        rows = self.caseTable.selectionModel().selectedRows() if self.caseTable.selectionModel() else []
        if not rows:
            current = self.caseTable.currentRow()
            if current < 0:
                return None
            row = current
        else:
            row = rows[0].row()
        item = self.caseTable.item(row, 0)
        if item is None:
            return None
        try:
            return int(item.data(qt.Qt.UserRole))
        except (TypeError, ValueError):
            try:
                return int(item.text())
            except (TypeError, ValueError):
                return None

    def _onCaseSelectionChanged(self):
        # Selection alone is enough for Fetch; status edits are per-row combos.
        pass

    def _clearCaseTable(self):
        self.caseTable.blockSignals(True)
        self.caseTable.setRowCount(0)
        self.caseTable.blockSignals(False)
        self._case_nums = []
        self._case_status_by_num = {}

    def _populateCaseTable(self, case_nums, status_by_num):
        self._case_nums = list(case_nums or [])
        self._case_status_by_num = dict(status_by_num or {})
        self.caseTable.blockSignals(True)
        self.caseTable.setRowCount(0)
        for num in self._case_nums:
            row = self.caseTable.rowCount
            self.caseTable.insertRow(row)
            item = qt.QTableWidgetItem(str(num))
            item.setData(qt.Qt.UserRole, int(num))
            item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
            self.caseTable.setItem(row, 0, item)
            status = self._case_status_by_num.get(int(num), "")
            self._setRowStatusCombo(row, int(num), status)
        self.caseTable.blockSignals(False)
        if self.caseTable.rowCount > 0:
            self.caseTable.selectRow(0)

    def _comboText(self, combo):
        text = combo.currentText
        return text() if callable(text) else text

    def _setRowStatusCombo(self, row, num, status):
        combo = qt.QComboBox()
        combo.setEditable(True)
        for opt in nnunet_client.LABEL_STATUS_OPTIONS:
            combo.addItem(opt)
        status = str(status or "")
        if status and combo.findText(status) < 0:
            combo.addItem(status)
        combo.setCurrentText(status)
        combo.setProperty("case_num", int(num))
        combo.setProperty("last_saved_status", status)
        combo.setToolTip("Change label status for this case")
        combo.activated.connect(
            lambda _idx, n=int(num), c=combo: self._onRowStatusChanged(
                n, self._comboText(c), c
            )
        )
        line = combo.lineEdit()
        if line is not None:
            line.editingFinished.connect(
                lambda n=int(num), c=combo: self._onRowStatusChanged(
                    n, self._comboText(c), c
                )
            )
        self.caseTable.setCellWidget(row, 1, combo)

    def _onRowStatusChanged(self, num, status_text, combo):
        url = self._currentServerUrl()
        ds = self._selected_dataset
        if not url or not ds:
            return
        new_status = (status_text or "").strip()
        last_saved = str(combo.property("last_saved_status") or "")
        if new_status == last_saved:
            return
        ds_id = ds.get("id")
        images_for = self._selectedImagesFor()
        try:
            with slicer.util.tryWithErrorDisplay(
                f"Failed to update status for case {num}", waitCursor=True
            ):
                self.logic.setLabelStatus(url, ds_id, images_for, num, new_status)
                combo.setProperty("last_saved_status", new_status)
                self._case_status_by_num[int(num)] = new_status
                self._log(f"Case {num} status → '{new_status or '(empty)'}'")
        except Exception:
            # Revert combo to last known good value.
            try:
                combo.blockSignals(True)
                combo.setCurrentText(last_saved)
                combo.blockSignals(False)
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------ actions

    def onRegister(self):
        url = module_settings.registration_url_for_use()
        if not url:
            slicer.util.infoDisplay(
                "No registration URL is configured.\n"
                "Open Settings… to set Keycloak URL / realm, or ask an administrator."
            )
            return
        slicer.util.infoDisplay(
            "Your browser will open the myphysics sign-in page.\n\n"
            "Click Register on that page, verify your email, then return "
            "here and log in with your new email and password."
        )
        ok = qt.QDesktopServices.openUrl(qt.QUrl(url))
        if not ok:
            import webbrowser

            webbrowser.open(url)
        self._log(f"Opened registration page: {url}")

    def onConnect(self):
        url = self._currentServerUrl()
        email_attr = self.emailEdit.text
        email = (email_attr() if callable(email_attr) else email_attr or "").strip()
        password_attr = self.passwordEdit.text
        password = password_attr() if callable(password_attr) else (password_attr or "")
        if not url:
            slicer.util.errorDisplay("Enter a server URL.")
            return
        if not email or not password:
            slicer.util.errorDisplay("Enter email and password.")
            return

        try:
            with slicer.util.tryWithErrorDisplay("Login failed", waitCursor=True):
                self._log(f"Connecting to {url}…")
                result = self.logic.login(url, email, password)
                module_settings.save_settings(
                    selected_server_url=url,
                    last_email=result.get("user_email") or email,
                )
                # If the user typed a new server URL, keep it in the saved list.
                cfg = module_settings.load_settings()
                if url not in (cfg.get("server_urls") or []):
                    module_settings.save_settings(
                        server_urls=list(cfg.get("server_urls") or []) + [url]
                    )
                    self._applySettingsToUi(module_settings.load_settings())
                self._log(f"Login OK: {result.get('user_email') or email}")
                self._datasets = self.logic.listDatasets(url) or []
                self._populateDatasets()
                self._setConnected(True, email=result.get("user_email") or email)
                if self.predictionPanel:
                    self.predictionPanel.set_connected(True)
                if self._datasets:
                    self.onDatasetChanged(self.datasetCombo.currentIndex)
        except Exception:
            self._setConnected(False)
            if self.predictionPanel:
                self.predictionPanel.set_connected(False)
            raise

    def onDisconnect(self):
        self.logic.logout()
        self._datasets = []
        self._selected_dataset = None
        self._loaded_case = None
        self.datasetCombo.clear()
        self._clearCaseTable()
        self._setConnected(False)
        if self.predictionPanel:
            self.predictionPanel.set_connected(False)
        self._updateCaseUi()
        self._log("Disconnected.")

    def _populateDatasets(self):
        self.datasetCombo.blockSignals(True)
        self.datasetCombo.clear()
        for ds in self._datasets:
            if not isinstance(ds, dict):
                continue
            ds_id = ds.get("id") or ds.get("name") or "?"
            name = ds.get("name") or ""
            label = f"{ds_id}" if not name or name == ds_id else f"{ds_id} — {name}"
            self.datasetCombo.addItem(label, ds)
        self.datasetCombo.blockSignals(False)

    def onDatasetChanged(self, index):
        self._selected_dataset = self._selectedDataset()
        if not self._selected_dataset:
            self._clearCaseTable()
            return
        self.onRefreshCases()

    def onSplitChanged(self, index):
        if self._selected_dataset:
            self.onRefreshCases()

    def onRefreshCases(self):
        url = self._currentServerUrl()
        ds = self._selected_dataset
        if not url or not ds:
            return
        ds_id = ds.get("id")
        images_for = self._selectedImagesFor()
        try:
            with slicer.util.tryWithErrorDisplay("Failed to list cases", waitCursor=True):
                name_list = self.logic.listCases(url, ds_id)
                case_nums = nnunet_client.case_nums_from_image_name_list(
                    name_list, images_for
                )
                status_by_num = nnunet_client.label_status_map_from_image_name_list(
                    name_list, images_for
                )
                self._populateCaseTable(case_nums, status_by_num)
                self._log(
                    f"Dataset {ds_id} ({images_for}): {len(case_nums)} case(s)"
                )
        except Exception:
            raise

    def onFetchCase(self):
        url = self._currentServerUrl()
        ds = self._selected_dataset
        num = self._selectedCaseNum()
        images_for = self._selectedImagesFor()
        if not url or not ds or num is None:
            slicer.util.warningDisplay("Select a dataset and case first.")
            return

        ds_id = ds.get("id")
        try:
            with slicer.util.tryWithErrorDisplay("Failed to fetch case", waitCursor=True):
                self._log(f"Downloading {ds_id} / {images_for} / case {num}…")
                result = self.logic.fetchCase(url, ds_id, images_for, num, ds)
                self._volume_node = result["volume_node"]
                self._segmentation_node = result["segmentation_node"]
                self._temp_dir = result["temp_dir"]
                self._loaded_case = {
                    "server_url": url,
                    "dataset_id": ds_id,
                    "images_for": images_for,
                    "num": num,
                    "dataset": ds,
                    "labels": result["labels"],
                    "had_label": result["had_label"],
                }
                self._updateCaseUi()
                self._log(
                    f"Loaded volume '{self._volume_node.GetName()}' and "
                    f"segmentation '{self._segmentation_node.GetName()}' "
                    f"(label file {'found' if result['had_label'] else 'missing — empty segments'})."
                )
                self.onOpenSegmentEditor()
        except Exception:
            raise

    def onOpenSegmentEditor(self):
        if not self._volume_node or not self._segmentation_node:
            return
        self.logic.showInSegmentEditor(self._volume_node, self._segmentation_node)
        self._log("Segment Editor opened. Paint segments, then Save Labels to Server.")

    def onSaveCase(self):
        if not self._loaded_case or not self._segmentation_node or not self._volume_node:
            slicer.util.warningDisplay("Fetch a case before saving.")
            return
        c = self._loaded_case
        try:
            with slicer.util.tryWithErrorDisplay("Failed to save case", waitCursor=True):
                self._log(
                    f"Saving {c['dataset_id']} / {c['images_for']} / case {c['num']}…"
                )
                self.logic.saveCase(
                    server_url=c["server_url"],
                    dataset_id=c["dataset_id"],
                    images_for=c["images_for"],
                    num=c["num"],
                    volume_node=self._volume_node,
                    segmentation_node=self._segmentation_node,
                    labels_map=c["labels"],
                )
                self._log("Saved image + labels to server.")
                slicer.util.infoDisplay("Case saved to nnU-Net server.")
        except Exception:
            raise


#
# NnUNetDashboardLogic
#


class NnUNetDashboardLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def login(self, server_url, email, password):
        return nnunet_client.login(server_url, email, password)

    def logout(self):
        nnunet_client.clear_auth_session()

    def listDatasets(self, server_url):
        return nnunet_client.get_dataset_json_list(server_url)

    def listCases(self, server_url, dataset_id):
        return nnunet_client.get_dataset_image_name_list(server_url, dataset_id)

    def setLabelStatus(self, server_url, dataset_id, images_for, num, status):
        return nnunet_client.set_label_status(
            server_url, dataset_id, images_for, num, status
        )

    def _workdir(self):
        root = os.path.join(tempfile.gettempdir(), "slicer_nnunet_dashboard")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, uuid.uuid4().hex)
        os.makedirs(path, exist_ok=True)
        return path

    def fetchCase(self, server_url, dataset_id, images_for, num, dataset_json):
        out_dir = self._workdir()
        image_info = nnunet_client.download_dataset_image(
            server_url, dataset_id, images_for, num, out_dir, ch_number=0
        )
        image_path = image_info["downloaded_base_image_path"]

        labels_map = nnunet_client.labels_from_dataset_json(dataset_json)
        label_path = None
        had_label = False
        try:
            label_info = nnunet_client.download_dataset_label(
                server_url, dataset_id, images_for, num, out_dir
            )
            label_path = label_info["downloaded_labels_image_path"]
            had_label = True
        except nnunet_client.ServerError as e:
            if "Label file not found" not in str(e):
                raise

        volume_name = f"{dataset_id}_{images_for}_{num}"
        volume_node = slicer.util.loadVolume(image_path, {"name": volume_name})
        if volume_node is None:
            raise RuntimeError(f"Slicer failed to load volume: {image_path}")

        seg_name = f"{volume_name}_Segmentation"
        # Remove prior node with same name if reloading.
        existing = slicer.mrmlScene.GetFirstNodeByName(seg_name)
        if existing:
            slicer.mrmlScene.RemoveNode(existing)

        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", seg_name
        )
        segmentation_node.CreateDefaultDisplayNodes()
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

        if had_label and label_path:
            labelmap_node = slicer.util.loadLabelVolume(
                label_path, {"name": f"{volume_name}_LabelMap"}
            )
            if labelmap_node is None:
                raise RuntimeError(f"Slicer failed to load label: {label_path}")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap_node, segmentation_node
            )
            # Rename segments to dataset.json names when label values match.
            self._renameSegmentsFromLabels(segmentation_node, labels_map)
            slicer.mrmlScene.RemoveNode(labelmap_node)
        else:
            # Create empty segments for each non-background label.
            for name in sorted(labels_map.keys(), key=lambda n: labels_map[n]):
                segmentation_node.GetSegmentation().AddEmptySegment(str(name))

        # Show in slice viewers
        slicer.util.setSliceViewerLayers(background=volume_node, fit=True)

        return {
            "temp_dir": out_dir,
            "volume_node": volume_node,
            "segmentation_node": segmentation_node,
            "labels": labels_map,
            "had_label": had_label,
            "image_path": image_path,
            "label_path": label_path,
        }

    def _renameSegmentsFromLabels(self, segmentation_node, labels_map):
        """Map imported segment names to dataset.json names by label value."""
        value_to_name = {int(v): n for n, v in (labels_map or {}).items()}
        segmentation = segmentation_node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(sid)
            # Imported name is often the label value as string, or "Segment_N"
            label_value = None
            try:
                # Slicer stores label value in segment tag in recent versions
                if hasattr(segment, "GetLabelValue"):
                    label_value = int(segment.GetLabelValue())
            except Exception:
                label_value = None
            if label_value is None:
                try:
                    label_value = int(segment.GetName())
                except Exception:
                    continue
            name = value_to_name.get(label_value)
            if name:
                segment.SetName(str(name))

    def showInSegmentEditor(self, volume_node, segmentation_node):
        slicer.util.selectModule("SegmentEditor")
        # Access the Segment Editor widget representation
        try:
            widget = slicer.modules.segmenteditor.widgetRepresentation()
            editor = widget.self().editor
            editor.setSegmentationNode(segmentation_node)
            editor.setSourceVolumeNode(volume_node)
        except Exception as e:
            logging.warning(f"Could not bind Segment Editor nodes automatically: {e}")

    def saveCase(
        self,
        server_url,
        dataset_id,
        images_for,
        num,
        volume_node,
        segmentation_node,
        labels_map,
    ):
        out_dir = self._workdir()
        image_path = os.path.join(out_dir, f"image_{num}_0000.mha")
        labels_path = os.path.join(out_dir, f"label_{num}.mha")

        if not slicer.util.saveNode(volume_node, image_path):
            raise RuntimeError(f"Failed to save volume to {image_path}")

        self._exportPackedLabelmap(
            segmentation_node, volume_node, labels_map, labels_path
        )

        return nnunet_client.update_image_and_labels(
            server_url,
            dataset_id,
            images_for,
            num,
            image_path,
            labels_path,
            ch_number=0,
        )

    def _exportPackedLabelmap(
        self, segmentation_node, reference_volume_node, labels_map, out_path
    ):
        """Pack Segment Editor segments into a multi-class .mha using dataset.json values."""
        import numpy as np

        ref = slicer.util.arrayFromVolume(reference_volume_node)
        packed = np.zeros(ref.shape, dtype=np.int16)
        segmentation = segmentation_node.GetSegmentation()
        logic = slicer.modules.segmentations.logic()

        # Build name -> segmentId map (case-insensitive)
        name_to_id = {}
        for i in range(segmentation.GetNumberOfSegments()):
            sid = segmentation.GetNthSegmentID(i)
            seg = segmentation.GetSegment(sid)
            name_to_id[seg.GetName().strip().lower()] = sid

        for name, value in (labels_map or {}).items():
            try:
                iv = int(value)
            except (TypeError, ValueError):
                continue
            if iv <= 0:
                continue
            sid = name_to_id.get(str(name).strip().lower())
            if not sid:
                logging.warning(f"No segment named '{name}' — skipping class {iv}")
                continue

            temp_lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(sid)
                ok = logic.ExportSegmentsToLabelmapNode(
                    segmentation_node, ids, temp_lm, reference_volume_node
                )
                if not ok:
                    logging.warning(f"Export failed for segment '{name}'")
                    continue
                arr = slicer.util.arrayFromVolume(temp_lm)
                packed[arr > 0] = iv
            finally:
                slicer.mrmlScene.RemoveNode(temp_lm)

        # Write packed array into an oriented labelmap and save as .mha
        out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        try:
            slicer.modules.volumes.logic().CreateLabelVolumeFromVolume(
                slicer.mrmlScene, out_node, reference_volume_node
            )
            out_arr = slicer.util.arrayFromVolume(out_node)
            if out_arr.shape != packed.shape:
                raise RuntimeError(
                    f"Label shape mismatch: packed {packed.shape} vs volume {out_arr.shape}"
                )
            out_arr[:] = packed
            slicer.util.arrayFromVolumeModified(out_node)
            if not slicer.util.saveNode(out_node, out_path):
                raise RuntimeError(f"Failed to save label to {out_path}")
        finally:
            slicer.mrmlScene.RemoveNode(out_node)

    def importPredictionLabelmap(
        self,
        labels_path,
        labels_map,
        volume_node,
        model_name="Prediction",
        job_seq=None,
        log_fn=None,
    ):
        """Load a prediction labelmap into a new segmentation; keep existing nodes."""
        import numpy as np

        log = log_fn or (lambda m: logging.info(m))
        keep_values = set()
        for _name, value in (labels_map or {}).items():
            try:
                iv = int(value)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                keep_values.add(iv)

        labelmap_node = slicer.util.loadLabelVolume(
            labels_path,
            {"name": f"PredictionLabelMap_{job_seq or uuid.uuid4().hex[:8]}"},
        )
        if labelmap_node is None:
            raise RuntimeError(f"Slicer failed to load prediction label: {labels_path}")

        try:
            if keep_values:
                arr = slicer.util.arrayFromVolume(labelmap_node)
                mask = np.isin(arr, list(keep_values))
                if not mask.all():
                    arr[~mask] = 0
                    slicer.util.arrayFromVolumeModified(labelmap_node)
                    log(
                        f"Kept label classes: {sorted(keep_values)} "
                        "(other values zeroed before import)."
                    )

            base = str(model_name or "Prediction").strip() or "Prediction"
            seq_suffix = f"_{job_seq}" if job_seq is not None else ""
            seg_name = _unique_prediction_seg_name(f"Pred_{base}{seq_suffix}")

            segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", seg_name
            )
            segmentation_node.CreateDefaultDisplayNodes()
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
                volume_node
            )
            ok = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap_node, segmentation_node
            )
            if not ok:
                slicer.mrmlScene.RemoveNode(segmentation_node)
                raise RuntimeError("ImportLabelmapToSegmentationNode failed.")

            self._renameSegmentsFromLabels(segmentation_node, labels_map)
            n = segmentation_node.GetSegmentation().GetNumberOfSegments()
            log(f"Imported {n} segment(s) into '{seg_name}'.")
            return segmentation_node
        finally:
            slicer.mrmlScene.RemoveNode(labelmap_node)


def _unique_prediction_seg_name(base_name):
    base = (base_name or "Prediction").strip() or "Prediction"
    # Sanitize characters that are awkward in the subject hierarchy.
    base = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in base)
    if not slicer.mrmlScene.GetFirstNodeByName(base):
        return base
    index = 2
    while slicer.mrmlScene.GetFirstNodeByName(f"{base}_{index}"):
        index += 1
    return f"{base}_{index}"


#
# NnUNetDashboardTest
#


class NnUNetDashboardTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_client_helpers()

    def test_client_helpers(self):
        self.delayDisplay("Testing NnUNetClient helpers")
        labels = nnunet_client.labels_from_dataset_json(
            {"labels": {"background": 0, "tumor": 1, "organ": 2}}
        )
        self.assertEqual(labels, {"tumor": 1, "organ": 2})
        nums = nnunet_client.case_nums_from_image_name_list(
            {
                "train_images": [
                    {"filename": "x_0_0000.mha", "num": 0},
                    {"filename": "x_3_0000.mha", "num": 3},
                ]
            },
            "train",
        )
        self.assertEqual(nums, [0, 3])
        self.delayDisplay("Test passed")
