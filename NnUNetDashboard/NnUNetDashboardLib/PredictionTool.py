"""nnU-Net Prediction Tool panel for the Slicer NnUNet Dashboard module.

Ports the desktop ``nnunet_prediction_tool_dialog`` workflow: select an approved
model, run auto-segment on the loaded volume (or multi-channel case download),
poll the job, and import result segments into a new segmentation node.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from urllib.parse import urlsplit

import qt
import ctk
import slicer

from . import NnUNetClient as nnunet_client
from . import ModuleSettings as module_settings

NEXT_AVAILABLE_SERVER = "Next Available Server"
_PREFS_PREFIX = "NnUNetDashboard/prediction"
_logger = logging.getLogger(__name__)

FOLD_MODE_ALL = "all"
FOLD_MODE_ENSEMBLE = "ensemble"
FOLD_LABEL_FAST = "Fast – Single Model"
FOLD_LABEL_ACCURATE = "Accurate – 5-Fold Ensemble"


def _short_host(url):
    parsed = urlsplit(str(url or ""))
    return parsed.netloc or str(url or "?")


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _importable_label_items(labels_map):
    items = []
    for name, value in (labels_map or {}).items():
        try:
            iv = int(value)
        except (TypeError, ValueError):
            continue
        if iv <= 0:
            continue
        items.append((str(name), iv))
    items.sort(key=lambda pair: (pair[1], pair[0].lower()))
    return items


def _labels_from_dataset_json(dataset_json):
    if not isinstance(dataset_json, dict):
        return {}
    labels = dataset_json.get("labels") or {}
    return labels if isinstance(labels, dict) else {}


class PredictionToolPanel:
    """Collapsible Prediction Tool UI embedded in NnUNetDashboardWidget."""

    def __init__(self, parent_widget):
        self.parent = parent_widget
        self._models = []
        self._model_detail = None
        self._jobs = {}
        self._job_seq = 0
        self._restoring_prefs = False
        self._preferred_model = None
        self._preferred_fold_mode = None

        self._poll_timer = qt.QTimer()
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._poll_all_jobs)

        self._filter_debounce_timer = qt.QTimer()
        self._filter_debounce_timer.setSingleShot(True)
        self._filter_debounce_timer.setInterval(300)
        self._filter_debounce_timer.timeout.connect(self._apply_model_filter)

        self._load_persisted_prefs()

    def build(self, parent_layout):
        collapsible = ctk.ctkCollapsibleButton()
        collapsible.text = "Prediction Tool"
        collapsible.collapsed = True
        parent_layout.addWidget(collapsible)
        layout = qt.QVBoxLayout(collapsible)

        note = qt.QLabel(
            "Run an approved nnU-Net model on the loaded volume. "
            "Results are imported as a new segmentation (existing labels are kept)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)

        form = qt.QFormLayout()

        self.modelFilterEdit = qt.QLineEdit()
        self.modelFilterEdit.setPlaceholderText(
            "Filter by organ, configuration, description, model name…"
        )
        self.modelFilterEdit.setToolTip(
            "Case-insensitive filter over model fields "
            "(dataset, configuration, trainer, plans, description, name, organ, etc.)."
        )
        if self._persisted_filter:
            self.modelFilterEdit.setText(self._persisted_filter)
        self.modelFilterEdit.textChanged.connect(self._on_model_filter_changed)
        form.addRow("Filter:", self.modelFilterEdit)

        modelRow = qt.QHBoxLayout()
        self.modelCombo = qt.QComboBox()
        self.modelCombo.setToolTip("Approved prediction models from the nnU-Net server")
        self.modelCombo.currentIndexChanged.connect(self._on_model_changed)
        self.refreshModelsButton = qt.QPushButton("Refresh")
        self.refreshModelsButton.setToolTip("Reload approved models from the logged-in server")
        self.refreshModelsButton.clicked.connect(self.load_approved_models)
        self.modelDocsButton = qt.QPushButton("Info")
        self.modelDocsButton.setToolTip("Open this model's documentation")
        self.modelDocsButton.setEnabled(False)
        self.modelDocsButton.clicked.connect(self._open_model_docs)
        modelRow.addWidget(self.modelCombo, 1)
        modelRow.addWidget(self.refreshModelsButton)
        modelRow.addWidget(self.modelDocsButton)
        form.addRow("Model:", modelRow)

        self.inputDatasetLabel = qt.QLabel("-")
        self.inputDatasetLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow("Input Dataset:", self.inputDatasetLabel)

        self.inputCaseLabel = qt.QLabel("-")
        self.inputCaseLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow("Input Image Set:", self.inputCaseLabel)

        self.channelsLabel = qt.QLabel("-")
        form.addRow("Model Channels:", self.channelsLabel)

        self.foldModeCombo = qt.QComboBox()
        self.foldModeCombo.setToolTip(
            "Fast – Single Model: fold_all (one model trained on all cases; quicker).\n"
            "Accurate – 5-Fold Ensemble: standard CV ensemble of folds 0–4 (slower)."
        )
        self.foldModeCombo.currentIndexChanged.connect(self._on_fold_mode_changed)
        form.addRow("Mode:", self.foldModeCombo)
        self._update_fold_mode_combo(fold_all_available=False, announce=False)

        labelsPanel = qt.QWidget()
        labelsLayout = qt.QVBoxLayout(labelsPanel)
        labelsLayout.setContentsMargins(0, 0, 0, 0)
        labelsLayout.setSpacing(4)

        labelsBtnRow = qt.QHBoxLayout()
        self.selectAllLabelsButton = qt.QPushButton("Select All")
        self.selectAllLabelsButton.clicked.connect(lambda: self._set_all_import_labels_checked(True))
        self.clearAllLabelsButton = qt.QPushButton("Clear All")
        self.clearAllLabelsButton.clicked.connect(lambda: self._set_all_import_labels_checked(False))
        labelsBtnRow.addWidget(self.selectAllLabelsButton)
        labelsBtnRow.addWidget(self.clearAllLabelsButton)
        labelsBtnRow.addStretch(1)
        labelsLayout.addLayout(labelsBtnRow)

        self.importLabelsList = qt.QListWidget()
        self.importLabelsList.setToolTip(
            "Checked labels will be imported into a new segmentation after prediction."
        )
        self.importLabelsList.setMinimumHeight(100)
        self.importLabelsList.setMaximumHeight(180)
        self.importLabelsList.setSelectionMode(qt.QAbstractItemView.NoSelection)
        labelsLayout.addWidget(self.importLabelsList)
        form.addRow("Labels to Import:", labelsPanel)

        layout.addLayout(form)

        layout.addWidget(qt.QLabel("Status:"))
        self.statusTabs = qt.QTabWidget()
        self.generalStatusView = qt.QTextEdit()
        self.generalStatusView.setReadOnly(True)
        self.generalStatusView.setPlaceholderText(
            "General messages appear here. Each Run Auto Segment adds a job tab."
        )
        self.generalStatusView.setMaximumHeight(140)
        self.statusTabs.addTab(self.generalStatusView, "General")
        layout.addWidget(self.statusTabs)

        btnRow = qt.QHBoxLayout()
        self.runButton = qt.QPushButton("Run Auto Segment")
        self.runButton.clicked.connect(self._on_run_clicked)
        self.predictionServerCombo = qt.QComboBox()
        self.predictionServerCombo.setMinimumWidth(180)
        self.predictionServerCombo.setToolTip(
            "Where to run the prediction.\n"
            f"“{NEXT_AVAILABLE_SERVER}” load-balances among configured servers "
            "that have the selected model.\n"
            "Pick a specific server to send the job there."
        )
        self.cancelJobsButton = qt.QPushButton("Cancel Jobs")
        self.cancelJobsButton.setToolTip("Cancel all active prediction jobs on their servers")
        self.cancelJobsButton.clicked.connect(self._on_cancel_clicked)
        btnRow.addWidget(self.runButton)
        btnRow.addWidget(qt.QLabel("Server:"))
        btnRow.addWidget(self.predictionServerCombo, 1)
        btnRow.addWidget(self.cancelJobsButton)
        layout.addLayout(btnRow)

        self._reload_prediction_server_combo()
        self._clear_import_labels_list()
        self.refresh_context()
        self.set_connected(False)

    # ------------------------------------------------------------------ prefs

    def _load_persisted_prefs(self):
        settings = qt.QSettings()
        self._persisted_filter = str(settings.value(f"{_PREFS_PREFIX}/filter", "") or "")
        dataset_id = str(settings.value(f"{_PREFS_PREFIX}/model_dataset_id", "") or "").strip()
        trainer = str(settings.value(f"{_PREFS_PREFIX}/model_trainer", "") or "").strip()
        plans = str(settings.value(f"{_PREFS_PREFIX}/model_plans", "") or "").strip()
        configuration = str(
            settings.value(f"{_PREFS_PREFIX}/model_configuration", "") or ""
        ).strip()
        fold_mode = str(settings.value(f"{_PREFS_PREFIX}/fold_mode", "") or "").strip().lower()
        self._preferred_model = None
        if dataset_id:
            self._preferred_model = {
                "dataset_id": dataset_id,
                "trainer": trainer or None,
                "plans": plans or None,
                "configuration": configuration or None,
            }
        self._preferred_fold_mode = (
            fold_mode if fold_mode in (FOLD_MODE_ALL, FOLD_MODE_ENSEMBLE) else None
        )

    def _persist_prefs(self):
        if self._restoring_prefs:
            return
        try:
            settings = qt.QSettings()
            settings.setValue(f"{_PREFS_PREFIX}/filter", self.modelFilterEdit.text or "")
            model = self._selected_model()
            if isinstance(model, dict) and model.get("dataset_id"):
                settings.setValue(
                    f"{_PREFS_PREFIX}/model_dataset_id", model.get("dataset_id") or ""
                )
                settings.setValue(
                    f"{_PREFS_PREFIX}/model_trainer", model.get("trainer") or ""
                )
                settings.setValue(
                    f"{_PREFS_PREFIX}/model_plans", model.get("plans") or ""
                )
                settings.setValue(
                    f"{_PREFS_PREFIX}/model_configuration",
                    model.get("configuration") or "",
                )
                self._preferred_model = {
                    "dataset_id": model.get("dataset_id"),
                    "trainer": model.get("trainer"),
                    "plans": model.get("plans"),
                    "configuration": model.get("configuration"),
                }
            fold_mode = self._selected_fold_mode()
            if fold_mode:
                settings.setValue(f"{_PREFS_PREFIX}/fold_mode", fold_mode)
                self._preferred_fold_mode = fold_mode
            settings.sync()
        except Exception as e:
            _logger.warning("Failed to persist prediction prefs: %s", e)

    # ------------------------------------------------------------------ context / connection

    def set_connected(self, connected):
        self.refreshModelsButton.setEnabled(bool(connected))
        self.modelFilterEdit.setEnabled(bool(connected))
        self.modelCombo.setEnabled(bool(connected))
        self.predictionServerCombo.setEnabled(bool(connected))
        if hasattr(self, "foldModeCombo") and self.foldModeCombo is not None:
            self.foldModeCombo.setEnabled(bool(connected))
        if connected:
            self._reload_prediction_server_combo()
            self.load_approved_models()
        else:
            self._models = []
            self._model_detail = None
            self.modelCombo.blockSignals(True)
            self.modelCombo.clear()
            self.modelCombo.addItem("(not connected)")
            self.modelCombo.blockSignals(False)
            self.channelsLabel.setText("-")
            self._clear_import_labels_list()
            self._update_fold_mode_combo(fold_all_available=False, announce=False)
            self.refresh_context()

    def refresh_context(self):
        ctx = self._context()
        case = ctx.get("case") or {}
        dataset = ctx.get("dataset") or {}
        dataset_id = case.get("dataset_id") or dataset.get("id") or "-"
        images_for = case.get("images_for") or "-"
        num = case.get("num")
        case_txt = f"{images_for} / case {num}" if num is not None else "-"
        self.inputDatasetLabel.setText(str(dataset_id))
        self.inputCaseLabel.setText(case_txt)

        has_image = self._input_volume() is not None
        has_server = bool(ctx.get("server_url")) and nnunet_client.is_authenticated()
        has_model = self._selected_model() is not None
        model_count = self._combo_count(self.modelCombo)
        can_run = bool(has_image and has_server and model_count > 0 and has_model)
        self.runButton.setEnabled(can_run)

        reasons = []
        if not has_server:
            reasons.append("connect to the server")
        if not has_image:
            reasons.append("fetch/load a volume")
        if not has_model:
            reasons.append("select an approved model")
        if can_run:
            self.runButton.setToolTip("Run auto-segmentation with the selected model")
        else:
            self.runButton.setToolTip(
                "Run Auto Segment is disabled — " + ", ".join(reasons) + "."
                if reasons
                else "Run Auto Segment is disabled."
            )

    def on_servers_changed(self):
        self._reload_prediction_server_combo()

    def _combo_count(self, combo):
        count = combo.count
        return int(count() if callable(count) else count)

    def _combo_current_index(self, combo):
        index = combo.currentIndex
        return int(index() if callable(index) else index)

    def _context(self):
        w = self.parent
        case = getattr(w, "_loaded_case", None) or {}
        server_url = ""
        try:
            server_url = w._currentServerUrl()
        except Exception:
            server_url = ""
        if not nnunet_client.is_authenticated():
            server_url = ""
        return {
            "server_url": server_url,
            "case": {
                "dataset_id": case.get("dataset_id"),
                "images_for": case.get("images_for"),
                "num": case.get("num"),
            }
            if case
            else {},
            "dataset": case.get("dataset")
            or getattr(w, "_selected_dataset", None)
            or {},
        }

    def _input_volume(self):
        """Return the volume to predict on.

        Prefers the module's fetched volume, but falls back to the slice
        background / any scalar volume so Reload does not leave Run disabled
        while an image is still visible in the scene.
        """
        w = self.parent
        vol = getattr(w, "_volume_node", None)
        if self._is_usable_volume(vol):
            return vol

        # Recover after module reload: match fetched case name if present.
        case = getattr(w, "_loaded_case", None) or {}
        if case.get("dataset_id") is not None and case.get("num") is not None:
            expected = f"{case['dataset_id']}_{case.get('images_for')}_{case['num']}"
            node = slicer.mrmlScene.GetFirstNodeByName(expected)
            if self._is_usable_volume(node):
                w._volume_node = node
                return node

        # Active slice background (what the user is looking at).
        try:
            layoutManager = slicer.app.layoutManager()
            if layoutManager is not None:
                red = layoutManager.sliceWidget("Red")
                if red is not None:
                    logic = red.sliceLogic()
                    bg_id = logic.GetSliceCompositeNode().GetBackgroundVolumeID()
                    if bg_id:
                        node = slicer.mrmlScene.GetNodeByID(bg_id)
                        if self._is_usable_volume(node):
                            w._volume_node = node
                            return node
        except Exception:
            pass

        # Last resort: first scalar volume that is not a labelmap.
        try:
            nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            for node in nodes or []:
                if self._is_usable_volume(node):
                    w._volume_node = node
                    return node
        except Exception:
            pass
        return None

    @staticmethod
    def _is_usable_volume(node):
        if node is None:
            return False
        try:
            if node.IsA("vtkMRMLLabelMapVolumeNode"):
                return False
            # Exclude empty / proxy nodes.
            if hasattr(node, "GetImageData") and node.GetImageData() is None:
                return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ logging

    def _append_general(self, msg):
        self.generalStatusView.append(str(msg))
        sb = self.generalStatusView.verticalScrollBar()
        sb.setValue(sb.maximum)
        slicer.app.processEvents()

    def _set_general(self, msg):
        self.generalStatusView.setPlainText(str(msg))

    def _append_job(self, job, msg):
        view = job.get("status_view") if isinstance(job, dict) else None
        if view is None:
            self._append_general(msg)
            return
        view.append(str(msg))
        sb = view.verticalScrollBar()
        sb.setValue(sb.maximum)
        slicer.app.processEvents()

    # ------------------------------------------------------------------ models

    def _model_display_name(self, model):
        base = nnunet_client.model_display_name(model)
        license_ = model.get("license") if isinstance(model, dict) else None
        if not license_:
            return base
        if model.get("required_scope"):
            return f"{base} [research-only]"
        return f"{base} [open license]"

    def _model_tooltip(self, model):
        license_ = model.get("license") if isinstance(model, dict) else None
        if not license_:
            return "No external license — locally trained model."
        lines = [f"License: {license_}"]
        scope = model.get("required_scope")
        if scope:
            lines.append(f"Requires the '{scope}' scope/role on your account.")
        return "\n".join(lines)

    def _filtered_models(self):
        query_attr = self.modelFilterEdit.text
        query = (query_attr() if callable(query_attr) else query_attr or "").strip().lower()
        if not query:
            return list(self._models)
        return [
            m
            for m in self._models
            if query in nnunet_client.model_search_text(m).lower()
        ]

    def _selected_model(self):
        data = self.modelCombo.itemData(self._combo_current_index(self.modelCombo))
        return data if isinstance(data, dict) else None

    def _selected_fold_mode(self):
        combo = getattr(self, "foldModeCombo", None)
        if combo is None:
            return FOLD_MODE_ENSEMBLE
        data = combo.itemData(self._combo_current_index(combo))
        if data in (FOLD_MODE_ALL, FOLD_MODE_ENSEMBLE):
            return data
        return FOLD_MODE_ENSEMBLE

    def _fold_all_available_for_selection(self):
        if isinstance(self._model_detail, dict) and "fold_all_available" in self._model_detail:
            return bool(self._model_detail.get("fold_all_available"))
        model = self._selected_model()
        if isinstance(model, dict) and "fold_all_available" in model:
            return bool(model.get("fold_all_available"))
        return False

    def _update_fold_mode_combo(self, fold_all_available=None, announce=True):
        combo = getattr(self, "foldModeCombo", None)
        if combo is None:
            return
        if fold_all_available is None:
            fold_all_available = self._fold_all_available_for_selection()

        previous = self._selected_fold_mode() if self._combo_count(combo) > 0 else None
        preferred = self._preferred_fold_mode
        self._restoring_prefs = True
        combo.blockSignals(True)
        try:
            combo.clear()
            if fold_all_available:
                combo.addItem(FOLD_LABEL_FAST, FOLD_MODE_ALL)
            combo.addItem(FOLD_LABEL_ACCURATE, FOLD_MODE_ENSEMBLE)

            want = preferred if preferred in (FOLD_MODE_ALL, FOLD_MODE_ENSEMBLE) else None
            if want is None and fold_all_available:
                want = FOLD_MODE_ALL
            if want is None:
                want = FOLD_MODE_ENSEMBLE
            idx = combo.findData(want)
            if idx < 0:
                idx = 0
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
            self._restoring_prefs = False

        selected = self._selected_fold_mode()
        if announce and previous == FOLD_MODE_ALL and selected != FOLD_MODE_ALL:
            self._append_general(
                "fold_all is not available for this model; "
                "switched Mode to Accurate – 5-Fold Ensemble."
            )

    def _on_fold_mode_changed(self, _index=0):
        if self._restoring_prefs:
            return
        self._persist_prefs()

    def _on_model_filter_changed(self, _text=""):
        self._filter_debounce_timer.start()

    def _apply_model_filter(self):
        self._persist_prefs()
        if not self._models:
            return
        self._populate_model_combo()

    def load_approved_models(self):
        self._models = []
        ctx = self._context()
        base_url = ctx.get("server_url")
        if not base_url:
            self.modelCombo.blockSignals(True)
            self.modelCombo.clear()
            self.modelCombo.addItem("(no server)")
            self.modelCombo.blockSignals(False)
            self._set_general("Connect to the nnU-Net server first.")
            self.refresh_context()
            return

        try:
            self._append_general("Fetching approved prediction models…")
            with slicer.util.tryWithErrorDisplay(
                "Failed to load approved models", waitCursor=True
            ):
                models = nnunet_client.get_approved_models(base_url) or []
            self._models = models
            self._populate_model_combo()
            query_attr = self.modelFilterEdit.text
            query = (
                query_attr() if callable(query_attr) else query_attr or ""
            ).strip()
            matched = len(self._filtered_models())
            if not self._models:
                self._set_general("No approved models found on the server.")
            elif query:
                self._set_general(
                    f"Loaded {len(self._models)} approved model(s); "
                    f"showing {matched} matching “{query}”."
                )
            else:
                self._set_general(
                    f"Loaded {len(self._models)} approved model(s)."
                )
        except Exception as e:
            self.modelCombo.blockSignals(True)
            self.modelCombo.clear()
            self.modelCombo.addItem("(failed to load models)")
            self.modelCombo.blockSignals(False)
            self._set_general(f"Failed to load approved models:\n{e}")
            self.refresh_context()


    def _populate_model_combo(self, preferred_model=None):
        previous = (
            preferred_model
            if isinstance(preferred_model, dict)
            else (
                self._preferred_model
                if isinstance(self._preferred_model, dict)
                else None
            )
        )
        if previous is None:
            previous = self._selected_model()

        self._restoring_prefs = True
        self.modelCombo.blockSignals(True)
        try:
            self.modelCombo.clear()
            filtered = self._filtered_models()
            if not self._models:
                self.modelCombo.addItem("(no approved models)")
            elif not filtered:
                self.modelCombo.addItem("(no matching models)")
            else:
                select_index = 0
                for i, m in enumerate(filtered):
                    self.modelCombo.addItem(self._model_display_name(m), m)
                    idx = self.modelCombo.count - 1
                    self.modelCombo.setItemData(
                        idx, self._model_tooltip(m), qt.Qt.ToolTipRole
                    )
                    if previous and (
                        previous.get("dataset_id") == m.get("dataset_id")
                        and (
                            not previous.get("trainer")
                            or previous.get("trainer") == m.get("trainer")
                        )
                        and (
                            not previous.get("plans")
                            or previous.get("plans") == m.get("plans")
                        )
                        and (
                            not previous.get("configuration")
                            or previous.get("configuration") == m.get("configuration")
                        )
                    ):
                        select_index = i
                self.modelCombo.setCurrentIndex(select_index)
        finally:
            self.modelCombo.blockSignals(False)
            self._restoring_prefs = False

        selected = self._selected_model()
        same_as_previous = bool(
            previous
            and selected
            and previous.get("dataset_id") == selected.get("dataset_id")
            and (
                not previous.get("trainer")
                or previous.get("trainer") == selected.get("trainer")
            )
            and (
                not previous.get("plans")
                or previous.get("plans") == selected.get("plans")
            )
            and (
                not previous.get("configuration")
                or previous.get("configuration") == selected.get("configuration")
            )
        )
        if selected and not same_as_previous:
            self._on_model_changed(self.modelCombo.currentIndex)
        elif not selected:
            self._model_detail = None
            self.modelDocsButton.setEnabled(False)
            self.channelsLabel.setText("-")
            self._clear_import_labels_list()
            self._update_fold_mode_combo(fold_all_available=False, announce=False)
        else:
            if self._model_detail is None:
                self._on_model_changed(self.modelCombo.currentIndex)
            else:
                self.modelDocsButton.setEnabled(
                    bool(
                        selected.get("docs_url")
                        or (self._model_detail or {}).get("docs_url")
                    )
                )
        self.refresh_context()
        self._persist_prefs()

    def _on_model_changed(self, index):
        model = self._selected_model()
        self._model_detail = None
        self.modelDocsButton.setEnabled(bool(model and model.get("docs_url")))
        if not model:
            self.channelsLabel.setText("-")
            self._clear_import_labels_list()
            self._update_fold_mode_combo(fold_all_available=False, announce=False)
            self.refresh_context()
            return

        ctx = self._context()
        base_url = ctx.get("server_url")
        if not base_url:
            return

        self._update_fold_mode_combo(
            fold_all_available=bool(model.get("fold_all_available")),
            announce=True,
        )

        try:
            with slicer.util.tryWithErrorDisplay(
                "Failed to fetch model detail", waitCursor=True
            ):
                detail = nnunet_client.get_model_detail(
                    base_url,
                    model["dataset_id"],
                    model["trainer"],
                    model["plans"],
                    model["configuration"],
                )
            self._model_detail = detail
            dataset_json = detail.get("dataset_json") if isinstance(detail, dict) else {}
            channel_names = nnunet_client.channel_names_from_dataset_json(
                dataset_json or {}
            )
            n_ch = max(1, len(channel_names)) if channel_names else 1
            labels = _labels_from_dataset_json(dataset_json or {})
            importable = _importable_label_items(labels)
            self.channelsLabel.setText(
                nnunet_client.format_channel_names(channel_names)
            )
            self._populate_import_labels_list(labels, checked=True)
            docs = (detail or {}).get("docs_url") or model.get("docs_url")
            self.modelDocsButton.setEnabled(bool(docs))
            fold_all = bool(
                (detail or {}).get("fold_all_available", model.get("fold_all_available"))
            )
            self._update_fold_mode_combo(fold_all_available=fold_all, announce=True)
            mode_label = (
                FOLD_LABEL_FAST
                if self._selected_fold_mode() == FOLD_MODE_ALL
                else FOLD_LABEL_ACCURATE
            )
            self._append_general(
                f"Selected model channels: "
                f"{nnunet_client.format_channel_names(channel_names)} ({n_ch}). "
                f"Labels to import: {len(importable)}. "
                f"fold_all_available={fold_all}. Mode: {mode_label}."
            )
        except Exception as e:
            self.channelsLabel.setText("?")
            self._clear_import_labels_list()
            self._update_fold_mode_combo(
                fold_all_available=bool(model.get("fold_all_available")),
                announce=True,
            )
            self._append_general(f"Failed to fetch model detail: {e}")
        self.refresh_context()
        self._persist_prefs()

    def _open_model_docs(self):
        model = self._selected_model()
        if not model:
            return
        docs_url = (self._model_detail or {}).get("docs_url") or model.get("docs_url")
        if not docs_url:
            return
        ctx = self._context()
        base_url = ctx.get("server_url") or ""
        parsed = urlsplit(base_url)
        host_root = (
            f"{parsed.scheme}://{parsed.netloc}"
            if parsed.scheme and parsed.netloc
            else ""
        )
        full = host_root + docs_url
        ok = qt.QDesktopServices.openUrl(qt.QUrl(full))
        if not ok:
            import webbrowser

            webbrowser.open(full)

    def _clear_import_labels_list(self):
        self.importLabelsList.clear()
        self.selectAllLabelsButton.setEnabled(False)
        self.clearAllLabelsButton.setEnabled(False)

    def _populate_import_labels_list(self, labels_map, checked=True):
        self.importLabelsList.clear()
        items = _importable_label_items(labels_map)
        state = qt.Qt.Checked if checked else qt.Qt.Unchecked
        for name, value in items:
            item = qt.QListWidgetItem(f"{name}  (class {value})")
            item.setFlags(
                qt.Qt.ItemIsEnabled
                | qt.Qt.ItemIsUserCheckable
                | qt.Qt.ItemIsSelectable
            )
            item.setCheckState(state)
            item.setData(qt.Qt.UserRole, {"name": name, "value": value})
            self.importLabelsList.addItem(item)
        enabled = bool(items)
        self.selectAllLabelsButton.setEnabled(enabled)
        self.clearAllLabelsButton.setEnabled(enabled)

    def _set_all_import_labels_checked(self, checked):
        state = qt.Qt.Checked if checked else qt.Qt.Unchecked
        for i in range(self.importLabelsList.count):
            self.importLabelsList.item(i).setCheckState(state)

    def _checked_import_labels(self):
        selected = {}
        for i in range(self.importLabelsList.count):
            item = self.importLabelsList.item(i)
            if item.checkState() != qt.Qt.Checked:
                continue
            data = item.data(qt.Qt.UserRole) or {}
            name = data.get("name")
            value = data.get("value")
            if name is None or value is None:
                continue
            selected[str(name)] = int(value)
        return selected

    # ------------------------------------------------------------------ servers / LB

    def _reload_prediction_server_combo(self):
        combo = getattr(self, "predictionServerCombo", None)
        if combo is None:
            return
        previous = combo.itemData(combo.currentIndex)
        cfg = module_settings.load_settings()
        urls = list(cfg.get("server_urls") or [])
        preferred = (cfg.get("selected_server_url") or "").strip().rstrip("/")
        if preferred and preferred not in urls:
            urls = [preferred] + urls

        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(NEXT_AVAILABLE_SERVER, None)
            for url in urls:
                host = _short_host(url)
                label = f"{host}  ({url})" if host and host != url else str(url)
                combo.addItem(label, url)

            select_index = 0
            if previous:
                idx = combo.findData(previous)
                if idx >= 0:
                    select_index = idx
            combo.setCurrentIndex(select_index)
        finally:
            combo.blockSignals(False)

    def _selected_prediction_server_url(self):
        data = self.predictionServerCombo.itemData(
            self.predictionServerCombo.currentIndex
        )
        return str(data).strip() if data else None

    def _pick_prediction_server(self, model, log_fn):
        require_fold_all = self._selected_fold_mode() == FOLD_MODE_ALL
        forced_url = self._selected_prediction_server_url()
        if forced_url:
            return self._validate_prediction_server(
                forced_url, model, log_fn, require_fold_all=require_fold_all
            )

        cfg = module_settings.load_settings()
        urls = list(cfg.get("server_urls") or [])
        preferred = (cfg.get("selected_server_url") or "").strip().rstrip("/")
        if preferred and preferred not in urls:
            urls = [preferred] + urls

        candidates = []
        for url in urls:
            host = _short_host(url)
            try:
                load = self._probe_prediction_server(
                    url, model, log_fn, require_fold_all=require_fold_all
                )
            except Exception as e:
                log_fn(f"Skip {host}: {e}")
                continue
            if load is None:
                continue
            jobs_ahead = load.get("jobs_ahead")
            wait = load.get("estimated_wait_seconds")
            try:
                jobs_ahead_n = float(jobs_ahead) if jobs_ahead is not None else 1e9
            except (TypeError, ValueError):
                jobs_ahead_n = 1e9
            try:
                wait_n = float(wait) if wait is not None else jobs_ahead_n
            except (TypeError, ValueError):
                wait_n = jobs_ahead_n
            candidates.append((wait_n, jobs_ahead_n, url, load))

        if not candidates:
            if require_fold_all:
                raise RuntimeError(
                    "No configured server has this model with fold_all available "
                    "and reported queue load. Try Accurate – 5-Fold Ensemble, "
                    "or pick a server that has fold_all weights."
                )
            raise RuntimeError(
                "No configured server both has the selected model and reported queue load."
            )

        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        chosen = candidates[0][2]
        log_fn(f"Selected prediction server (next available): {_short_host(chosen)}")
        return chosen

    def _validate_prediction_server(self, url, model, log_fn, require_fold_all=False):
        host = _short_host(url)
        log_fn(f"Using selected prediction server: {host}")
        load = self._probe_prediction_server(
            url,
            model,
            log_fn,
            require_inferencing=True,
            require_fold_all=require_fold_all,
        )
        if load is None:
            extra = (
                ", or fold_all weights missing" if require_fold_all else ""
            )
            raise RuntimeError(
                f"Selected server {_short_host(url)} cannot run this model "
                f"(missing model, inferencing disabled{extra}, "
                "or queue load unavailable)."
            )
        return url

    def _probe_prediction_server(
        self, url, model, log_fn, require_inferencing=True, require_fold_all=False
    ):
        host = _short_host(url)
        try:
            if not nnunet_client.server_has_approved_model(
                url, model, require_fold_all=require_fold_all
            ):
                if require_fold_all:
                    log_fn(
                        f"Skip {host}: selected model not available "
                        "or fold_all weights missing."
                    )
                else:
                    log_fn(f"Skip {host}: selected model not available.")
                return None
        except Exception as e:
            log_fn(f"Skip {host}: could not list approved models ({e}).")
            return None

        try:
            load = nnunet_client.get_prediction_queue_load(
                url,
                dataset_id=model.get("dataset_id"),
                configuration=model.get("configuration"),
            )
        except Exception as e:
            log_fn(f"Skip {host}: /predictions/load failed ({e}).")
            return None

        jobs_ahead = load.get("jobs_ahead")
        wait = load.get("estimated_wait_seconds")
        inferencing = load.get("inferencing_enabled", True)
        log_fn(
            f"{host}: model OK, jobs_ahead={jobs_ahead}, "
            f"estimated_wait_s={wait}, inferencing_enabled={inferencing}"
            + (", fold_all required" if require_fold_all else "")
        )
        if require_inferencing and inferencing is False:
            log_fn(f"Skip {host}: inferencing disabled.")
            return None
        return load

    # ------------------------------------------------------------------ run / poll / import

    def _create_job_tab(self, title):
        view = qt.QTextEdit()
        view.setReadOnly(True)
        view.setMaximumHeight(140)
        index = self.statusTabs.addTab(view, title)
        self.statusTabs.setCurrentIndex(index)
        return view

    def _set_job_tab_title(self, job, title):
        view = job.get("status_view")
        if view is None:
            return
        idx = self.statusTabs.indexOf(view)
        if idx >= 0:
            self.statusTabs.setTabText(idx, title)

    def _active_jobs(self):
        return [
            j
            for j in self._jobs.values()
            if j.get("state") in ("preparing", "queued", "running", "importing")
        ]

    def _ensure_poll_timer(self):
        if self._active_jobs():
            if not self._poll_timer.isActive():
                self._poll_timer.start()
        else:
            self._poll_timer.stop()

    def _download_case_channels(self, base_url, case, num_channels, out_dir, log_fn):
        paths = []
        for ch in range(num_channels):
            log_fn(f"Downloading input channel {ch}/{max(num_channels - 1, 0)}...")
            slicer.app.processEvents()
            result = nnunet_client.download_dataset_image(
                BASE_URL=base_url,
                dataset_id=case["dataset_id"],
                images_for=case["images_for"],
                num=case["num"],
                out_dir=out_dir,
                ch_number=ch,
            )
            path = result.get("downloaded_base_image_path")
            if not path or not os.path.exists(path):
                raise RuntimeError(
                    f"Failed to download channel {ch} for case {case['num']}."
                )
            paths.append(path)
            log_fn(f"Downloaded channel {ch}: {os.path.basename(path)}")
            slicer.app.processEvents()
        return paths

    def _export_viewer_volume(self, out_dir, log_fn):
        volume_node = self._input_volume()
        if volume_node is None:
            raise RuntimeError("No image is loaded. Fetch a case first.")
        path = os.path.join(out_dir, "viewer_channel_0.mha")
        log_fn("Exporting loaded volume…")
        slicer.app.processEvents()
        if not slicer.util.saveNode(volume_node, path):
            raise RuntimeError("Failed to export the loaded volume.")
        log_fn(f"Exported volume: {os.path.basename(path)}")
        return path

    def _on_run_clicked(self):
        ctx = self._context()
        case_base_url = ctx.get("server_url")
        case = ctx.get("case") or {}
        model = self._selected_model()
        volume_node = self._input_volume()

        if volume_node is None:
            slicer.util.warningDisplay("Fetch / load an image first.")
            return
        if not case_base_url or not nnunet_client.is_authenticated():
            slicer.util.warningDisplay("Connect to the nnU-Net server first.")
            return
        if not model:
            slicer.util.warningDisplay("Select an approved prediction model.")
            return

        dataset_json = {}
        if isinstance(self._model_detail, dict):
            dataset_json = self._model_detail.get("dataset_json") or {}
        num_channels = nnunet_client.channel_count_from_dataset_json(dataset_json)
        all_labels = _labels_from_dataset_json(dataset_json)
        selected_labels = self._checked_import_labels()

        if self.importLabelsList.count > 0 and not selected_labels:
            slicer.util.warningDisplay(
                "Select at least one label to import into the segmentation."
            )
            return

        has_case = case.get("dataset_id") is not None and case.get("num") is not None
        use_downloaded_case = False
        if num_channels > 1:
            if not has_case:
                slicer.util.warningDisplay(
                    f"This model requires {num_channels} input channels, but only "
                    "one volume is loaded.\n\n"
                    "Fetch a case from Dataset & Case so the full image set can be "
                    "downloaded from the server."
                )
                return
            reply = qt.QMessageBox.question(
                slicer.util.mainWindow(),
                "Download Case Image Set?",
                f"This model requires {num_channels} input channels, but only one "
                "volume is loaded.\n\n"
                f"Download the full case image set "
                f"({case.get('images_for')} / case {case.get('num')}) from the "
                "logged-on server and use it for prediction?",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
                qt.QMessageBox.Yes,
            )
            if reply != qt.QMessageBox.Yes:
                return
            use_downloaded_case = True

        labels = selected_labels if selected_labels else all_labels
        fold_mode = self._selected_fold_mode()
        fold_label = (
            FOLD_LABEL_FAST if fold_mode == FOLD_MODE_ALL else FOLD_LABEL_ACCURATE
        )

        self._job_seq += 1
        job_uid = uuid.uuid4().hex
        model_label = (
            model.get("dataset_id")
            or model.get("name")
            or model.get("model_name")
            or f"Job {self._job_seq}"
        )
        tab_title = f"#{self._job_seq} {model_label}"
        status_view = self._create_job_tab(tab_title)

        job = {
            "uid": job_uid,
            "seq": self._job_seq,
            "tab_title": tab_title,
            "status_view": status_view,
            "state": "preparing",
            "job_id": None,
            "req_id": None,
            "base_url": None,
            "case_base_url": case_base_url,
            "model_dataset_id": model.get("dataset_id"),
            "labels": labels,
            "fold": fold_mode,
            "out_dir": None,
            "submitted_at": time.monotonic(),
            "model": dict(model),
            "volume_node_id": volume_node.GetID(),
        }
        self._jobs[job_uid] = job

        def log(msg):
            self._append_job(job, msg)

        log(f"Mode: {fold_label} (fold={fold_mode}).")
        log(
            f"Will import "
            f"{len(selected_labels) if selected_labels else len(_importable_label_items(labels))} "
            f"label segment(s) after prediction."
        )
        slicer.app.processEvents()

        out_dir = os.path.join(tempfile.gettempdir(), f"slicer_nnunet_pred_{job_uid}")
        os.makedirs(out_dir, exist_ok=True)
        job["out_dir"] = out_dir

        try:
            if self._selected_prediction_server_url():
                log("Validating selected prediction server...")
            else:
                log(
                    "Choosing next available prediction server "
                    "(model + queue load)..."
                )
            slicer.app.processEvents()
            predict_url = self._pick_prediction_server(model, log)

            if use_downloaded_case:
                log(
                    f"Downloading {num_channels} input channels for case {case['num']} "
                    f"from {_short_host(case_base_url)}..."
                )
                slicer.app.processEvents()
                channel_paths = self._download_case_channels(
                    case_base_url, case, num_channels, out_dir, log
                )
                image_id = f"{case['dataset_id']}_{case['images_for']}_{case['num']}"
            else:
                channel_paths = [self._export_viewer_volume(out_dir, log)]
                if has_case:
                    image_id = f"{case['dataset_id']}_{case['images_for']}_{case['num']}"
                else:
                    image_id = f"viewer_{job_uid}"

            log(f"Submitting prediction to {_short_host(predict_url)}...")
            slicer.app.processEvents()
            submit = nnunet_client.post_prediction(
                BASE_URL=predict_url,
                model_dataset_id=model["dataset_id"],
                image_id=image_id,
                channel_image_paths=channel_paths,
                trainer=model.get("trainer", "nnUNetTrainer"),
                plans=model.get("plans", "nnUNetPlans"),
                configuration=model.get("configuration", "3d_lowres"),
                fold=fold_mode if fold_mode == FOLD_MODE_ALL else None,
            )
        except Exception as e:
            job["state"] = "failed"
            self._set_job_tab_title(job, f"{tab_title} ✕")
            log(f"Submit failed: {e}")
            self._ensure_poll_timer()
            return

        job_id = submit.get("job_id")
        req_id = submit.get("req_id")
        if not job_id or not req_id:
            job["state"] = "failed"
            self._set_job_tab_title(job, f"{tab_title} ✕")
            log(f"Submit failed: unexpected response (missing job_id/req_id): {submit}")
            self._ensure_poll_timer()
            return

        job["job_id"] = job_id
        job["req_id"] = req_id
        job["base_url"] = predict_url
        job["state"] = "queued"
        job["submitted_at"] = time.monotonic()
        ahead = submit.get("number_of_jobs_ahead", "?")
        log(
            f"Job queued on {_short_host(predict_url)}. "
            f"req_id={req_id}, job_id={job_id}, jobs ahead={ahead}"
        )
        self._ensure_poll_timer()

    def _poll_all_jobs(self):
        active = self._active_jobs()
        if not active:
            self._poll_timer.stop()
            return

        for job in list(active):
            if job.get("state") not in ("queued", "running"):
                continue
            if not job.get("job_id") or not job.get("base_url"):
                continue
            try:
                status = nnunet_client.get_prediction_job_status(
                    job["base_url"], job["job_id"]
                )
            except Exception as e:
                self._append_job(job, f"Status check failed: {e}")
                continue

            state = str(status.get("status", "")).lower()
            progress = status.get("progress", "")
            ahead = status.get("number_of_jobs_ahead", "")
            self._append_job(
                job, f"Status: {state}  progress={progress}  ahead={ahead}"
            )

            if state in ("finished", "completed", "success"):
                job["state"] = "importing"
                self._on_job_finished(job)
            elif state in ("failed", "stopped", "canceled", "cancelled"):
                job["state"] = "failed" if state == "failed" else "canceled"
                self._set_job_tab_title(job, f"{job.get('tab_title')} ✕")
                err = status.get("error") or state
                self._append_job(job, f"Prediction ended: {err}")

        self._ensure_poll_timer()

    def _on_job_finished(self, job):
        if not job:
            return
        try:
            self._append_job(job, "Downloading prediction result...")
            slicer.app.processEvents()
            result = nnunet_client.download_prediction_result(
                BASE_URL=job["base_url"],
                dataset_id=job["model_dataset_id"],
                req_id=job["req_id"],
                image_number=0,
                out_dir=job["out_dir"],
            )
            labels_path = result.get("labels_path")
            if not labels_path or not os.path.exists(str(labels_path)):
                raise RuntimeError(
                    f"Prediction finished but label file was not found: {result}"
                )

            self._append_job(job, "Importing result into Slicer…")
            slicer.app.processEvents()
            volume_node = slicer.mrmlScene.GetNodeByID(job.get("volume_node_id") or "")
            if volume_node is None:
                volume_node = self._input_volume()
            if volume_node is None:
                raise RuntimeError("Reference volume is no longer in the scene.")

            seg_node = self.parent.logic.importPredictionLabelmap(
                labels_path=labels_path,
                labels_map=job.get("labels") or {},
                volume_node=volume_node,
                model_name=job.get("model_dataset_id") or "Prediction",
                job_seq=job.get("seq"),
                log_fn=lambda m: self._append_job(job, m),
            )
            # Keep a handle so user can open Segment Editor on it.
            if seg_node is not None:
                job["segmentation_node_id"] = seg_node.GetID()
                self.parent.logic.showInSegmentEditor(volume_node, seg_node)
        except Exception as e:
            job["state"] = "failed"
            self._set_job_tab_title(job, f"{job.get('tab_title')} ✕")
            self._append_job(job, f"Failed to apply result: {e}")
            self._ensure_poll_timer()
            return

        elapsed = time.monotonic() - job.get("submitted_at", time.monotonic())
        elapsed_str = _format_duration(elapsed)
        job["state"] = "finished"
        self._set_job_tab_title(job, f"{job.get('tab_title')} ✓")
        self._append_job(job, f"Done in {elapsed_str}.")
        self._ensure_poll_timer()

    def _on_cancel_clicked(self):
        active = self._active_jobs()
        if not active:
            self._append_general("No active prediction jobs to cancel.")
            return
        reply = qt.QMessageBox.question(
            slicer.util.mainWindow(),
            "Cancel Predictions",
            f"Cancel {len(active)} active prediction job(s)?",
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.No,
        )
        if reply != qt.QMessageBox.Yes:
            return

        self._poll_timer.stop()
        for job in list(active):
            job_id = job.get("job_id")
            base_url = job.get("base_url")
            if job_id and base_url:
                try:
                    result = nnunet_client.cancel_prediction_job(base_url, job_id)
                    self._append_job(
                        job,
                        f"Cancel requested: {result.get('status')} — "
                        f"{result.get('message', '')}",
                    )
                except Exception as e:
                    self._append_job(job, f"Cancel request failed: {e}")
            job["state"] = "canceled"
            self._set_job_tab_title(job, f"{job.get('tab_title')} ✕")

        self._append_general("Canceled active prediction jobs.")
        self._poll_timer.stop()
