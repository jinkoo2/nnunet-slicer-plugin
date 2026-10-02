"""nnU-Net server HTTP client for the Slicer NnUNet Dashboard module.

Ported from vtk_image_labeler_3d ``nnunet_service.py`` (auth, dataset I/O,
and prediction / approved-model APIs).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time

import requests


class ServerError(Exception):
    """Raised when the nnU-Net server returns an error response."""


# In-memory auth session (password kept only for silent token renew).
_session = {
    "email": None,
    "password": None,
    "base_url": None,
    "token": None,
    "token_exp": None,
    "is_admin": False,
    "roles": [],
}

_TOKEN_REFRESH_SKEW_SECONDS = 60
NNUNET_ADMIN_ROLE = "nnunet-admin"

DEFAULT_SERVER_URLS = [
    "https://nnunet-server-01.apps.myphysics.net/api/v3",
    "https://nnunet-server-02.apps.myphysics.net/api/v3",
]

DEFAULT_KEYCLOAK_URL = "https://login.apps.myphysics.net"
DEFAULT_KEYCLOAK_REALM = "myphysics"


def default_registration_url(keycloak_url: str = "", realm: str = "") -> str:
    """Keycloak Account Console URL (user clicks Register on that page)."""
    base = (keycloak_url or "").rstrip("/") or DEFAULT_KEYCLOAK_URL
    realm = (realm or "").strip() or DEFAULT_KEYCLOAK_REALM
    return f"{base}/realms/{realm}/account/"


DEFAULT_REGISTRATION_URL = default_registration_url()


def _decode_jwt_payload(access_token):
    try:
        parts = (access_token or "").split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(payload + padding)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _roles_from_token(access_token):
    roles = _decode_jwt_payload(access_token).get("realm_access", {}).get("roles", []) or []
    return [str(r) for r in roles]


def _exp_from_token(access_token):
    exp = _decode_jwt_payload(access_token).get("exp")
    try:
        return int(exp) if exp is not None else None
    except (TypeError, ValueError):
        return None


def set_auth_session(
    access_token,
    user_email=None,
    is_admin=False,
    roles=None,
    password=None,
    base_url=None,
):
    token_roles = list(roles) if roles is not None else _roles_from_token(access_token)
    _session["token"] = access_token
    _session["token_exp"] = _exp_from_token(access_token)
    _session["email"] = user_email
    _session["is_admin"] = bool(is_admin) or (NNUNET_ADMIN_ROLE in token_roles)
    _session["roles"] = token_roles
    if password is not None:
        _session["password"] = password
    if base_url is not None:
        _session["base_url"] = (base_url or "").rstrip("/") or None


def clear_auth_session():
    _session.update(
        {
            "token": None,
            "token_exp": None,
            "email": None,
            "password": None,
            "base_url": None,
            "is_admin": False,
            "roles": [],
        }
    )


def get_auth_session():
    data = dict(_session)
    data.pop("password", None)
    return data


def is_authenticated():
    return bool(_session.get("token"))


def _token_needs_renewal(skew_seconds=_TOKEN_REFRESH_SKEW_SECONDS):
    exp = _session.get("token_exp")
    if not _session.get("token") or exp is None:
        return False
    return time.time() >= (exp - skew_seconds)


def can_renew_auth():
    return bool(_session.get("email") and _session.get("password") and _session.get("base_url"))


def renew_auth_session(timeout_seconds=30, force=False):
    if not force and not _token_needs_renewal():
        return True
    if not can_renew_auth():
        return False
    login(_session["base_url"], _session["email"], _session["password"], timeout_seconds)
    return True


def ensure_auth(skew_seconds=_TOKEN_REFRESH_SKEW_SECONDS):
    if not is_authenticated():
        raise ServerError("Not signed in. Connect to the nnU-Net server first.")
    if _token_needs_renewal(skew_seconds):
        if not renew_auth_session(force=True):
            raise ServerError("Session expired and could not be renewed. Sign in again.")


def _auth_headers():
    ensure_auth()
    return {"Authorization": f"Bearer {_session['token']}"}


def _is_token_expired_response(response):
    if response is None:
        return False
    if response.status_code not in (401, 403):
        return False
    text = (response.text or "").lower()
    return any(k in text for k in ("expired", "token", "unauthorized", "not authenticated"))


def request_with_auth(method, url, timeout=30, retry_auth=True, **kwargs):
    headers = kwargs.pop("headers", {}) or {}
    headers.update(_auth_headers())
    response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    if retry_auth and _is_token_expired_response(response) and can_renew_auth():
        renew_auth_session(force=True)
        headers = dict(headers)
        headers.update(_auth_headers())
        # Rewind file handles for multipart retries.
        files = kwargs.get("files")
        if files:
            for item in files.values() if isinstance(files, dict) else files:
                fh = item[1] if isinstance(item, tuple) and len(item) > 1 else item
                if hasattr(fh, "seek"):
                    try:
                        fh.seek(0)
                    except Exception:
                        pass
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    return response


def login(BASE_URL, email, password, timeout_seconds=30):
    base = (BASE_URL or "").rstrip("/")
    url = f"{base}/auth/login"
    try:
        response = requests.post(
            url,
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
    except requests.exceptions.RequestException as e:
        raise ServerError(f"Login request failed: {e}") from e

    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ServerError(f"Login failed ({response.status_code}): {detail}")

    data = response.json()
    token = data.get("access_token")
    if not token:
        raise ServerError(f"Login response missing access_token: {data}")

    set_auth_session(
        access_token=token,
        user_email=data.get("user_email") or email,
        is_admin=bool(data.get("is_admin", False)),
        password=password,
        base_url=base,
    )
    return data


def _filename_from_content_disposition(response, fallback):
    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="([^"]+)"', content_disposition, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"filename=([^;]+)", content_disposition, flags=re.IGNORECASE)
    if match:
        return os.path.basename(match.group(1).strip().strip('"'))
    return fallback


def _raise_for_status(response, action):
    if response.status_code == 200:
        return
    detail = response.text
    try:
        detail = response.json().get("detail", detail)
    except Exception:
        pass
    raise ServerError(f"Failed {action}: {response.status_code}, {detail}")


def get_ping(BASE_URL, timeout_seconds=10):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/status/ping",
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "pinging server")
    return response.json()


def get_dataset_json_list(BASE_URL, timeout_seconds=30):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/datasets/list",
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "listing datasets")
    return response.json()


def get_dataset_image_name_list(BASE_URL, dataset_id, timeout_seconds=30):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/datasets/image_name_list",
        params={"dataset_id": dataset_id},
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "listing cases")
    return response.json()


def download_dataset_image(BASE_URL, dataset_id, images_for, num, out_dir, ch_number=0):
    os.makedirs(out_dir, exist_ok=True)
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/datasets/download_image",
        params={
            "dataset_id": dataset_id,
            "images_for": images_for,
            "num": num,
            "ch_number": ch_number,
        },
        headers=_auth_headers(),
    )
    _raise_for_status(response, "downloading image")
    filename = _filename_from_content_disposition(
        response, f"image_{num}_{int(ch_number):04d}.mha"
    )
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(response.content)
    return {
        "base_image_filename": filename,
        "downloaded_base_image_path": path,
    }


def download_dataset_label(BASE_URL, dataset_id, images_for, num, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/datasets/download_label",
        params={"dataset_id": dataset_id, "images_for": images_for, "num": num},
        headers=_auth_headers(),
    )
    if response.status_code == 404:
        raise ServerError(
            f"Label file not found for dataset_id={dataset_id}, "
            f"images_for={images_for}, num={num}"
        )
    _raise_for_status(response, "downloading label")
    filename = _filename_from_content_disposition(response, f"image_{num}.mha")
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(response.content)
    return {
        "labels_filename": filename,
        "downloaded_labels_image_path": path,
    }


def update_image_and_labels(
    BASE_URL,
    dataset_id,
    images_for,
    num,
    image_path,
    labels_path,
    ch_number=0,
):
    """Upsert image + label for an existing case (PUT, fallback POST on 404)."""
    base = BASE_URL.rstrip("/")
    image_data = {
        "dataset_id": dataset_id,
        "images_for": images_for,
        "num": num,
        "ch_number": ch_number,
    }
    with open(image_path, "rb") as img_file:
        image_response = request_with_auth(
            "PUT",
            f"{base}/datasets/update_image",
            data=image_data,
            files={"image": img_file},
        )
        if image_response.status_code == 404:
            img_file.seek(0)
            image_response = request_with_auth(
                "POST",
                f"{base}/datasets/add_image",
                data=image_data,
                files={"image": img_file},
            )
    _raise_for_status(image_response, "updating image")

    label_data = {
        "dataset_id": dataset_id,
        "images_for": images_for,
        "num": num,
    }
    with open(labels_path, "rb") as lbl_file:
        label_response = request_with_auth(
            "PUT",
            f"{base}/datasets/update_label",
            data=label_data,
            files={"label": lbl_file},
        )
        if label_response.status_code == 404:
            lbl_file.seek(0)
            label_response = request_with_auth(
                "POST",
                f"{base}/datasets/add_label",
                data=label_data,
                files={"label": lbl_file},
            )
    _raise_for_status(label_response, "updating label")
    return {
        "image": image_response.json(),
        "label": label_response.json(),
    }


def labels_from_dataset_json(dataset_json):
    """Return ``{name: int_value}`` from dataset.json, excluding background (<=0)."""
    if not isinstance(dataset_json, dict):
        return {}
    labels = dataset_json.get("labels") or {}
    if not isinstance(labels, dict):
        return {}
    out = {}
    for name, value in labels.items():
        try:
            iv = int(value)
        except (TypeError, ValueError):
            continue
        if iv <= 0:
            continue
        out[str(name)] = iv
    return out


def case_nums_from_image_name_list(name_list, images_for="train"):
    """Return sorted unique case numbers for train or test."""
    if not isinstance(name_list, dict):
        return []
    key = "train_images" if images_for == "train" else "test_images"
    items = name_list.get(key) or []
    nums = set()
    for item in items:
        if isinstance(item, dict) and item.get("num") is not None:
            try:
                nums.add(int(item["num"]))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, str):
            match = re.match(r"^.*?_(\d+)_\d+\.(nii\.gz|mha|mhd)$", item)
            if match:
                nums.add(int(match.group(1)))
    return sorted(nums)


LABEL_STATUS_OPTIONS = [
    "",
    "inprogress",
    "complete",
    "reviewed",
    "labeled",
    "empty",
    "fixed",
]


def label_status_map_from_image_name_list(name_list, images_for="train"):
    """Return ``{num: status_str}`` from bulk ``train_label_status`` / ``test_label_status``."""
    if not isinstance(name_list, dict):
        return {}
    key = "train_label_status" if images_for == "train" else "test_label_status"
    bulk = name_list.get(key) or {}
    if not isinstance(bulk, dict):
        return {}
    out = {}
    for num_key, entry in bulk.items():
        try:
            num = int(num_key)
        except (TypeError, ValueError):
            continue
        if isinstance(entry, dict):
            out[num] = str(entry.get("status") or "")
        else:
            out[num] = str(entry or "")
    return out


def get_label_meta(BASE_URL, dataset_id, images_for, num, timeout_seconds=30):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/datasets/get_label_meta",
        params={"dataset_id": dataset_id, "images_for": images_for, "num": num},
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "fetching label meta")
    return response.json()


def update_label_meta(BASE_URL, dataset_id, images_for, num, meta, timeout_seconds=30):
    response = request_with_auth(
        "PUT",
        f"{BASE_URL.rstrip('/')}/datasets/update_label_meta",
        params={"dataset_id": dataset_id, "images_for": images_for, "num": num},
        json=meta,
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "updating label meta")
    return response.json()


def set_label_status(BASE_URL, dataset_id, images_for, num, status):
    """Merge ``status`` into existing label meta and PUT (full-replace semantics)."""
    response = get_label_meta(BASE_URL, dataset_id, images_for, num)
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    meta = dict(meta)
    new_status = (status or "").strip()
    if new_status:
        meta["status"] = new_status
    else:
        meta.pop("status", None)
    return update_label_meta(BASE_URL, dataset_id, images_for, num, meta)


# ---------------------------------------------------------------------------
# Prediction / approved models
# ---------------------------------------------------------------------------


def get_approved_models(BASE_URL, timeout_seconds=30):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/models/list/approved",
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "fetching approved models")
    data = response.json()
    return data if isinstance(data, list) else []


def get_model_detail(
    BASE_URL, dataset_id, trainer, plans, configuration, timeout_seconds=30
):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/models/model_detail",
        params={
            "dataset_id": dataset_id,
            "trainer": trainer,
            "plans": plans,
            "configuration": configuration,
        },
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "fetching model detail")
    return response.json()


def post_prediction(
    BASE_URL,
    model_dataset_id,
    image_id,
    channel_image_paths,
    trainer="nnUNetTrainer",
    plans="nnUNetPlans",
    configuration="3d_lowres",
    fold=None,
    timeout_seconds=120,
):
    """POST /predictions/predict.

    fold: None/``"ensemble"`` omits the field (5-fold CV default).
    ``"all"`` / ``"fold_all"`` sends ``fold=all`` (single fold_all model).
    """
    if not channel_image_paths:
        raise ValueError("At least one channel image path is required.")

    fold_value = None
    if fold is not None and str(fold).strip() != "":
        fold_norm = str(fold).strip().lower()
        if fold_norm in ("ensemble",):
            fold_value = None
        elif fold_norm in ("all", "fold_all"):
            fold_value = "all"
        else:
            raise ValueError(
                f'Invalid fold={fold!r}; expected None/"ensemble" or "all".'
            )

    form_data = {
        "dataset_id": model_dataset_id,
        "image_id": image_id,
        "trainer": trainer,
        "plans": plans,
        "configuration": configuration,
        "num_channels": str(len(channel_image_paths)),
    }
    if fold_value is not None:
        form_data["fold"] = fold_value

    opened = []
    try:
        files = []
        for i, path in enumerate(channel_image_paths):
            basename = os.path.basename(path)
            if i == 0:
                fh_image = open(path, "rb")
                opened.append(fh_image)
                files.append(("image", (basename, fh_image, "application/octet-stream")))
            fh = open(path, "rb")
            opened.append(fh)
            files.append((f"channel_{i}", (basename, fh, "application/octet-stream")))
        response = requests.post(
            f"{BASE_URL.rstrip('/')}/predictions/predict",
            data=form_data,
            files=files,
            headers=_auth_headers(),
            timeout=timeout_seconds,
        )
        _raise_for_status(response, "posting prediction")
        return response.json()
    finally:
        for fh in opened:
            try:
                fh.close()
            except Exception:
                pass


def get_prediction_job_status(BASE_URL, job_id, timeout_seconds=30):
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/predictions/status/{job_id}",
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "fetching prediction job status")
    return response.json()


def get_prediction_queue_load(
    BASE_URL, dataset_id=None, configuration=None, timeout_seconds=15
):
    params = {}
    if dataset_id:
        params["dataset_id"] = dataset_id
    if configuration:
        params["configuration"] = configuration
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/predictions/load",
        params=params or None,
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "fetching prediction queue load")
    data = response.json()
    return data if isinstance(data, dict) else {}


def cancel_prediction_job(BASE_URL, job_id, timeout_seconds=30):
    response = requests.post(
        f"{BASE_URL.rstrip('/')}/predictions/cancel/{job_id}",
        headers=_auth_headers(),
        timeout=timeout_seconds,
    )
    _raise_for_status(response, "canceling prediction job")
    data = response.json()
    return data if isinstance(data, dict) else {"job_id": job_id, "status": "canceled"}


def find_approved_model_entry(BASE_URL, model, timeout_seconds=30):
    """Return the matching approved-model dict from this server, or None."""
    if not isinstance(model, dict):
        return None
    models = get_approved_models(BASE_URL, timeout_seconds=timeout_seconds) or []
    want = (
        model.get("dataset_id"),
        model.get("trainer"),
        model.get("plans"),
        model.get("configuration"),
    )
    for entry in models:
        if not isinstance(entry, dict):
            continue
        have = (
            entry.get("dataset_id"),
            entry.get("trainer"),
            entry.get("plans"),
            entry.get("configuration"),
        )
        if have == want:
            return entry
    return None


def server_has_approved_model(
    BASE_URL, model, timeout_seconds=30, require_fold_all=False
):
    """True if ``model`` is on this server's approved list.

    When ``require_fold_all`` is True, the entry must also report
    ``fold_all_available`` (Fast / single-model inference).
    """
    entry = find_approved_model_entry(
        BASE_URL, model, timeout_seconds=timeout_seconds
    )
    if entry is None:
        return False
    if require_fold_all and not bool(entry.get("fold_all_available")):
        return False
    return True


def download_prediction_result(BASE_URL, dataset_id, req_id, image_number, out_dir):
    """Download prediction ZIP and extract; return path to label file."""
    import zipfile

    os.makedirs(out_dir, exist_ok=True)
    meta_response = requests.get(
        f"{BASE_URL.rstrip('/')}/predictions/image_and_label_metadata",
        params={
            "dataset_id": dataset_id,
            "req_id": req_id,
            "image_number": image_number,
        },
        headers=_auth_headers(),
    )
    _raise_for_status(meta_response, "fetching prediction metadata")
    metadata = meta_response.json()
    label_name = metadata.get("label_name") or ""
    download_url = f"{BASE_URL.rstrip('/')}{metadata.get('download_url')}"

    zip_path = os.path.join(out_dir, f"{req_id}_image_{image_number}.zip")
    zip_response = requests.get(download_url, headers=_auth_headers())
    if zip_response.status_code != 200:
        raise ServerError(
            f"Failed to download prediction ZIP: {zip_response.status_code}"
        )
    with open(zip_path, "wb") as f:
        f.write(zip_response.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    labels_path = os.path.join(out_dir, label_name) if label_name else ""
    if not labels_path or not os.path.exists(labels_path):
        for name in os.listdir(out_dir):
            lower = name.lower()
            if lower.endswith((".mha", ".mhd", ".nii", ".nii.gz")) and (
                "label" in lower or "seg" in lower or "pred" in lower
            ):
                labels_path = os.path.join(out_dir, name)
                break
    if not labels_path or not os.path.exists(labels_path):
        raise ServerError("Prediction ZIP downloaded but label file was not found.")

    return {
        "label_name": label_name,
        "zip_path": zip_path,
        "labels_path": labels_path,
        "metadata": metadata,
    }


def channel_names_from_dataset_json(dataset_json):
    if not isinstance(dataset_json, dict):
        return []
    channel_names = dataset_json.get("channel_names") or dataset_json.get("modality") or {}
    if isinstance(channel_names, dict) and channel_names:
        def _sort_key(key):
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        return [str(channel_names[k]) for k in sorted(channel_names.keys(), key=_sort_key)]
    if isinstance(channel_names, (list, tuple)) and channel_names:
        return [str(name) for name in channel_names]
    return []


def channel_count_from_dataset_json(dataset_json):
    names = channel_names_from_dataset_json(dataset_json)
    return max(1, len(names)) if names else 1


def format_channel_names(channel_names):
    names = [str(n).strip() for n in (channel_names or []) if str(n).strip()]
    if not names:
        return "-"
    return "[" + ", ".join(names) + "]"


def model_display_name(model):
    if not isinstance(model, dict):
        return str(model)
    dataset = (
        model.get("dataset_id")
        or model.get("dataset")
        or model.get("name")
        or model.get("model_name")
        or "?"
    )
    config = model.get("configuration") or model.get("config") or "?"
    trainer = model.get("trainer") or "?"
    return f"{dataset} | {config} | {trainer}"


def model_search_text(model):
    def _collect(obj, depth=0):
        if depth > 5:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _collect(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _collect(v, depth + 1)

    return " ".join(_collect(model or {}))

