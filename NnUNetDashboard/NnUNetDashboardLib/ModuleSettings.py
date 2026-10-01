"""Persisted settings for the NnUNetDashboard Slicer module.

Stored in Slicer's QSettings (user ``Slicer.ini`` / revision settings), not
the desktop app's ``settings.json``.
"""

from __future__ import annotations

from . import NnUNetClient as nnunet_client

SETTINGS_PREFIX = "NnUNetDashboard"

KEY_SERVER_URLS = f"{SETTINGS_PREFIX}/server_urls"
KEY_SELECTED_SERVER = f"{SETTINGS_PREFIX}/selected_server_url"
KEY_KEYCLOAK_URL = f"{SETTINGS_PREFIX}/keycloak_url"
KEY_KEYCLOAK_REALM = f"{SETTINGS_PREFIX}/keycloak_realm"
KEY_REGISTRATION_URL = f"{SETTINGS_PREFIX}/registration_url"
KEY_LAST_EMAIL = f"{SETTINGS_PREFIX}/last_email"


def _qsettings():
    import qt

    return qt.QSettings()


def _get(key, default=""):
    import slicer

    return slicer.util.settingsValue(key, default)


def _set(key, value):
    settings = _qsettings()
    settings.setValue(key, value)
    settings.sync()


def _urls_to_text(urls):
    return "\n".join(u.strip().rstrip("/") for u in (urls or []) if str(u).strip())


def _text_to_urls(text):
    urls = []
    seen = set()
    for line in str(text or "").splitlines():
        url = line.strip().rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def load_settings():
    """Return a dict of module settings with defaults filled in."""
    default_urls = list(nnunet_client.DEFAULT_SERVER_URLS)
    has_urls_key = _qsettings().contains(KEY_SERVER_URLS)
    urls_text = _get(KEY_SERVER_URLS, _urls_to_text(default_urls))
    urls = _text_to_urls(urls_text)
    # Only fall back to built-in defaults when nothing has been saved yet.
    if not urls and not has_urls_key:
        urls = list(default_urls)

    selected = str(_get(KEY_SELECTED_SERVER, "") or "").strip().rstrip("/")
    # Do NOT re-insert a removed server into the list. Point selection at a
    # remaining URL (or blank) instead.
    if selected and selected not in urls:
        selected = urls[0] if urls else ""
    if not selected:
        selected = urls[0] if urls else ""

    keycloak_url = str(
        _get(KEY_KEYCLOAK_URL, nnunet_client.DEFAULT_KEYCLOAK_URL) or ""
    ).strip().rstrip("/") or nnunet_client.DEFAULT_KEYCLOAK_URL
    keycloak_realm = str(
        _get(KEY_KEYCLOAK_REALM, nnunet_client.DEFAULT_KEYCLOAK_REALM) or ""
    ).strip() or nnunet_client.DEFAULT_KEYCLOAK_REALM

    registration_url = str(_get(KEY_REGISTRATION_URL, "") or "").strip()
    if not registration_url:
        registration_url = nnunet_client.default_registration_url(
            keycloak_url, keycloak_realm
        )

    last_email = str(_get(KEY_LAST_EMAIL, "") or "").strip()

    return {
        "server_urls": urls,
        "selected_server_url": selected,
        "keycloak_url": keycloak_url,
        "keycloak_realm": keycloak_realm,
        "registration_url": registration_url,
        "last_email": last_email,
    }


def save_settings(
    server_urls=None,
    selected_server_url=None,
    keycloak_url=None,
    keycloak_realm=None,
    registration_url=None,
    last_email=None,
):
    """Persist any provided fields (``None`` means leave unchanged)."""
    current = load_settings()

    if server_urls is not None:
        if isinstance(server_urls, str):
            urls = _text_to_urls(server_urls)
        else:
            urls = _text_to_urls(_urls_to_text(server_urls))
        # Persist even an empty list so removals stick; do not silently restore
        # built-in defaults here.
        _set(KEY_SERVER_URLS, _urls_to_text(urls))
        current["server_urls"] = urls
        # Drop selection if it was removed from the list.
        if current.get("selected_server_url") not in urls:
            new_selected = urls[0] if urls else ""
            _set(KEY_SELECTED_SERVER, new_selected)
            current["selected_server_url"] = new_selected

    if selected_server_url is not None:
        selected = str(selected_server_url or "").strip().rstrip("/")
        urls = current.get("server_urls") or []
        if selected and selected not in urls:
            # Only keep selection if it is still in the configured list.
            selected = urls[0] if urls else ""
        _set(KEY_SELECTED_SERVER, selected)
        current["selected_server_url"] = selected

    if keycloak_url is not None:
        value = str(keycloak_url or "").strip().rstrip("/") or nnunet_client.DEFAULT_KEYCLOAK_URL
        _set(KEY_KEYCLOAK_URL, value)
        current["keycloak_url"] = value

    if keycloak_realm is not None:
        value = str(keycloak_realm or "").strip() or nnunet_client.DEFAULT_KEYCLOAK_REALM
        _set(KEY_KEYCLOAK_REALM, value)
        current["keycloak_realm"] = value

    if registration_url is not None:
        # Empty string means "derive from Keycloak URL + realm".
        value = str(registration_url or "").strip()
        _set(KEY_REGISTRATION_URL, value)
        if not value:
            value = nnunet_client.default_registration_url(
                current.get("keycloak_url"), current.get("keycloak_realm")
            )
        current["registration_url"] = value

    if last_email is not None:
        value = str(last_email or "").strip()
        _set(KEY_LAST_EMAIL, value)
        current["last_email"] = value

    return current


def registration_url_for_use(settings=None):
    """Effective Register URL (explicit override or derived from Keycloak)."""
    cfg = settings if isinstance(settings, dict) else load_settings()
    explicit = str(_get(KEY_REGISTRATION_URL, "") or "").strip()
    if explicit:
        return explicit
    return nnunet_client.default_registration_url(
        cfg.get("keycloak_url"), cfg.get("keycloak_realm")
    )
