"""Schwab API client utilities with encrypted token storage."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
from requests import Response, Session

from .config import SchwabAPIConfig

try:  # pragma: no cover - optional dependency resolved at runtime
    from cryptography.fernet import Fernet
except ImportError as exc:  # pragma: no cover - bubble up a helpful error later
    raise ImportError(
        "The 'cryptography' package is required for Schwab token encryption"
    ) from exc


@dataclass
class TokenBundle:
    """OAuth tokens cached locally."""

    access_token: str
    refresh_token: str
    expires_at: float

    @classmethod
    def from_response(cls, payload: Dict[str, Any]) -> "TokenBundle":
        access = payload.get("access_token") or payload.get("accessToken")
        refresh = payload.get("refresh_token") or payload.get("refreshToken")
        expires_in = payload.get("expires_in") or payload.get("expiresIn") or 0

        if not access or not refresh:
            raise ValueError("Token response missing access or refresh token")

        expires_at = time.time() + float(expires_in)
        return cls(access_token=access, refresh_token=refresh, expires_at=expires_at)

    def is_expired(self, skew_seconds: float = 30.0) -> bool:
        return time.time() >= (self.expires_at - skew_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }


class EncryptedTokenStore:
    """Persist Schwab OAuth tokens encrypted on disk."""

    def __init__(self, path: Path, passphrase: str) -> None:
        if not passphrase:
            raise ValueError("Passphrase required to initialize encrypted token store")
        key = base64.urlsafe_b64encode(hashlib.sha256(passphrase.encode("utf-8")).digest())
        self._fernet = Fernet(key)
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[TokenBundle]:
        if not self._path.exists():
            return None
        data = self._path.read_bytes()
        try:
            decrypted = self._fernet.decrypt(data)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Failed to decrypt Schwab token cache") from exc
        payload = json.loads(decrypted.decode("utf-8"))
        return TokenBundle(
            access_token=payload["access_token"],
            refresh_token=payload["refresh_token"],
            expires_at=float(payload["expires_at"]),
        )

    def save(self, bundle: TokenBundle) -> None:
        serialized = json.dumps(bundle.to_dict()).encode("utf-8")
        encrypted = self._fernet.encrypt(serialized)
        self._path.write_bytes(encrypted)

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()


class SchwabAuthManager:
    """Handle OAuth flows and automatic refresh for Schwab APIs."""

    authorize_endpoint = "https://api.schwabapi.com/v1/oauth/authorize"
    token_endpoint = "https://api.schwabapi.com/v1/oauth/token"

    def __init__(
        self,
        config: SchwabAPIConfig,
        *,
        session: Optional[Session] = None,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()

        oauth_cfg = config.oauth
        passphrase = os.getenv(oauth_cfg.encryption_passphrase_env, "")
        if not passphrase:
            raise RuntimeError(
                f"Environment variable {oauth_cfg.encryption_passphrase_env} must be set"
            )

        self._store = EncryptedTokenStore(Path(oauth_cfg.token_cache).expanduser(), passphrase)
        self._tokens: Optional[TokenBundle] = None

    @property
    def client_id(self) -> str:
        oauth_cfg = self._config.oauth
        if not oauth_cfg.client_id:
            raise RuntimeError("Schwab client ID not configured")
        return oauth_cfg.client_id

    @property
    def client_secret(self) -> str:
        oauth_cfg = self._config.oauth
        if not oauth_cfg.client_secret:
            raise RuntimeError("Schwab client secret not configured")
        return oauth_cfg.client_secret

    @property
    def redirect_uri(self) -> str:
        oauth_cfg = self._config.oauth
        if not oauth_cfg.redirect_uri:
            raise RuntimeError("Schwab redirect URI not configured")
        return oauth_cfg.redirect_uri

    @property
    def scopes(self) -> str:
        return " ".join(self._config.oauth.scopes)

    def authorization_url(self, state: Optional[str] = None) -> str:
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
        }
        if state:
            params["state"] = state
        return f"{self.authorize_endpoint}?{urlencode(params)}"

    def exchange_authorization_code(self, code: str) -> TokenBundle:
        payload = self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            }
        )
        bundle = TokenBundle.from_response(payload)
        self._store.save(bundle)
        self._tokens = bundle
        return bundle

    def refresh_access_token(self) -> TokenBundle:
        bundle = self._tokens or self._store.load()
        if not bundle:
            raise RuntimeError(
                "No Schwab refresh token available. Run the authorization code flow first."
            )
        payload = self._token_request(
            {
                "grant_type": "refresh_token",
                "refresh_token": bundle.refresh_token,
            }
        )
        # Schwab issues a new refresh token with each refresh. Persist it.
        new_bundle = TokenBundle.from_response(payload)
        self._store.save(new_bundle)
        self._tokens = new_bundle
        return new_bundle

    def get_access_token(self) -> str:
        bundle = self._tokens or self._store.load()
        if not bundle:
            raise RuntimeError(
                "Schwab tokens not cached. Complete the initial OAuth handshake first."
            )
        if bundle.is_expired():
            bundle = self.refresh_access_token()
        else:
            self._tokens = bundle
        return bundle.access_token

    def _token_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        auth = (self.client_id, self.client_secret)
        response = self._session.post(self.token_endpoint, data=data, headers=headers, auth=auth)
        self._raise_for_status(response)
        return response.json()

    @staticmethod
    def _raise_for_status(response: Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network errors
            raise RuntimeError(
                f"Schwab OAuth request failed ({response.status_code}): {response.text}"
            ) from exc


class RateLimiter:
    """Token bucket limiting requests per minute."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max(0, max_per_minute)
        self._events: deque[float] = deque()

    def throttle(self) -> None:
        if self._max <= 0:
            return
        now = time.time()
        window_start = now - 60.0
        while self._events and self._events[0] < window_start:
            self._events.popleft()
        if len(self._events) >= self._max:
            sleep_for = 60.0 - (now - self._events[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            return self.throttle()
        self._events.append(time.time())


class SchwabRESTClient:
    """Concrete SchwabClient implementation backed by REST endpoints."""

    def __init__(
        self,
        config: SchwabAPIConfig,
        *,
        session: Optional[Session] = None,
        auth_manager: Optional[SchwabAuthManager] = None,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()
        self._auth = auth_manager or SchwabAuthManager(config, session=self._session)
        self._limiter = RateLimiter(config.rate_limit.max_requests_per_minute)

    # --- SchwabClient protocol methods -------------------------------------------------
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        url = f"{self._config.market_data_base}/{symbol}/quotes"
        response = self._api_request("GET", url)
        if isinstance(response, dict) and "quotes" in response:
            # API may wrap quotes in {"quotes": {"AAPL": {...}}}
            quotes = response["quotes"]
            if isinstance(quotes, dict) and symbol.upper() in quotes:
                return quotes[symbol.upper()]
            return quotes
        return response

    def get_option_chain(self, symbol: str, **params: Any) -> Dict[str, Any]:
        url = f"{self._config.market_data_base}/chains"
        query = {"symbol": symbol, **params}
        return self._api_request("GET", url, params=query)

    # --- Helpers ----------------------------------------------------------------------
    def _api_request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Dict[str, Any]:
        self._limiter.throttle()
        token = self._auth.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        response = self._session.request(
            method,
            url,
            params=params,
            data=data,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 401 and retry:
            # Token likely expired but not marked yet. Refresh and retry once.
            self._auth.refresh_access_token()
            return self._api_request(
                method,
                url,
                params=params,
                data=data,
                retry=False,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network errors
            raise RuntimeError(
                f"Schwab API request failed ({response.status_code}): {response.text}"
            ) from exc

        if response.headers.get("Content-Type", "").startswith("application/json"):
            return response.json()
        raise RuntimeError("Unexpected Schwab API response type; expected JSON")


def main() -> None:  # pragma: no cover - convenience helper
    """Interactive helper to complete the OAuth code exchange."""

    from .config import SchwabAPIConfig

    config = SchwabAPIConfig()
    manager = SchwabAuthManager(config)
    print("Schwab OAuth authorization")
    print("===========================")
    print("1. Visit the following URL in your browser and complete the login flow:")
    print(manager.authorization_url())
    print("2. After consenting, copy the 'code' query parameter from the redirect URL and paste it below.")
    code = input("Authorization code: ").strip()
    bundle = manager.exchange_authorization_code(code)
    print("Tokens stored. Access token expires at:", time.ctime(bundle.expires_at))


if __name__ == "__main__":  # pragma: no cover - manual usage
    main()

