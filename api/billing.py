"""
api/billing.py

Stripe Checkout integration for Ethos/Verum.

Products:
  single_report   — one-time, full 15-value report ($2.99)
  single_dataset  — one-time, dataset compilation ($4.99)
  single_cert     — one-time, Verum certification ($24.99)
  pro_monthly     — subscription, Pro tier ($49/month)

Flow:
  1. Frontend calls POST /billing/checkout with product type
  2. Server creates a Stripe Checkout Session
  3. Returns the checkout URL
  4. Frontend redirects user to Stripe-hosted checkout
  5. On success, Stripe redirects to /billing/success?session_id=...
  6. Webhook POST /billing/webhook handles payment confirmation
     and provisions API keys for Pro subscribers

All Stripe keys loaded from environment variables. Never committed to code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_KEYS_PATH = _DATA_DIR / "api_keys.json"
_PURCHASES_PATH = _DATA_DIR / "purchases.json"

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

def _stripe_key() -> str:
    key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail="Payment system not configured.")
    return key


def _webhook_secret() -> str:
    return os.environ.get("STRIPE_WEBHOOK_SECRET", "")


def _site_domain() -> str:
    return os.environ.get("SITE_DOMAIN", "https://trust-forged.com")


# Product config: maps product type to env var for Stripe Price ID
_PRODUCT_MAP = {
    "single_report":  "STRIPE_PRICE_SINGLE_REPORT",
    "single_dataset": "STRIPE_PRICE_SINGLE_DATASET",
    "single_cert":    "STRIPE_PRICE_SINGLE_CERT",
    "pro_monthly":    "STRIPE_PRICE_PRO_MONTHLY",
}

_PRODUCT_LABELS = {
    "single_report":  "Full 15-Value Report",
    "single_dataset": "Dataset Compilation",
    "single_cert":    "Verum Certification",
    "pro_monthly":    "Verum Pro (Monthly)",
}


def _get_price_id(product: str) -> str:
    env_var = _PRODUCT_MAP.get(product)
    if not env_var:
        raise HTTPException(status_code=400, detail=f"Unknown product: {product}")
    price_id = os.environ.get(env_var, "")
    if not price_id or price_id.startswith("price_placeholder"):
        raise HTTPException(status_code=503, detail=f"Product '{product}' not configured in Stripe.")
    return price_id


# ---------------------------------------------------------------------------
# Purchase tracking (lightweight JSON store)
# ---------------------------------------------------------------------------

def _load_purchases() -> dict:
    if not _PURCHASES_PATH.exists():
        _PURCHASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PURCHASES_PATH.write_text(json.dumps({"purchases": []}, indent=2))
        return {"purchases": []}
    try:
        return json.loads(_PURCHASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"purchases": []}


def _save_purchase(purchase: dict) -> None:
    data = _load_purchases()
    data["purchases"].append(purchase)
    _PURCHASES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _generate_api_key() -> str:
    """Generate a unique API key for Pro subscribers."""
    raw = f"verum_pro_{uuid.uuid4().hex}_{int(time.time())}"
    return f"vr_{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


def _add_pro_key(api_key: str, email: str = "") -> None:
    """Add a Pro API key to the keys store."""
    _KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _KEYS_PATH.exists():
        data = json.loads(_KEYS_PATH.read_text(encoding="utf-8"))
    else:
        data = {"pro_keys": []}

    if api_key not in data.get("pro_keys", []):
        data.setdefault("pro_keys", []).append(api_key)

    # Track key metadata separately
    data.setdefault("key_metadata", {})[api_key] = {
        "email": email,
        "created_at": time.time(),
        "status": "active",
    }

    _KEYS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _revoke_pro_key(api_key: str) -> None:
    """Revoke a Pro API key."""
    if not _KEYS_PATH.exists():
        return
    data = json.loads(_KEYS_PATH.read_text(encoding="utf-8"))
    if api_key in data.get("pro_keys", []):
        data["pro_keys"].remove(api_key)
    if api_key in data.get("key_metadata", {}):
        data["key_metadata"][api_key]["status"] = "revoked"
        data["key_metadata"][api_key]["revoked_at"] = time.time()
    _KEYS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkout endpoint
# ---------------------------------------------------------------------------

class CheckoutRequest(BaseModel):
    product: str = Field(..., description="Product type: single_report, single_dataset, single_cert, pro_monthly")
    email: str = Field("", description="Customer email (optional, pre-fills Stripe checkout)")


@router.post("/checkout")
def create_checkout(body: CheckoutRequest):
    """
    Create a Stripe Checkout Session and return the URL.
    Frontend redirects the user to this URL.
    """
    import stripe
    stripe.api_key = _stripe_key()

    price_id = _get_price_id(body.product)
    domain = _site_domain()

    is_subscription = body.product == "pro_monthly"

    session_params = {
        "line_items": [{"price": price_id, "quantity": 1}],
        "mode": "subscription" if is_subscription else "payment",
        "success_url": f"{domain}/verum?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{domain}/verum?checkout=cancelled",
        "metadata": {
            "product": body.product,
            "product_label": _PRODUCT_LABELS.get(body.product, body.product),
        },
    }

    if body.email:
        session_params["customer_email"] = body.email

    # For one-time payments, allow invoicing
    if not is_subscription:
        session_params["invoice_creation"] = {"enabled": True}

    try:
        session = stripe.checkout.Session.create(**session_params)
    except stripe.error.StripeError as e:
        _log.error("Stripe checkout error: %s", e)
        raise HTTPException(status_code=502, detail="Payment provider error. Please try again.")

    return {"checkout_url": session.url, "session_id": session.id}


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.

    Events handled:
      checkout.session.completed  — payment succeeded, provision access
      customer.subscription.deleted — subscription cancelled, revoke access
    """
    import stripe
    stripe.api_key = _stripe_key()

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    webhook_secret = _webhook_secret()

    if webhook_secret:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        except (ValueError, stripe.error.SignatureVerificationError) as e:
            _log.warning("Webhook signature verification failed: %s", e)
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        # No webhook secret configured; parse but log warning
        _log.warning("STRIPE_WEBHOOK_SECRET not set; accepting unverified webhook")
        event = json.loads(payload)

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_cancelled(data)
    else:
        _log.debug("Unhandled Stripe event: %s", event_type)

    return JSONResponse({"status": "ok"})


def _handle_checkout_completed(session: dict) -> None:
    """Process a completed checkout session."""
    metadata = session.get("metadata", {})
    product = metadata.get("product", "")
    email = session.get("customer_email") or session.get("customer_details", {}).get("email", "")
    session_id = session.get("id", "")

    _log.info("Checkout completed: product=%s email=%s session=%s", product, email, session_id)

    if product == "pro_monthly":
        # Generate and store a Pro API key
        api_key = _generate_api_key()
        _add_pro_key(api_key, email)
        _log.info("Pro key provisioned for %s: %s...%s", email, api_key[:8], api_key[-4:])

        _save_purchase({
            "type": "pro_monthly",
            "email": email,
            "api_key": api_key,
            "session_id": session_id,
            "subscription_id": session.get("subscription", ""),
            "timestamp": time.time(),
        })
    else:
        # One-time purchase: generate a single-use token
        token = f"vu_{uuid.uuid4().hex[:16]}"
        _save_purchase({
            "type": product,
            "email": email,
            "token": token,
            "session_id": session_id,
            "timestamp": time.time(),
            "redeemed": False,
        })
        _log.info("Single-use token created for %s: %s (%s)", email, token, product)


def _handle_subscription_cancelled(subscription: dict) -> None:
    """Revoke Pro access when subscription is cancelled."""
    sub_id = subscription.get("id", "")
    _log.info("Subscription cancelled: %s", sub_id)

    # Find and revoke the API key associated with this subscription
    data = _load_purchases()
    for purchase in data.get("purchases", []):
        if purchase.get("subscription_id") == sub_id and purchase.get("api_key"):
            _revoke_pro_key(purchase["api_key"])
            _log.info("Revoked Pro key for cancelled subscription: %s", sub_id)
            break
