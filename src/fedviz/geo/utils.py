from __future__ import annotations

import requests


def get_location(ip: str | None = None) -> dict:
    """Resolve the current client or a specific IP to location metadata using ipinfo.io."""
    try:
        url = "https://ipinfo.io/json" if not ip else f"https://ipinfo.io/{ip}/json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            location = {
                "ip": data.get("ip", ip or "Unknown"),
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country", "Unknown"),
                "org": data.get("org", "Unknown"),
                "lat": None,
                "lng": None,
            }
            loc = data.get("loc", "")
            if loc:
                lat, lng = loc.split(",")
                location["lat"] = float(lat)
                location["lng"] = float(lng)
            return location
    except Exception as exc:
        target = ip or "current client"
        print(f"[fedviz/geo] Failed to resolve IP for {target}: {exc}")
    return {}


def extract_client_id(obj):
    """Extract ``client_id`` from a request-like object."""
    try:
        return obj.header.client_id
    except (AttributeError, TypeError):
        for attr in ("client_id", "id"):
            value = getattr(obj, attr, None)
            if value:
                return value
    return None
