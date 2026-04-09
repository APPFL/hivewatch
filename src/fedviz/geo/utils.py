from __future__ import annotations

import requests


def is_local(peer: str) -> bool:
    """Check if peer is local or otherwise non-routable."""
    return not peer or any(token in peer for token in ("127.0.0.1", "localhost", "[::1]"))


def parse_ip(raw_peer: str) -> str:
    """Extract a plain IP from a gRPC peer string like ``ipv4:1.2.3.4:50546``."""
    if not raw_peer:
        return ""
    parts = raw_peer.replace("ipv4:", "").replace("ipv6:", "").split(":")
    return parts[0] if parts else ""


def get_location(ip: str) -> dict:
    """Resolve an IP to location metadata using ipinfo.io."""
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            location = {
                "ip": data.get("ip", ip),
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
        print(f"[fedviz/geo] Failed to resolve IP {ip}: {exc}")
    return {}


def extract_client_id(obj):
    """Extract ``client_id`` from an APPFL request object."""
    try:
        return obj.header.client_id
    except (AttributeError, TypeError):
        for attr in ("client_id", "id"):
            value = getattr(obj, attr, None)
            if value:
                return value
    return None
