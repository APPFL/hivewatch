"""Utility functions for fedviz geo-location and client IP resolution."""

import requests


def is_local(peer: str) -> bool:
    """Check if peer is local/non-routable."""
    return not peer or any(x in peer for x in ("127.0.0.1", "localhost", "[::1]"))


def parse_ip(raw_peer: str) -> str:
    """Extract plain IP from gRPC peer string like 'ipv4:54.183.207.31:50546'."""
    if not raw_peer:
        return ""
    parts = raw_peer.replace("ipv4:", "").replace("ipv6:", "").split(":")
    return parts[0] if parts else ""


def get_location(ip: str) -> dict:
    """Resolve IP to location using ipinfo.io (free, no API key needed)."""
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
    except Exception as e:
        print(f"[fedviz/geo] Failed to resolve IP {ip}: {e}")
    return {}


def extract_client_id(obj):
    """Extract client_id from request object."""
    try:
        return obj.header.client_id
    except (AttributeError, TypeError):
        for attr in ("client_id", "id"):
            val = getattr(obj, attr, None)
            if val:
                return val
    return None
