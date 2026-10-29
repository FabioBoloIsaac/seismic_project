"""
Utilita' geografiche: distanza great-circle tra due coordinate (formula di
Haversine), usata per filtrare le storie temporali in base alla distanza
tra la stazione che le ha registrate e la localita' di interesse.
"""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanza great-circle (Haversine) in km tra due punti (gradi decimali)."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
