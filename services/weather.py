"""Client Open-Meteo avec cache SQLite et fallback aux dernières données."""
import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class WeatherService:
    def __init__(self, db_path: Path, ttl_seconds: int = 1800):
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS weather_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at INTEGER NOT NULL)")

    @staticmethod
    def _get_json(url):
        req = Request(url, headers={"User-Agent": "PRAG/1.0 (agricultural decision support)"})
        with urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))

    def _cached(self, key):
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT payload, fetched_at FROM weather_cache WHERE cache_key=?", (key,)).fetchone()
        return (json.loads(row[0]), row[1]) if row else (None, None)

    def _store(self, key, data):
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT OR REPLACE INTO weather_cache VALUES (?, ?, ?)", (key, json.dumps(data), int(time.time())))

    def search_places(self, query):
        params = urlencode({"name": query, "count": 8, "language": "fr", "format": "json", "countryCode": "CI"})
        data = self._get_json("https://geocoding-api.open-meteo.com/v1/search?" + params)
        return [{"name": p["name"], "admin1": p.get("admin1", ""), "country": p.get("country", ""),
                 "latitude": p["latitude"], "longitude": p["longitude"], "timezone": p.get("timezone", "")}
                for p in data.get("results", [])]

    def forecast(self, latitude, longitude):
        lat, lon = round(float(latitude), 3), round(float(longitude), 3)
        key = f"{lat}:{lon}"
        cached, fetched_at = self._cached(key)
        now = int(time.time())
        if cached and now - fetched_at < self.ttl_seconds:
            return {"data": cached, "cached": True, "stale": False, "fetched_at": fetched_at}
        params = urlencode({"latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,precipitation_probability,precipitation,relative_humidity_2m,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,et0_fao_evapotranspiration",
            "forecast_days": 7, "timezone": "auto"})
        try:
            data = self._get_json("https://api.open-meteo.com/v1/forecast?" + params)
            self._store(key, data)
            return {"data": data, "cached": False, "stale": False, "fetched_at": now}
        except Exception:
            if cached:
                return {"data": cached, "cached": True, "stale": True, "fetched_at": fetched_at}
            raise
