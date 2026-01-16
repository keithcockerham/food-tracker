from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Blueprint, request, jsonify

try:
    import pyarrow.parquet as pq
except Exception:  
    pq = None

bp = Blueprint("reports_api", __name__)

METRIC_MAP = {
    "calories": "calories_kcal",
    "protein": "protein_g",
    "carbs": "carbs_g",
    "fat": "fat_g",
    "fiber": "fiber_g",
    "sugar": "sugar_g",
    "sodium": "sodium_mg",  
}

# PARQUET_PATH = Path("calendar.parquet")
PARQUET_PATH = Path("data") / "calendar.parquet"


@dataclass
class _Cache:
    mtime_ns: int = -1
    rows: List[Dict[str, Any]] = None

CACHE = _Cache(rows=[])


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _parse_timestamp(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None
    return None


def _load_rows() -> List[Dict[str, Any]]:
    p = PARQUET_PATH
    if not p.exists():
        return []

    st = p.stat()
    if st.st_mtime_ns == CACHE.mtime_ns and CACHE.rows:
        return CACHE.rows

    if pq is None:
        import pandas as pd
        df = pd.read_parquet(p)
        py = df.to_dict(orient="records")
    else:
        table = pq.read_table(str(p))
        py = table.to_pylist()

    out: List[Dict[str, Any]] = []
    for r in py:
        eaten_at = _parse_timestamp(r.get("timestamp"))

        if eaten_at is None:
            ds = r.get("date")
            if isinstance(ds, str):
                try:
                    eaten_at = datetime.strptime(ds, "%Y-%m-%d")
                except Exception:
                    continue
            else:
                continue

        metrics: Dict[str, float] = {}
        for src_col, dst_key in METRIC_MAP.items():
            v = r.get(src_col)
            if v is None:
                continue
            try:
                metrics[dst_key] = float(v)
            except Exception:
                pass

        out.append({
            "eaten_at": eaten_at,
            "metrics": metrics,
            "source_type": r.get("source_type"),
            "source_id": r.get("source_id"),
            "qty": r.get("qty"),
            "unit": r.get("unit"),
            "entry_id": r.get("entry_id"),
        })

    CACHE.mtime_ns = st.st_mtime_ns
    CACHE.rows = out
    return out


@bp.get("/api/report")
def api_report():
    start_s = request.args.get("start", "").strip()
    end_s = request.args.get("end", "").strip()

    end = _parse_date(end_s) or date.today()
    start = _parse_date(start_s)

    rows = _load_rows()

    if start is None:
        if rows:
            start = min(r["eaten_at"].date() for r in rows)
        else:
            start = end - timedelta(days=365)

    filt = [r for r in rows if start <= r["eaten_at"].date() <= end]

    totals: Dict[str, float] = {}
    daily_map: Dict[date, Dict[str, float]] = {d: {} for d in _daterange(start, end)}

    HOURLY_METRICS = ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]

    hourly_sum: Dict[str, list] = {k: [0.0] * 24 for k in HOURLY_METRICS}

    ndays = (end - start).days + 1
    by_source_type: Dict[str, int] = {}

    for r in filt:
        eaten_at: datetime = r["eaten_at"]
        d = eaten_at.date()
        m = r.get("metrics") or {}

        st = r.get("source_type") or "unknown"
        by_source_type[st] = by_source_type.get(st, 0) + 1

        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + float(v)
            daily_map[d][k] = daily_map[d].get(k, 0.0) + float(v)

     
        hr = eaten_at.hour
        for k in HOURLY_METRICS:
            v = m.get(k)
            if v is not None:
                hourly_sum[k][hr] += float(v)

    daily = [{"date": d.isoformat(), "metrics": daily_map[d]} for d in sorted(daily_map.keys())]

    hourly_avg = {k: [v / ndays for v in arr] for k, arr in hourly_sum.items()}

    return jsonify({
        "start": start.isoformat(),
        "end": end.isoformat(),
        "entries_count": len(filt),
        "totals": totals,
        "daily": daily,
        "hourly_avg": hourly_avg,
        "meta": {
            "source_type_counts": by_source_type,
            "metric_map": METRIC_MAP,
            "parquet_path": str(PARQUET_PATH),
        }
    })

