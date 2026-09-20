"""计数存储（SQLite）。

一条事件一行，不做"计数器 +1"式的就地累加 —— 计数是查出来的。这样：
* 记错了可以删掉某一行，不会留下一笔说不清的账；
* 能回答"今天几点打的""这周梦魇多少把"这类问题；
* 手工补记和自动识别走同一张表，只是 ``source`` 不同。
"""

from __future__ import annotations

import os
import sqlite3
import time

from . import kinds

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL    NOT NULL,
    day     TEXT    NOT NULL,
    kind    TEXT    NOT NULL,
    label   TEXT    NOT NULL,
    score   REAL    NOT NULL DEFAULT 0,
    frames  INTEGER NOT NULL DEFAULT 0,
    source  TEXT    NOT NULL DEFAULT 'auto'
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_day  ON events(day);

-- 数值型读数（地狱狂潮的余烬/灾祸之心）：只在数值变化时追加一行，
-- 于是这张表本身就是一条变化时间线，峰值和"累计获得"都能从它算出来。
CREATE TABLE IF NOT EXISTS readings (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL    NOT NULL,
    day     TEXT    NOT NULL,
    key     TEXT    NOT NULL,
    value   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readings_key ON readings(key);
CREATE INDEX IF NOT EXISTS idx_readings_day ON readings(day, key);

-- 自定义条目：只手动记录，不参与自动识别。用 kind='custom' + label=用户起的名字，
-- 这样事件表本身就能回答"今天这个条目几次"，不需要另开一张计数表。
CREATE TABLE IF NOT EXISTS custom_items (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT    NOT NULL UNIQUE,
    created REAL    NOT NULL
);
"""


class Store:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------------------------------------------------------- 写

    def record(self, kind: str, score: float = 0.0, frames: int = 0,
               ts: float | None = None, source: str = "auto",
               label: str | None = None) -> int:
        ts = time.time() if ts is None else ts
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        label = label or kinds.label_of(kind)
        cur = self.conn.execute(
            "INSERT INTO events (ts, day, kind, label, score, frames, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, day, kind, label, float(score), int(frames), source))
        self.conn.commit()
        return int(cur.lastrowid)

    def remove_last(self, kind: str, label: str | None = None) -> bool:
        """删掉该条目最新的一条记录 —— 这就是界面上的 −1。

        不做"负数计数"：计数是数出来的，往回退就该少一条记录，
        这样任何一行的数都能追到具体时刻，不会留下说不清的账。
        """
        if label is None:
            row = self.conn.execute(
                "SELECT id FROM events WHERE kind = ? ORDER BY id DESC LIMIT 1",
                (kind,)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT id FROM events WHERE kind = ? AND label = ? ORDER BY id DESC LIMIT 1",
                (kind, label)).fetchone()
        return self.delete(row["id"]) if row else False

    def set_count(self, kind: str, label: str | None, target: int,
                  day: str | None = None) -> int:
        """把某条目在指定日期（默认今天）的计数直接设成 target。

        手动"自己填进去"用：多了就删最新的几条，少了就补几条。
        """
        target = max(0, int(target))
        d = day or time.strftime("%Y-%m-%d")
        if label is None:
            rows = self.conn.execute(
                "SELECT id FROM events WHERE kind = ? AND day = ? ORDER BY id",
                (kind, d)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id FROM events WHERE kind = ? AND label = ? AND day = ? ORDER BY id",
                (kind, label, d)).fetchall()

        current = len(rows)
        if target > current:
            for _ in range(target - current):
                self.record(kind, source="manual", label=label)
        elif target < current:
            for r in rows[target:]:          # rows 是升序，砍掉最新的
                self.delete(r["id"])
        return target

    def delete(self, event_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def undo_last(self) -> bool:
        row = self.conn.execute("SELECT id FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return self.delete(row["id"]) if row else False

    # ---------------------------------------------------------------- 读

    def counts_by_kind(self, day: str | None = None) -> dict[str, int]:
        """默认返回今天；day=None 表示今天，day='all' 表示全部。"""
        if day == "all":
            rows = self.conn.execute(
                "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind").fetchall()
        else:
            d = day or time.strftime("%Y-%m-%d")
            rows = self.conn.execute(
                "SELECT kind, COUNT(*) AS n FROM events WHERE day = ? GROUP BY kind",
                (d,)).fetchall()
        out = {k: 0 for k in kinds.all_kinds()}
        for r in rows:
            out[r["kind"]] = r["n"]
        return out

    def total(self, day: str | None = None) -> int:
        return sum(self.counts_by_kind(day).values())

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def days(self, limit: int = 14) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT day, COUNT(*) AS n FROM events GROUP BY day ORDER BY day DESC LIMIT ?",
            (limit,)).fetchall()
        return [(r["day"], r["n"]) for r in rows]

    # ---------------------------------------------------------------- 自定义条目

    def add_custom(self, name: str) -> bool:
        name = (name or "").strip()
        if not name:
            return False
        try:
            self.conn.execute(
                "INSERT INTO custom_items (name, created) VALUES (?, ?)",
                (name, time.time()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False                       # 重名

    def rename_custom(self, old: str, new: str) -> bool:
        new = (new or "").strip()
        if not new or new == old:
            return False
        try:
            self.conn.execute("UPDATE custom_items SET name = ? WHERE name = ?", (new, old))
            # 历史记录也要跟着改，否则改名之后旧计数会"消失"
            self.conn.execute(
                "UPDATE events SET label = ? WHERE kind = ? AND label = ?",
                (new, kinds.CUSTOM_KIND, old))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_custom(self, name: str) -> bool:
        cur = self.conn.execute("DELETE FROM custom_items WHERE name = ?", (name,))
        self.conn.execute("DELETE FROM events WHERE kind = ? AND label = ?",
                          (kinds.CUSTOM_KIND, name))
        self.conn.commit()
        return cur.rowcount > 0

    def custom_items(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM custom_items ORDER BY id").fetchall()
        return [r["name"] for r in rows]

    def custom_counts(self, day: str | None = None) -> dict[str, int]:
        if day == "all":
            rows = self.conn.execute(
                "SELECT label, COUNT(*) AS n FROM events WHERE kind = ? GROUP BY label",
                (kinds.CUSTOM_KIND,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT label, COUNT(*) AS n FROM events WHERE kind = ? AND day = ?"
                " GROUP BY label", (kinds.CUSTOM_KIND, day or time.strftime("%Y-%m-%d"))).fetchall()
        counts = {name: 0 for name in self.custom_items()}
        for r in rows:
            counts[r["label"]] = r["n"]
        return counts

    # ---------------------------------------------------------------- 数值读数

    def record_reading(self, key: str, value: int, ts: float | None = None) -> int:
        ts = time.time() if ts is None else ts
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        cur = self.conn.execute(
            "INSERT INTO readings (ts, day, key, value) VALUES (?, ?, ?, ?)",
            (ts, day, key, int(value)))
        self.conn.commit()
        return int(cur.lastrowid)

    def readings_summary(self, key: str, day: str | None = None) -> dict:
        """返回该数值的：最新值、当日峰值、当日累计获得（只累加增量）。

        累计获得的算法要小心：余烬会被消耗（去祭坛换取），所以"峰值"不等于"获得量"。
        这里对相邻两次读数取正增量求和，消耗带来的下降不计入。
        """
        d = day if day is not None else time.strftime("%Y-%m-%d")
        rows = self.conn.execute(
            "SELECT ts, value, day FROM readings WHERE key = ? ORDER BY ts", (key,)).fetchall()
        if not rows:
            return {"latest": None, "latest_ts": None, "max": None, "gained": 0, "samples": 0}

        latest = rows[-1]["value"]
        latest_ts = rows[-1]["ts"]

        day_rows = [r for r in rows if r["day"] == d]
        if not day_rows:
            return {"latest": latest, "latest_ts": latest_ts, "max": None,
                    "gained": 0, "samples": 0}

        peak = max(r["value"] for r in day_rows)
        gained = 0
        prev = None
        for r in day_rows:
            if prev is not None and r["value"] > prev:
                gained += r["value"] - prev
            prev = r["value"]

        return {"latest": latest, "latest_ts": latest_ts, "max": peak,
                "gained": gained, "samples": len(day_rows)}

    def reset_readings(self, key: str | None = None) -> int:
        if key:
            cur = self.conn.execute("DELETE FROM readings WHERE key = ?", (key,))
        else:
            cur = self.conn.execute("DELETE FROM readings")
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass


__all__ = ["Store"]
