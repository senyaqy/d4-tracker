"""逐帧检测分数 -> 事件。这是整个工具最需要小心的一段逻辑。

要解决的问题
------------
一次副本完成，横幅会在屏幕上停留 1.3~4.7 秒。3fps 轮询下那就是 4~14 帧连续命中。
**必须只算一次**，不能每帧都记一笔。

做法：上升沿 + 释放间隔
----------------------
* 连续命中达到 ``confirm_frames`` 帧才算"横幅出现"（滤掉单帧噪点）；
* 横幅**消失**的判定要更迟钝：必须连续 ``release_gap`` 秒低于阈值，才认为
  "这次完成的横幅结束了"。否则淡出过程中分数在阈值附近抖动，会被拆成两次事件；
* 再加一道 per-kind 冷却做兜底 —— 真副本不可能在 ``cooldown`` 秒内完成两次。

三者叠加的结果：一次完成 = 恰好一个事件，且对闪烁、淡入淡出不敏感。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Event:
    ts: float
    kind: str
    label: str
    score: float
    frames: int

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))


@dataclass
class _State:
    pending: int = 0            # 连续命中帧数
    active: bool = False        # 横幅"正在显示"
    last_hit: float = 0.0       # 最近一次命中时间
    last_event: float = 0.0     # 最近一次触发事件的时间
    last_score: float = 0.0
    fired_frames: int = 0
    suppressed: int = 0         # 因冷却被压掉的次数（诊断用）
    run_best: float = 0.0       # 本段连续命中的最高分（诊断用）


class EventCounter:
    def __init__(self, threshold: float, confirm_frames: int = 2,
                 release_gap: float = 1.0, cooldown: float = 20.0) -> None:
        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self.release_gap = release_gap
        self.cooldown = cooldown
        self.states: dict[str, _State] = {}
        self.events: list[Event] = []
        # 命中过阈值、但连续帧数没凑够 confirm_frames 的"短信号"。
        # 这是"我明明看到提示了却没给我算"的头号嫌疑：3fps 下 2 帧 = 0.67 秒，
        # 一闪而过的提示就够不上。不记下来的话，事后完全无从判断。
        self.short_runs: list[tuple[str, float, int, float]] = []

    def _state(self, kind: str) -> _State:
        if kind not in self.states:
            self.states[kind] = _State()
        return self.states[kind]

    def update(self, scores: dict[str, float], now: float | None = None) -> list[Event]:
        """喂入"每类模板的当前分数"，返回本次新产生的事件（通常为空）。"""
        now = time.time() if now is None else now
        fired: list[Event] = []

        for kind, score in scores.items():
            st = self._state(kind)
            st.last_score = score

            if score >= self.threshold:
                st.pending += 1
                st.last_hit = now
                st.run_best = max(st.run_best, score)

                ready_to_fire = (not st.active) and st.pending >= self.confirm_frames
                if ready_to_fire:
                    in_cooldown = (st.last_event > 0) and (now - st.last_event) < self.cooldown
                    if in_cooldown:
                        st.suppressed += 1
                    else:
                        st.fired_frames = st.pending
                        ev = Event(ts=now, kind=kind, label=kind, score=score,
                                   frames=st.pending)
                        st.last_event = now
                        self.events.append(ev)
                        fired.append(ev)
                    # 无论是否被冷却压掉，都进入 active，避免同一横幅反复尝试触发
                    st.active = True
            else:
                # 这一段命中结束了。没凑够帧、也没进过 active，就是"短信号"。
                if 0 < st.pending < self.confirm_frames and not st.active:
                    self.short_runs.append((kind, st.run_best, st.pending, now))
                st.pending = 0
                st.run_best = 0.0
                if st.active and (now - st.last_hit) >= self.release_gap:
                    st.active = False

        return fired

    def active_kinds(self) -> list[str]:
        return [k for k, s in self.states.items() if s.active]

    def drain_short_runs(self) -> list[tuple[str, float, int, float]]:
        out = self.short_runs
        self.short_runs = []
        return out

    def report(self) -> str:
        lines = []
        for kind, st in sorted(self.states.items()):
            lines.append(f"{kind}: 事件={sum(1 for e in self.events if e.kind == kind)} "
                         f"冷却压掉={st.suppressed} 末次分数={st.last_score:.3f}")
        return "\n".join(lines)


__all__ = ["EventCounter", "Event"]
