#!/usr/bin/env python3
"""Play a visible Chrome Dino game using screenshots and keyboard events only."""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
from mss.exception import ScreenShotError
from pynput import keyboard


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Obstacle:
    id: int
    box: Box
    kind: str


class StopSwitch:
    def __init__(self) -> None:
        self.stopped = False
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.esc or getattr(key, "char", None) == "q":
            self.stopped = True

    def __enter__(self) -> "StopSwitch":
        self.listener.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.listener.stop()


def parse_region(value: str) -> dict[str, int]:
    try:
        left, top, width, height = map(int, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("需要 x,y,width,height") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width 和 height 必须为正数")
    return dict(left=left, top=top, width=width, height=height)


def foreground(frame: np.ndarray, contrast: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY) if frame.shape[2] == 4 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    background = int(np.median(gray[:max(4, h // 3), w // 4:3 * w // 4]))
    return cv2.inRange(cv2.absdiff(gray, np.full_like(gray, background)), contrast, 255)


def ground_row(mask: np.ndarray) -> int | None:
    h, w = mask.shape
    start, end = int(h * .35), int(h * .97)
    counts = np.count_nonzero(mask[start:end], axis=1)
    strong = np.flatnonzero(counts >= max(60, int(w * .22)))
    return start + int(strong[-1]) if strong.size else None


def auto_region(frame: np.ndarray, monitor: dict[str, int], contrast: int) -> dict[str, int] | None:
    mask = foreground(frame, contrast)
    h, w = mask.shape
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 21), np.uint8))
    best = None
    for y in range(int(h * .30), int(h * .90)):
        edges = np.flatnonzero(np.diff(np.r_[False, closed[y] != 0, False]))
        for left, right in zip(edges[::2], edges[1::2]):
            length = int(right - left)
            if length >= max(300, w // 8):
                candidate = (length, y, int(left))
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        return None
    width, ground, left = best
    # In the official chrome://dino arcade view the dinosaur can cover the
    # beginning of an otherwise screen-wide ground line. Keep that left edge.
    if width >= w * .50:
        left, width = 0, w
    else:
        left, width = max(0, left - 80), min(w - max(0, left - 80), width + 160)
    top, bottom = max(0, ground - 300), min(h, ground + 12)
    return dict(left=monitor['left'] + left, top=monitor['top'] + top, width=width, height=bottom - top)


def boxes(mask: np.ndarray, top: int, bottom: int, left: int, right: int) -> list[Box]:
    crop = cv2.morphologyEx(mask[top:bottom, left:right], cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [Box(left + x, top + y, w, h) for c in contours for x, y, w, h in [cv2.boundingRect(c)]]


def detect(frame: np.ndarray, ground: int, contrast: int, hint: Box | None = None) -> tuple[Box | None, list[Box]]:
    mask = foreground(frame, contrast)
    h, w = mask.shape
    bottom = min(h, ground - 2)
    if bottom <= 25:
        return None, []
    dino_boxes = boxes(mask, max(0, ground - 260), bottom, 0, min(w, max(160, int(w * .3))))
    dino_boxes = [b for b in dino_boxes if 14 <= b.w <= 200 and 17 <= b.h <= 155]
    if hint is not None:
        dino_boxes = [b for b in dino_boxes if abs(b.x - hint.x) <= max(25, hint.w)]
    dino = max(dino_boxes, key=lambda b: b.w * b.h, default=None)
    scale = max(.7, (hint.h if hint else (dino.h if dino else 41)) / 41)
    start = max(0, (dino.right if dino else (hint.right if hint else 0)) + int(10 * scale))
    candidates = boxes(mask, max(0, ground - 240), bottom, start, w)
    obstacles = [b for b in candidates if 10 * scale <= b.w <= 250 and 18 * scale <= b.h <= 150 and b.bottom >= ground - 210]
    return dino, sorted(obstacles, key=lambda b: b.x)


def is_game_over(frame: np.ndarray, ground: int, contrast: int, scale: float) -> bool:
    """Recognize the centered GAME OVER lettering from screen pixels."""
    mask = foreground(frame, contrast)
    height, width = mask.shape
    top = max(0, ground - int(120 * scale))
    bottom = min(height, ground - int(80 * scale))
    left, right = int(width * .30), int(width * .70)
    if bottom <= top:
        return False
    letters = [b for b in boxes(mask, top, bottom, left, right)
               if 5 * scale <= b.w <= 22 * scale
               and 8 * scale <= b.h <= 22 * scale]
    return len(letters) >= 6


def kind(box: Box, ground: int, scale: float = 1.0) -> str:
    gap = ground - box.bottom
    return 'bird' if gap > 11 * scale or (box.w >= 1.5 * box.h and gap > 3 * scale) else 'cactus'


class Tracker:
    def __init__(self, default_speed: float, max_speed: float, scale: float = 1.0) -> None:
        self.default_speed = default_speed
        self.max_speed = max_speed
        self.scale = scale
        self.samples: deque[float] = deque(maxlen=9)
        self.previous: list[Obstacle] = []
        self.previous_time: float | None = None
        self.last_motion: float | None = None
        self.next_id = 1

    @property
    def speed(self) -> float:
        return float(np.median(self.samples)) if self.samples else self.default_speed

    def reset(self) -> None:
        self.samples.clear()
        self.previous.clear()
        self.previous_time = None
        self.last_motion = None

    def update(self, found: list[Box], ground: int, now: float) -> list[Obstacle]:
        dt = now - self.previous_time if self.previous_time is not None else 0
        available = self.previous.copy()
        result = []
        for box in found:
            label = kind(box, ground, self.scale)
            match = None
            if 0 < dt < .4:
                predicted = self.speed * dt
                candidates = [item for item in available if item.kind == label
                              and -2 <= item.box.x - box.x <= self.max_speed * dt + 10
                              and abs(item.box.x - box.x - predicted) <= max(20, predicted * .7)
                              and abs(item.box.y - box.y) <= 15
                              and abs(item.box.w - box.w) <= max(12, box.w // 3)]
                if candidates:
                    match = min(candidates, key=lambda item: abs(item.box.x - box.x - predicted))
                    available.remove(match)
                    measured = (match.box.x - box.x) / dt
                    if 20 <= measured <= self.max_speed:
                        self.samples.append(measured)
                        self.last_motion = now
            if match:
                identifier = match.id
            else:
                identifier = self.next_id
                self.next_id += 1
            result.append(Obstacle(identifier, box, label))
        self.previous, self.previous_time = result, now
        return result


def required_action(item: Obstacle, ground: int, standing_height: int, scale: float = 1.0) -> str:
    if item.kind == 'cactus':
        return 'jump'
    if item.box.bottom <= ground - standing_height + 2 * scale:
        return 'none'
    # A pterodactyl's wings flap across a large vertical range. Its collision
    # body sits higher; duck unless the visible sprite is very close to ground.
    if item.box.bottom <= ground - max(15 * scale, standing_height * .30):
        return 'duck'
    return 'jump'


def annotate(frame: np.ndarray, ground: int, dino: Box | None, obstacles: list[Obstacle], threat: Obstacle | None, speed: float, action: str) -> np.ndarray:
    output = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame.copy()
    cv2.line(output, (0, ground), (output.shape[1] - 1, ground), (255, 0, 0), 1)
    if dino:
        cv2.rectangle(output, (dino.x, dino.y), (dino.right, dino.bottom), (0, 200, 0), 2)
    for item in obstacles:
        b = item.box
        color = (0, 0, 255) if item == threat else (0, 200, 200)
        cv2.rectangle(output, (b.x, b.y), (b.right, b.bottom), color, 2)
        cv2.putText(output, f'{item.kind}:{item.id}', (b.x, max(12, b.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
    cv2.putText(output, f'speed={speed:.0f} action={action}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    return output


def run(monitor: dict[str, int], region: dict[str, int] | None, args: argparse.Namespace) -> None:
    pyautogui.PAUSE = 0
    acted: set[int] = set()
    last_jump, last_debug = -10.0, 0.0
    jumps = ducks = restarts = 0
    duck_id: int | None = None
    duck_until = 0.0
    previous_motion_mask: np.ndarray | None = None
    last_visual_motion = time.monotonic()
    with mss.MSS() as capture, StopSwitch() as stop:
        print(f'{args.start_delay:g} 秒后开始；按 q、Esc 或 Ctrl-C 停止')
        time.sleep(args.start_delay)
        pyautogui.press('space')
        # The official offline page has only a short line under the idle Dino.
        # Starting it switches to a full-width game, so calibrate afterwards.
        deadline = time.monotonic() + 3.0
        ground = None
        hint = None
        while time.monotonic() < deadline and not stop.stopped:
            full = np.asarray(capture.grab(monitor)) if region is None else None
            candidate_region = region or auto_region(full, monitor, args.contrast)
            if candidate_region is None:
                time.sleep(.04)
                continue
            initial = np.asarray(capture.grab(candidate_region)) if region is not None else full[
                candidate_region['top'] - monitor['top']:candidate_region['top'] - monitor['top'] + candidate_region['height'],
                candidate_region['left'] - monitor['left']:candidate_region['left'] - monitor['left'] + candidate_region['width'],
            ]
            candidate_ground = ground_row(foreground(initial, args.contrast))
            if candidate_ground is None:
                time.sleep(.04)
                continue
            candidate_dino, _ = detect(initial, candidate_ground, args.contrast)
            if candidate_dino is not None and candidate_dino.bottom >= candidate_ground - 14:
                region, ground, hint = candidate_region, candidate_ground, candidate_dino
                break
            time.sleep(.04)
        if region is None or ground is None or hint is None:
            raise RuntimeError('启动后仍无法从截图定位游戏；请确认 Chrome 在前台，并尝试 --region 框住奔跑画面')
        scale = max(.7, hint.h / 41)
        standing_height, standing_bottom = hint.h, hint.bottom
        tracker = Tracker(args.default_speed * scale, args.max_speed * scale, scale)
        print(f'游戏区域 {region}；地面 y={ground}；恐龙 {hint}；缩放 {scale:.2f}')
        play_started = time.monotonic()
        try:
            while not stop.stopped and (args.duration <= 0 or time.monotonic() - play_started < args.duration):
                now = time.monotonic()
                frame = np.asarray(capture.grab(region))
                motion_mask = foreground(frame, args.contrast)[max(0, ground - 75):min(frame.shape[0], ground + 9)]
                if previous_motion_mask is not None and np.count_nonzero(cv2.absdiff(motion_mask, previous_motion_mask)) >= 12:
                    last_visual_motion = now
                previous_motion_mask = motion_mask
                dino, found = detect(frame, ground, args.contrast, hint)
                if dino:
                    hint = dino
                else:
                    dino = hint
                tracked = tracker.update(found, ground, now)
                acted.intersection_update({item.id for item in tracked})
                action = 'wait'

                if duck_id is not None:
                    current = next((item for item in tracked if item.id == duck_id), None)
                    if current:
                        duck_until = now + max(0, (current.box.right - dino.x + 8) / tracker.speed)
                    if now >= duck_until:
                        pyautogui.keyUp('down')
                        duck_id = None
                        action = 'release duck'

                threats = [item for item in tracked if item.box.x > dino.right
                           and item.box.x - dino.right <= args.lookahead * scale
                           and required_action(item, ground, standing_height, scale) != 'none']
                threat = min(threats, key=lambda item: item.box.x) if threats else None
                if threat:
                    distance = threat.box.x - dino.right
                    requested = required_action(threat, ground, standing_height, scale)
                    if requested == 'duck' and distance <= max(40 * scale, tracker.speed * args.duck_ttc):
                        if duck_id is None and threat.id not in acted:
                            pyautogui.keyDown('down')
                            duck_id = threat.id
                            duck_until = now + (distance + threat.box.w + dino.w + 8) / tracker.speed
                            action = 'duck'
                            ducks += 1
                    elif requested == 'jump':
                        lead = max(args.min_lead * scale, tracker.speed * args.jump_apex - (threat.box.w + dino.w) / 2)
                        grounded = dino.bottom >= standing_bottom - 7 * scale
                        if (0 < distance <= lead and grounded and now - last_jump >= args.cooldown
                                and threat.id not in acted and threat.id != duck_id):
                            if duck_id is not None:
                                pyautogui.keyUp('down')
                                duck_id = None
                            pyautogui.press('space')
                            last_jump = now
                            acted.add(threat.id)
                            action = 'jump'
                            jumps += 1
                            print(f'跳跃 {threat.kind}#{threat.id} 距离={distance}px 速度={tracker.speed:.0f}px/s')

                if args.auto_restart and now - last_visual_motion > 1.2:
                    if duck_id is not None:
                        pyautogui.keyUp('down')
                        duck_id = None
                    pyautogui.press('space')
                    tracker.reset()
                    acted.clear()
                    last_visual_motion = now
                    previous_motion_mask = None
                    action = 'restart'
                    restarts += 1
                    print('检测到画面停止移动，尝试重新开始')

                if args.debug and now - last_debug >= .5:
                    cv2.imwrite(str(args.debug_image), annotate(frame, ground, dino, tracked, threat, tracker.speed, action))
                    last_debug = now
                remaining = 1 / args.fps - (time.monotonic() - now)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if duck_id is not None:
                pyautogui.keyUp('down')
            print(f'运行 {time.monotonic() - play_started:.1f} 秒：跳跃 {jumps} 次，下蹲 {ducks} 次，重开 {restarts} 次')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', type=parse_region, help='屏幕区域 x,y,width,height；默认自动寻找')
    parser.add_argument('--contrast', type=int, default=45)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--lookahead', type=int, default=350)
    parser.add_argument('--default-speed', type=float, default=360)
    parser.add_argument('--max-speed', type=float, default=1500)
    parser.add_argument('--jump-apex', type=float, default=.30, help='起跳到最高点的目标时间（秒）')
    parser.add_argument('--min-lead', type=float, default=35)
    parser.add_argument('--duck-ttc', type=float, default=.30)
    parser.add_argument('--cooldown', type=float, default=.20)
    parser.add_argument('--start-delay', type=float, default=3)
    parser.add_argument('--duration', type=float, default=0, help='运行秒数；0 表示一直运行')
    parser.add_argument('--no-auto-restart', dest='auto_restart', action='store_false')
    parser.add_argument('--debug', action='store_true', help='每 0.5 秒覆盖保存识别标注图')
    parser.add_argument('--debug-image', type=Path, default=Path('dino_debug.png'))
    args = parser.parse_args()
    if args.fps <= 0 or args.default_speed <= 0 or args.max_speed <= 0 or not 1 <= args.contrast <= 255:
        parser.error('fps、speed 必须为正数，contrast 必须在 1..255')
    try:
        with mss.MSS() as capture:
            monitor = capture.monitors[0]
            if monitor['width'] <= 0 or monitor['height'] <= 0:
                raise RuntimeError('系统未提供可捕获的显示器；请在桌面会话中运行，并检查屏幕录制权限')
        run(monitor, args.region, args)
    except KeyboardInterrupt:
        print('已停止')
    except RuntimeError as exc:
        parser.error(str(exc))
    except ScreenShotError as exc:
        parser.error(f'屏幕截图失败：{exc}；请检查屏幕录制权限')


if __name__ == '__main__':
    main()
