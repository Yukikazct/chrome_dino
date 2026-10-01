#!/usr/bin/env python3
"""Stable Chrome Dino player using the game's live Runner state.

Obstacle positions and sizes are read from the game's Runner instance instead
of inferred from screenshots.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import  time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By


@dataclass(frozen=True)
class Obstacle:
    x: int
    y: int
    width: int
    height: int
    kind: str = ""


@dataclass(frozen=True)
class Detection:
    label: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RunnerSnapshot:
    canvas_width: int
    canvas_height: int
    dino_x: int
    dino_width: int
    dino_jumping: bool
    dino: Detection
    obstacles: list[Obstacle]
    display_obstacles: list[Obstacle]
    ground_y: int
    game_ground_y: int
    standing_height: int
    duck_height: int
    scale_x: float
    scale_y: float
    speed: float
    crashed: bool


OBSTACLE_SPRITES = (
    ("CACTUS_SMALL", 228, 0, 17, 35),
    ("CACTUS_LARGE", 332, 0, 25, 50),
    ("PTERODACTYL", 134, 0, 46, 40),
)


def read_runner_snapshot(driver: webdriver.Chrome) -> RunnerSnapshot | None:
    state = driver.execute_script(
        """
        const runner = window.Runner && window.Runner.instance_;
        const canvas = document.querySelector('.runner-canvas');
        if (!runner || !canvas) return null;
        const gameWidth = runner.dimensions.WIDTH;
        const gameHeight = runner.dimensions.HEIGHT;
        const scaleX = canvas.width / gameWidth;
        const scaleY = canvas.height / gameHeight;
        const trex = runner.tRex;
        const dinoWidth = trex.ducking ? trex.config.WIDTH_DUCK : trex.config.WIDTH;
        const dinoHeight = trex.ducking ? trex.config.HEIGHT_DUCK : trex.config.HEIGHT;
        return {
          canvasWidth: canvas.width,
          canvasHeight: canvas.height,
          dinoX: trex.xPos,
          dinoY: trex.yPos,
          dinoWidth,
          dinoHeight,
          dinoJumping: trex.jumping,
          groundY: gameHeight - runner.config.BOTTOM_PAD,
          standingHeight: trex.config.HEIGHT,
          duckHeight: trex.config.HEIGHT_DUCK,
          speed: runner.currentSpeed * 60,
          crashed: runner.crashed,
          obstacles: runner.horizon.obstacles.map((item) => ({
            x: item.xPos,
            y: item.yPos,
            width: item.width,
            height: item.typeConfig.height,
            kind: item.typeConfig.type,
          })),
          scaleX,
          scaleY,
        };
        """
    )
    if state is None:
        return None
    if not isinstance(state, dict):
        raise RuntimeError(f"Runner 返回了无效游戏状态：{state!r}")

    canvas_width = int(state["canvasWidth"])
    canvas_height = int(state["canvasHeight"])
    scale_x = float(state["scaleX"])
    scale_y = float(state["scaleY"])
    dino_x = int(state["dinoX"])
    dino_width = int(state["dinoWidth"])
    dino_y = int(state["dinoY"])
    dino_height = int(state["dinoHeight"])
    obstacles = [
        Obstacle(
            int(item["x"]),
            int(item["y"]),
            int(item["width"]),
            int(item["height"]),
            str(item["kind"]),
        )
        for item in state["obstacles"]
    ]
    return RunnerSnapshot(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        dino_x=dino_x,
        dino_width=dino_width,
        dino_jumping=bool(state["dinoJumping"]),
        dino=Detection(
            "DINO",
            round(dino_x * scale_x),
            round(dino_y * scale_y),
            round(dino_width * scale_x),
            round(dino_height * scale_y),
        ),
        obstacles=obstacles,
        display_obstacles=[
            Obstacle(
                round(item.x * scale_x),
                round(item.y * scale_y),
                round(item.width * scale_x),
                round(item.height * scale_y),
            )
            for item in obstacles
        ],
        ground_y=round(int(state["groundY"]) * scale_y),
        game_ground_y=int(state["groundY"]),
        standing_height=int(state["standingHeight"]),
        duck_height=int(state["duckHeight"]),
        scale_x=scale_x,
        scale_y=scale_y,
        speed=float(state["speed"]),
        crashed=bool(state["crashed"]),
    )


def load_obstacle_templates() -> list[tuple[str, np.ndarray, np.ndarray]]:
    sprite_path = Path(__file__).with_name("assets") / "100-offline-sprite.png"
    if not sprite_path.exists():
        return []
    sprite = cv2.imread(str(sprite_path), cv2.IMREAD_GRAYSCALE)
    if sprite is None:
        return []
    templates = []
    for name, x, y, width, height in OBSTACLE_SPRITES:
        template = sprite[y : y + height, x : x + width]
        mask = np.where(template < 160, 255, 0).astype(np.uint8)
        templates.append((name, template, mask))
    return templates


def find_sprite_obstacles(
    frame: np.ndarray,
    ground_y: int,
    templates: list[tuple[str, np.ndarray, np.ndarray]],
) -> list[Obstacle]:
    if not templates:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result: list[Obstacle] = []
    for name, template, mask in templates:
        height, width = template.shape
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        ys, xs = np.where(scores >= 0.88)
        for x, y in zip(xs, ys):
            if x < 100:
                continue
            bottom = y + height
            if name == "PTERODACTYL":
                if bottom < ground_y - 85 or bottom > ground_y - 12:
                    continue
            elif bottom < ground_y - 8 or bottom > ground_y + 2:
                continue
            candidate = Obstacle(int(x), int(y), width, height)
            if all(abs(candidate.x - item.x) > max(width, item.width) for item in result):
                result.append(candidate)
    return sorted(result, key=lambda item: item.x)


def canvas_frame(driver: webdriver.Chrome, canvas) -> np.ndarray:
    png = None
    for _ in range(3):
        try:
            png = canvas.screenshot_as_png
        except AttributeError as exc:
            if "encode" not in str(exc):
                raise
        if png:
            break
        time.sleep(0.05)
    if not png:
        raise RuntimeError("Canvas 返回了空截图，页面可能正在重载")
    full = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    canvas_size = driver.execute_script(
        "return [arguments[0].width, arguments[0].height];", canvas
    )
    if not canvas_size or not all(canvas_size):
        raise RuntimeError("Canvas 的尺寸信息为空，页面可能正在重载")
    normalized = cv2.resize(
        full,
        (int(canvas_size[0]), int(canvas_size[1])),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)


def install_browser_overlay(driver: webdriver.Chrome, canvas) -> None:
    driver.execute_script(
        """
        if (!document.getElementById('dino-cv-overlay')) {
          const overlay = document.createElement('div');
          overlay.id = 'dino-cv-overlay';
          overlay.style.cssText =
            'position:fixed;left:8px;top:8px;z-index:2147483647;' +
            'background:rgba(255,255,0,.92);color:#000;padding:6px 8px;' +
            'font:13px monospace;white-space:pre;pointer-events:none;';
          overlay.textContent = 'Dino Runner starting...';
          document.body.appendChild(overlay);
          const layer = document.createElement('div');
          layer.id = 'dino-cv-boxes';
          layer.style.cssText =
            'position:fixed;z-index:2147483646;pointer-events:none;' +
            'left:0;top:0;overflow:visible;box-sizing:border-box;';
          document.body.appendChild(layer);
        }
        """,
        canvas,
    )


def update_browser_overlay(
    driver: webdriver.Chrome,
    canvas,
    dino: Detection | None,
    obstacles: list[Obstacle],
    threat: Obstacle | None,
    ground_y: int,
    speed: float,
    ttc: float,
    action: str,
) -> None:
    payload = {
        "dino": None if dino is None else [dino.x, dino.y, dino.width, dino.height],
        "obstacles": [[item.x, item.y, item.width, item.height] for item in obstacles],
        "threat": None
        if threat is None
        else [threat.x, threat.y, threat.width, threat.height],
        "ground": ground_y,
        "speed": round(speed),
        "ttc": "inf" if not np.isfinite(ttc) else f"{ttc:.2f}",
        "action": action,
    }
    driver.execute_script(
        """
        const canvas = arguments[0], data = arguments[1];
        const id = 'dino-cv-overlay';
        let overlay = document.getElementById(id);
        if (!overlay) return;
        const lines = [
          'DINO Runner browser overlay',
          `canvas=${canvas.width}x${canvas.height} ground=${data.ground}`,
          `dino=${data.dino || 'none'}`,
          `obstacles=${data.obstacles.length} threat=${data.threat || 'none'}`,
          `speed=${data.speed} ttc=${data.ttc} action=${data.action}`,
        ];
        const text = lines.join('\\n');
        if (overlay.textContent !== text) overlay.textContent = text;
        const layer = document.getElementById('dino-cv-boxes');
        if (!layer) return;
        const rect = canvas.getBoundingClientRect();
        layer.style.left = `${rect.left}px`;
        layer.style.top = `${rect.top}px`;
        layer.style.width = `${rect.width}px`;
        layer.style.height = `${rect.height}px`;
        const sx = rect.width / canvas.width;
        const sy = rect.height / canvas.height;
        const boxes = [
          [data.dino, '#00aa00'],
          ...data.obstacles.map((box) => [box, '#00aaaa']),
          [data.threat, '#ff0000'],
        ].filter(([box]) => box);
        while (layer.children.length < boxes.length) {
          const node = document.createElement('div');
          node.style.cssText =
            'position:absolute;left:0;top:0;display:block;' +
            'border:3px solid;box-sizing:border-box;pointer-events:none;';
          layer.appendChild(node);
        }
        for (let index = 0; index < layer.children.length; index++) {
          const node = layer.children[index];
          if (index >= boxes.length) {
            node.style.display = 'none';
            continue;
          }
          const [box, color] = boxes[index];
          const [x, y, w, h] = box;
          node.style.display = 'block';
          node.style.borderColor = color;
          node.style.transform =
            `translate3d(${Math.round(x * sx)}px,${Math.round(y * sy)}px,0)`;
          node.style.width = `${Math.max(3, Math.round(w * sx))}px`;
          node.style.height = `${Math.max(3, Math.round(h * sy))}px`;
        }
        """,
        canvas,
        payload,
    )


def find_obstacles(frame: np.ndarray, ground_y: int, threshold: int) -> list[Obstacle]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, threshold)
    # Ignore the score at the top and the ground row at the bottom.  Scanning
    # almost to the ground is important for short cacti, whose visible
    # contour can be only a few pixels high after canvas resizing.
    scan = mask[max(0, ground_y - 70) : max(1, ground_y), :]
    contours, _ = cv2.findContours(scan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[Obstacle] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        y += ground_y - 70
        # The Dino occupies the fixed left-side spawn area; it is detected by
        # find_dino separately and must not enter the obstacle list.
        if x < 100:
            continue
        # Ground pixels can be split into wide, shallow contours. They are not
        # obstacles and otherwise dominate the nearest-threat decision.
        if (
            2 <= width <= 70
            and 8 <= height <= 60
            and ground_y - (y + height) <= 8
        ):
            result.append(Obstacle(x, y, width, height))
    return sorted(result, key=lambda item: item.x)


def detect_ground_y(frame: np.ndarray, threshold: int, fallback_ratio: float) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, threshold)
    counts = np.count_nonzero(mask, axis=1)
    start = int(frame.shape[0] * 0.60)
    end = int(frame.shape[0] * 0.98)
    if end > start:
        return int(start + np.argmax(counts[start:end]))
    return int(frame.shape[0] * fallback_ratio)


def find_dino(frame: np.ndarray, ground_y: int, threshold: int) -> Detection | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, threshold)
    scan = mask[max(0, ground_y - 55) : max(1, ground_y - 4), :120]
    contours, _ = cv2.findContours(scan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Detection] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        y += max(0, ground_y - 55)
        if 5 <= width <= 70 and 8 <= height <= 55 and y + height >= ground_y - 42:
            candidates.append(Detection("DINO", x, y, width, height))
    return max(candidates, key=lambda item: item.width * item.height, default=None)


def overlay_text(frame: np.ndarray, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (8, 18 + index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (8, 18 + index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def choose_obstacle(obstacles: list[Obstacle], dino_x: int, dino_width: int) -> Obstacle | None:
    right = dino_x + dino_width
    ahead = [item for item in obstacles if item.x > right and item.x < right + 350]
    return min(ahead, key=lambda item: item.x) if ahead else None


def obstacle_action(
    obstacle: Obstacle,
    ground_y: int,
    standing_height: int,
    duck_height: int,
) -> str:
    if obstacle.kind != "PTERODACTYL":
        return "JUMP"

    bird_bottom = obstacle.y + obstacle.height
    standing_top = ground_y - standing_height
    ducking_top = ground_y - duck_height
    if bird_bottom > ducking_top:
        return "JUMP"
    if bird_bottom > standing_top:
        return "DUCK"
    return "NONE"


def choose_threat(
    obstacles: list[Obstacle],
    dino_x: int,
    dino_width: int,
    ground_y: int,
    standing_height: int,
    duck_height: int,
) -> tuple[Obstacle | None, str]:
    right = dino_x + dino_width
    ahead = [
        item
        for item in obstacles
        if right < item.x < right + 350
        and obstacle_action(item, ground_y, standing_height, duck_height) != "NONE"
    ]
    threat = min(ahead, key=lambda item: item.x) if ahead else None
    return (
        (None, "NONE")
        if threat is None
        else (
            threat,
            obstacle_action(threat, ground_y, standing_height, duck_height),
        )
    )


def update_jumped_obstacle(
    obstacles: list[Obstacle],
    jumped: Obstacle | None,
    updated_at: float | None,
    speed: float,
    now: float,
    dino_x: int,
    passed_margin: int,
    max_tracking_jump: int,
    max_width_change: int,
) -> tuple[Obstacle | None, float | None]:
    if jumped is None or updated_at is None:
        return jumped, updated_at

    elapsed = max(0.0, now - updated_at)
    predicted_x = jumped.x - round(speed * elapsed)
    if predicted_x + jumped.width < dino_x - passed_margin:
        return None, None

    tolerance = max(max_tracking_jump, round(speed * elapsed) + 12)
    matches = [
        item
        for item in obstacles
        if abs(item.x - predicted_x) <= tolerance
        and abs(item.width - jumped.width) <= max_width_change
    ]
    if matches:
        jumped = min(matches, key=lambda item: abs(item.x - predicted_x))
    else:
        jumped = Obstacle(predicted_x, jumped.y, jumped.width, jumped.height)
    return jumped, now


def jump_decision(
    distance: float,
    obstacle_width: int,
    speed: float,
    now: float,
    last_jump: float,
    airborne_until: float,
    args: argparse.Namespace,
    dino_width: int = 44,
) -> tuple[bool, str, float]:
    effective_speed = max(args.default_speed, min(speed, args.max_speed))
    effective_distance = max(0.0, distance)
    if distance <= 0:
        return False, "TOO_LATE", effective_distance / effective_speed
    ttc = effective_distance / effective_speed
    overlap_width = obstacle_width + dino_width
    lead_distance = max(
        args.min_lead_distance,
        effective_speed * args.jump_ttc - overlap_width / 2,
    )
    if now < airborne_until or now - last_jump < args.cooldown:
        return False, "AIRBORNE", ttc
    if effective_distance <= lead_distance:
        return True, "TTC", ttc
    return False, "WAIT", ttc


def press_space(driver: webdriver.Chrome) -> None:
    """Send a browser-level key event that does not depend on element focus."""
    driver.execute_cdp_cmd(
        "Input.dispatchKeyEvent",
        {"type": "keyDown", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
    )
    driver.execute_cdp_cmd(
        "Input.dispatchKeyEvent",
        {"type": "keyUp", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
    )


def set_duck(driver: webdriver.Chrome, ducking: bool) -> None:
    driver.execute_script(
        "Runner.instance_.tRex.setDuck(arguments[0]);",
        ducking,
    )


def run(args: argparse.Namespace) -> None:
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=800,400")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-dev-shm-usage")
    profile_dir = tempfile.mkdtemp(prefix="dino-chrome-")
    options.add_argument(f"--user-data-dir={profile_dir}")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
    except Exception:
        print("ChromeDriver 启动失败，下面是完整错误：")
        traceback.print_exc()
        print(
            "\n常见原因：Chrome/ChromeDriver 版本不匹配、已有 Chrome 调试进程冲突，"
            "或 ChromeDriver 没有权限启动。"
        )
        raise
    failed = False
    try:
        driver.get(args.url)
        deadline = time.monotonic() + 15
        canvas = None
        snapshot = None
        while time.monotonic() < deadline:
            canvas_elements = driver.find_elements(By.CSS_SELECTOR, ".runner-canvas")
            if canvas_elements:
                canvas = canvas_elements[0]
                snapshot = read_runner_snapshot(driver)
                if snapshot is not None:
                    break
            time.sleep(0.1)
        if canvas is None or snapshot is None:
            raise RuntimeError(
                "游戏页未提供可读取的 Runner 状态。请确认 --url 指向可访问的 "
                "Chrome Dino 游戏页面。"
            )
        install_browser_overlay(driver, canvas)
        press_space(driver)
        print(
            f"Runner 状态已连接：canvas={snapshot.canvas_width}x{snapshot.canvas_height}, "
            f"game-speed={snapshot.speed:.0f}px/s"
        )
        time.sleep(args.start_delay)

        last_jump = 0.0
        airborne_until = 0.0
        speed = args.default_speed
        threat_seen = 0
        jumped_obstacle: Obstacle | None = None
        jumped_obstacle_updated_at: float | None = None
        ducked_obstacle: Obstacle | None = None
        ducked_obstacle_updated_at: float | None = None
        tracked_threat_x: float | None = None
        tracked_threat_width: int | None = None

        while True:
            snapshot = read_runner_snapshot(driver)
            if snapshot is None:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if snapshot.crashed:
                set_duck(driver, False)
                obstacle_state = [
                    f"{item.kind}@{item.x},{item.y} {item.width}x{item.height}"
                    for item in snapshot.obstacles[:3]
                ]
                print(
                    "本局结束，正在重启游戏。"
                    f" dino={snapshot.dino.y} jumping={snapshot.dino_jumping}"
                    f" obstacles={obstacle_state}"
                )
                driver.execute_script("Runner.instance_.restart()")
                press_space(driver)
                last_jump = 0.0
                airborne_until = 0.0
                threat_seen = 0
                jumped_obstacle = None
                jumped_obstacle_updated_at = None
                ducked_obstacle = None
                ducked_obstacle_updated_at = None
                tracked_threat_x = None
                tracked_threat_width = None
                time.sleep(0.2)
                continue

            speed = args.alpha * snapshot.speed + (1 - args.alpha) * speed
            tracking_speed = max(args.default_speed, min(speed, args.max_speed))
            dino_x = snapshot.dino_x
            dino_width = snapshot.dino_width
            obstacles = snapshot.obstacles
            display_obstacles = snapshot.display_obstacles
            was_ducking = ducked_obstacle is not None
            ducked_obstacle, ducked_obstacle_updated_at = update_jumped_obstacle(
                obstacles,
                ducked_obstacle,
                ducked_obstacle_updated_at,
                tracking_speed,
                now,
                dino_x,
                0,
                args.max_tracking_jump,
                args.max_width_change,
            )
            if was_ducking and ducked_obstacle is None:
                set_duck(driver, False)
                print("duck released")
            threat, threat_action = choose_threat(
                obstacles,
                dino_x,
                dino_width,
                snapshot.game_ground_y,
                snapshot.standing_height,
                snapshot.duck_height,
            )
            jumped_obstacle, jumped_obstacle_updated_at = update_jumped_obstacle(
                obstacles,
                jumped_obstacle,
                jumped_obstacle_updated_at,
                tracking_speed,
                now,
                dino_x,
                args.passed_obstacle_margin,
                args.max_tracking_jump,
                args.max_width_change,
            )
            ttc = float("inf")
            action = "NONE"
            if threat is not None:
                if (
                    tracked_threat_x is None
                    or abs(threat.x - tracked_threat_x) > args.max_tracking_jump
                    or (
                        tracked_threat_width is not None
                        and abs(threat.width - tracked_threat_width) > args.max_width_change
                    )
                ):
                    threat_seen = 0
                tracked_threat_x = threat.x
                tracked_threat_width = threat.width
                threat_seen += 1
                distance = threat.x - (dino_x + dino_width)
                if threat_action == "DUCK":
                    if distance <= args.duck_lead_distance and ducked_obstacle is None:
                        set_duck(driver, True)
                        ducked_obstacle = threat
                        ducked_obstacle_updated_at = now
                        action = "DUCK"
                        print(
                            f"duck type={threat.kind} distance={distance:.0f} "
                            f"speed={speed:.0f}"
                        )
                    elif ducked_obstacle is not None:
                        action = "DUCK"
                    else:
                        action = "WAIT_DUCK"
                else:
                    airborne_until = (
                        now + 1 / args.fps if snapshot.dino_jumping else now
                    )
                    can_jump, reason, ttc = jump_decision(
                        distance,
                        threat.width,
                        speed,
                        now,
                        last_jump,
                        airborne_until,
                        args,
                        dino_width,
                    )
                    confirmed = threat_seen >= args.confirm_frames
                    emergency = distance <= args.emergency_distance
                    same_obstacle_tolerance = max(
                        12,
                        min(
                            args.max_tracking_jump,
                            round(tracking_speed * 2 / args.fps) + 4,
                        ),
                    )
                    same_as_jumped = (
                        jumped_obstacle is not None
                        and abs(threat.x - jumped_obstacle.x)
                        <= same_obstacle_tolerance
                        and abs(threat.width - jumped_obstacle.width)
                        <= args.max_width_change
                    )
                    if can_jump and (confirmed or emergency) and not same_as_jumped:
                        if ducked_obstacle is not None:
                            set_duck(driver, False)
                            ducked_obstacle = None
                            ducked_obstacle_updated_at = None
                        press_space(driver)
                        action = "JUMP"
                        last_jump = now
                        airborne_until = now + max(
                            args.airborne_time, args.jump_cooldown_after_press
                        )
                        jumped_obstacle = threat
                        jumped_obstacle_updated_at = now
                        print(
                            f"jump type={threat.kind} reason={reason} "
                            f"width={threat.width} distance={distance:.0f} "
                            f"speed={speed:.0f} ttc={ttc:.2f}"
                        )
            else:
                threat_seen = 0
                tracked_threat_x = None
                tracked_threat_width = None

            if args.debug:
                display_threat = (
                    None
                    if threat is None
                    else Obstacle(
                        round(threat.x * snapshot.scale_x),
                        round(threat.y * snapshot.scale_y),
                        round(threat.width * snapshot.scale_x),
                        round(threat.height * snapshot.scale_y),
                    )
                )
                update_browser_overlay(
                    driver,
                    canvas,
                    snapshot.dino,
                    display_obstacles,
                    display_threat,
                    snapshot.ground_y,
                    speed,
                    ttc,
                    action,
                )
            time.sleep(1 / args.fps)
    except Exception as exc:
        failed = True
        print("\n程序异常，Chrome 窗口暂时保留以便检查。完整错误：")
        traceback.print_exc()
        if "target window already closed" in str(exc):
            print(
                "Chrome 浏览器进程已退出，未进入识别逻辑。请检查 Chrome 崩溃报告。"
            )
        if args.keep_open_on_error:
            print("浏览器将在 %.1f 秒后关闭；按 Ctrl-C 可立即停止。" % args.error_wait)
            time.sleep(args.error_wait)
        raise
    finally:
        if driver is not None and (not failed or not args.keep_open_on_error):
            driver.quit()
        if not failed:
            try:
                os.rmdir(profile_dir)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="https://chromedino.com/",
        help="Dino 游戏页面；需要页面公开 Runner.instance_ 状态",
    )
    parser.add_argument(
        "--jump-ttc",
        type=float,
        default=0.29,
        help="目标起跳后障碍与恐龙横向重叠区间的中心时间（秒）",
    )
    parser.add_argument("--default-speed", type=float, default=280)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--duck-lead-distance", type=float, default=100)
    parser.add_argument("--cooldown", type=float, default=0.42)
    parser.add_argument("--airborne-time", type=float, default=0.48)
    parser.add_argument("--confirm-frames", type=int, default=1)
    parser.add_argument("--min-lead-distance", type=float, default=35)
    parser.add_argument(
        "--emergency-distance",
        type=float,
        default=95,
        help="障碍物已接近时跳过连续帧确认，避免漏跳",
    )
    parser.add_argument("--max-speed", type=float, default=1200)
    parser.add_argument("--jump-cooldown-after-press", type=float, default=0.50)
    parser.add_argument(
        "--passed-obstacle-margin",
        type=int,
        default=12,
        help="障碍物完全经过恐龙后，才允许触发后续跳跃",
    )
    parser.add_argument(
        "--max-tracking-jump",
        type=int,
        default=45,
        help="障碍物单帧位移超过此值时，视为换了障碍并重新确认",
    )
    parser.add_argument(
        "--max-width-change",
        type=int,
        default=12,
        help="障碍物宽度单帧变化超过此值时，重新确认",
    )
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--keep-open-on-error",
        action="store_true",
        help="发生 Selenium 异常时暂不立即关闭 Chrome",
    )
    parser.add_argument(
        "--error-wait",
        type=float,
        default=15,
        help="异常后保留 Chrome 的秒数",
    )
    parser.add_argument(
        "--pause-on-error",
        action="store_true",
        help="异常后等待按 Enter，避免终端窗口瞬间关闭",
    )
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception:
        error_path = os.path.abspath("dino_selenium_error.log")
        with open(error_path, "w", encoding="utf-8") as error_file:
            traceback.print_exc(file=error_file)
        print(f"\n程序失败，完整错误已保存到：{error_path}")
        print("请把这个文件中最前面的异常类型和消息发给我。")
        if args.pause_on_error:
            input("按 Enter 退出...")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
