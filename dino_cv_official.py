#!/usr/bin/env python3
"""Computer-vision autoplayer for Google's official chrome://dino game.

ChromeDriver is used only to capture rendered pixels and send keyboard events.
No DOM, Runner state, canvas internals, or game speed is read.
"""
from __future__ import annotations

import argparse
import io
import platform
import re
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service

from dino_cv import Box, Obstacle, Tracker, annotate, detect, foreground, ground_row, is_game_over, required_action


def find_chromedriver(override: Path | None) -> Path | None:
    if override is not None:
        if not override.is_file():
            raise RuntimeError(f'ChromeDriver 不存在：{override}')
        return override
    command = shutil.which('chromedriver')
    if command:
        return Path(command)
    cache = Path.home() / '.cache' / 'selenium' / 'chromedriver'
    candidates = list(cache.glob('**/chromedriver')) if cache.exists() else []
    architecture = platform.machine().lower()
    candidates = [p for p in candidates if p.is_file() and (
        ('arm64' in str(p)) == (architecture == 'arm64') or architecture not in ('arm64', 'x86_64')
    )]
    def version(path: Path) -> tuple[int, ...]:
        return tuple(map(int, re.findall(r'\d+', path.parent.name)))
    return max(candidates, key=version, default=None)


def start_chrome(args: argparse.Namespace) -> tuple[webdriver.Chrome, Path]:
    profile = Path(tempfile.mkdtemp(prefix='dino-cv-chrome-'))
    options = webdriver.ChromeOptions()
    options.add_argument(f'--user-data-dir={profile}')
    options.add_argument(f'--window-size={args.width},{args.height}')
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')
    options.add_argument('--force-device-scale-factor=1')
    if args.headless:
        options.add_argument('--headless=new')
    driver_path = find_chromedriver(args.chromedriver)
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(str(driver_path)) if driver_path else None, options=options)
        driver.set_page_load_timeout(8)
        try:
            driver.get('chrome://dino')
        except WebDriverException as exc:
            # Chrome reports the offline game page as a navigation error while
            # still rendering it. Other navigation failures remain errors.
            if 'ERR_INTERNET_DISCONNECTED' not in str(exc) or not driver.current_url.startswith('chrome://dino'):
                raise
        return driver, profile
    except BaseException:
        if driver is not None:
            driver.quit()
        shutil.rmtree(profile, ignore_errors=True)
        raise


def capture(driver: webdriver.Chrome) -> np.ndarray:
    png = driver.get_screenshot_as_png()
    image = np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def key_event(driver: webdriver.Chrome, key: str, down: bool) -> None:
    if key == 'space':
        code, name, virtual = 'Space', ' ', 32
    else:
        code, name, virtual = 'ArrowDown', 'ArrowDown', 40
    driver.execute_cdp_cmd('Input.dispatchKeyEvent', {
        'type': 'keyDown' if down else 'keyUp',
        'key': name, 'code': code,
        'windowsVirtualKeyCode': virtual,
    })


def press_space(driver: webdriver.Chrome) -> None:
    key_event(driver, 'space', True)
    key_event(driver, 'space', False)


def calibrate(driver: webdriver.Chrome, contrast: int) -> tuple[int, Box, np.ndarray]:
    press_space(driver)
    started = time.monotonic()
    deadline = started + 4.0
    stable: list[tuple[int, int, int]] = []
    while time.monotonic() < deadline:
        frame = capture(driver)
        ground = ground_row(foreground(frame, contrast))
        if ground is not None:
            dino, _ = detect(frame, ground, contrast)
            if dino is not None and dino.bottom >= ground - 14 and time.monotonic() - started >= 1.0:
                sample = (ground, dino.x, dino.bottom)
                if stable and (abs(sample[0] - stable[-1][0]) > 2 or abs(sample[1] - stable[-1][1]) > 3):
                    stable.clear()
                stable.append(sample)
                if len(stable) >= 4:
                    return ground, dino, frame
            else:
                stable.clear()
        time.sleep(.03)
    raise RuntimeError('官方游戏启动后无法从截图定位地面和恐龙')


def run(args: argparse.Namespace) -> None:
    driver, profile = start_chrome(args)
    duck_id: int | None = None
    jumps = ducks = restarts = 0
    try:
        ground, hint, frame = calibrate(driver, args.contrast)
        scale = max(.7, hint.h / 41)
        tracker = Tracker(args.default_speed * scale, args.max_speed * scale, scale)
        standing_height = hint.h
        standing_bottom = hint.bottom
        print(f'已连接官方 chrome://dino：画面 {frame.shape[1]}x{frame.shape[0]}，地面 {ground}，恐龙 {hint}，缩放 {scale:.2f}', flush=True)
        played_at = time.monotonic()
        last_jump = -10.0
        last_debug = 0.0
        last_motion = played_at
        previous_motion: np.ndarray | None = None
        acted: set[int] = set()
        recorded_birds: set[int] = set()
        duck_until = 0.0
        last_restart = -10.0
        while args.duration <= 0 or time.monotonic() - played_at < args.duration:
            now = time.monotonic()
            frame = capture(driver)
            if is_game_over(frame, ground, args.contrast, scale):
                if not args.auto_restart:
                    print('识别到 GAME OVER，已停止', flush=True)
                    break
                if now - last_restart > .7:
                    if duck_id is not None:
                        key_event(driver, 'down', False)
                        duck_id = None
                    press_space(driver)
                    tracker.reset()
                    acted.clear()
                    recorded_birds.clear()
                    previous_motion = None
                    last_motion = now
                    last_restart = now
                    restarts += 1
                    print('识别到 GAME OVER，重新开始', flush=True)
                time.sleep(.05)
                continue
            # Cropping below the score avoids treating score changes as motion.
            motion = foreground(frame, args.contrast)[max(0, ground - int(75 * scale)):min(frame.shape[0], ground + 9)]
            if previous_motion is not None and np.count_nonzero(cv2.absdiff(motion, previous_motion)) >= 12:
                last_motion = now
            previous_motion = motion
            dino, found = detect(frame, ground, args.contrast, hint)
            if dino is not None:
                hint = dino
            else:
                dino = hint
            tracked = tracker.update(found, ground, now)
            acted.intersection_update({item.id for item in tracked})
            action = 'wait'

            if duck_id is not None:
                current = next((item for item in tracked if item.id == duck_id), None)
                if current is not None:
                    duck_until = now + max(0, (current.box.right - dino.x + 8 * scale) / tracker.speed)
                if now >= duck_until:
                    key_event(driver, 'down', False)
                    duck_id = None
                    action = 'release duck'

            threats = [item for item in tracked
                       if dino.right + 25 * scale < item.box.x <= dino.right + args.lookahead * scale
                       and required_action(item, ground, standing_height, scale) != 'none']
            threat: Obstacle | None = min(threats, key=lambda item: item.box.x) if threats else None
            if threat is not None:
                distance = threat.box.x - dino.right
                requested = required_action(threat, ground, standing_height, scale)
                if args.event_dir and threat.kind == 'bird' and threat.id not in recorded_birds:
                    args.event_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.event_dir / f'bird_{threat.id}_{requested}_{distance}.png'), frame)
                    recorded_birds.add(threat.id)
                    print(f'bird seen #{threat.id} box={threat.box} action={requested}', flush=True)
                if requested == 'duck':
                    if (distance <= max(40 * scale, tracker.speed * args.duck_ttc)
                            and duck_id is None and threat.id not in acted):
                        key_event(driver, 'down', True)
                        duck_id = threat.id
                        duck_until = now + (distance + threat.box.w + dino.w + 8 * scale) / tracker.speed
                        action = 'duck'
                        ducks += 1
                        print(f'duck {threat.kind}#{threat.id} distance={distance} speed={tracker.speed:.0f}', flush=True)
                else:
                    lead = max(args.min_lead * scale, tracker.speed * args.jump_apex - (threat.box.w + dino.w) / 2)
                    grounded = dino.bottom >= standing_bottom - 7 * scale
                    if (0 < distance <= lead and grounded and now - last_jump >= args.cooldown
                            and threat.id not in acted and threat.id != duck_id):
                        if duck_id is not None:
                            key_event(driver, 'down', False)
                            duck_id = None
                        press_space(driver)
                        last_jump = now
                        acted.add(threat.id)
                        action = 'jump'
                        jumps += 1
                        print(f'jump {threat.kind}#{threat.id} distance={distance} lead={lead:.0f} speed={tracker.speed:.0f}', flush=True)

            if args.auto_restart and now - last_motion > 1.2:
                if duck_id is not None:
                    key_event(driver, 'down', False)
                    duck_id = None
                press_space(driver)
                tracker.reset()
                acted.clear()
                previous_motion = None
                last_motion = now
                action = 'restart'
                restarts += 1
                print('画面停止，重新开始', flush=True)

            if args.debug and now - last_debug >= .5:
                cv2.imwrite(str(args.debug_image), annotate(frame, ground, dino, tracked, threat, tracker.speed, action))
                last_debug = now
            remaining = 1 / args.fps - (time.monotonic() - now)
            if remaining > 0:
                time.sleep(remaining)
        print(f'运行 {time.monotonic()-played_at:.1f} 秒，跳跃 {jumps}，下蹲 {ducks}，重开 {restarts}', flush=True)
    finally:
        if duck_id is not None:
            try:
                key_event(driver, 'down', False)
            except WebDriverException:
                pass
        try:
            driver.quit()
        finally:
            shutil.rmtree(profile, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='不显示 Chrome 窗口；用于自动化测试')
    parser.add_argument('--chromedriver', type=Path, help='ChromeDriver 可执行文件路径')
    parser.add_argument('--width', type=int, default=1200)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--duration', type=float, default=0, help='运行秒数；0 表示一直运行')
    parser.add_argument('--contrast', type=int, default=45)
    parser.add_argument('--lookahead', type=float, default=350)
    parser.add_argument('--default-speed', type=float, default=360)
    parser.add_argument('--max-speed', type=float, default=1500)
    parser.add_argument('--jump-apex', type=float, default=.30)
    parser.add_argument('--min-lead', type=float, default=35)
    parser.add_argument('--duck-ttc', type=float, default=.30)
    parser.add_argument('--cooldown', type=float, default=.20)
    parser.add_argument('--no-auto-restart', dest='auto_restart', action='store_false')
    parser.add_argument('--debug', action='store_true', help='每半秒覆盖写入识别标注图')
    parser.add_argument('--debug-image', type=Path, default=Path('dino_debug.png'))
    parser.add_argument('--event-dir', type=Path, help='保存首次看到每只翼龙时的原始截图')
    args = parser.parse_args()
    if args.width < 600 or args.height < 400 or args.fps <= 0 or args.default_speed <= 0 or args.max_speed <= 0:
        parser.error('窗口至少 600x400；fps 和速度必须为正数')
    try:
        run(args)
    except KeyboardInterrupt:
        print('已停止')
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
