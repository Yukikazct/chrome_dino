"""Pixel-level checks using the repository's Chrome Dino sprite sheet."""
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dino_cv import Tracker, auto_region, detect, foreground, ground_row, is_game_over, kind, required_action

SPRITE = np.asarray(Image.open(Path(__file__).parent / 'assets/100-offline-sprite.png'))


def put_sprite(frame, x, y, source_x, width, height, ink):
    pixels = SPRITE[:height, source_x:source_x + width] > 120
    frame[y:y + height, x:x + width, :3][pixels] = ink


def scene(background=255, ink=80, cactus_x=270, bird_y=None):
    frame = np.full((200, 600, 4), background, np.uint8)
    frame[:, :, 3] = 255
    frame[180, :, :3] = ink
    put_sprite(frame, 50, 132, 40, 44, 48, ink)
    if cactus_x is not None:
        put_sprite(frame, cactus_x, 145, 228, 17, 35, ink)
    if bird_y is not None:
        put_sprite(frame, 350, bird_y, 1110, 46, 40, ink)
    return frame


class VisionTests(unittest.TestCase):
    def test_day_and_night_sprites(self):
        for background, ink in ((255, 80), (0, 220)):
            with self.subTest(background=background):
                frame = scene(background, ink)
                ground = ground_row(foreground(frame, 45))
                self.assertEqual(ground, 180)
                dino, obstacles = detect(frame, ground, 45)
                self.assertIsNotNone(dino)
                self.assertTrue(45 <= dino.x <= 55)
                self.assertEqual(len(obstacles), 1)
                self.assertEqual(kind(obstacles[0], ground), 'cactus')

    def test_bird_height_changes_action(self):
        for y, expected in ((85, 'none'), (113, 'duck'), (140, 'jump')):
            with self.subTest(y=y):
                frame = scene(cactus_x=None, bird_y=y)
                dino, obstacles = detect(frame, 180, 45)
                self.assertIsNotNone(dino)
                self.assertEqual(len(obstacles), 1)
                tracker = Tracker(360, 1500)
                item = tracker.update(obstacles, 180, 1.0)[0]
                self.assertEqual(required_action(item, 180, dino.h), expected)

    def test_tracks_motion_from_pixels(self):
        tracker = Tracker(360, 1500)
        _, first = detect(scene(cactus_x=270), 180, 45)
        _, second = detect(scene(cactus_x=258), 180, 45)
        before = tracker.update(first, 180, 1.0)
        after = tracker.update(second, 180, 1.04)
        self.assertEqual(before[0].id, after[0].id)
        self.assertAlmostEqual(tracker.speed, 300, delta=1)

    def test_auto_region_uses_ground_line(self):
        frame = np.full((500, 900, 4), 255, np.uint8)
        frame[:, :, 3] = 255
        frame[320, 100:700, :3] = 80
        found = auto_region(frame, dict(left=0, top=0, width=900, height=500), 45)
        self.assertIsNotNone(found)
        self.assertEqual(found['left'], 0)
        self.assertEqual(found['width'], 900)
        self.assertEqual(found['top'], 20)

    def test_official_chrome_frames(self):
        assets = Path(__file__).parent / 'assets'
        import cv2
        bird_frame = cv2.imread(str(assets / 'official_bird.png'))
        game_over_frame = cv2.imread(str(assets / 'official_game_over.png'))
        ground = ground_row(foreground(bird_frame, 45))
        dino, obstacles = detect(bird_frame, ground, 45)
        self.assertEqual(len(obstacles), 1)
        scale = dino.h / 41
        item = Tracker(360 * scale, 1500 * scale, scale).update(obstacles, ground, 1)[0]
        self.assertEqual(required_action(item, ground, dino.h, scale), 'duck')
        self.assertFalse(is_game_over(bird_frame, ground, 45, scale))
        self.assertTrue(is_game_over(game_over_frame, ground, 45, scale))


if __name__ == '__main__':
    unittest.main()
