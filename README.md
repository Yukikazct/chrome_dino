# 谷歌官方小恐龙 CV 自动游玩

主程序 [dino_cv_official.py](dino_cv_official.py) 打开 **Google Chrome 自带的 `chrome://dino`**。每一帧从 Chrome 截图，用 OpenCV 识别地面、恐龙、仙人掌和翼龙，跟踪障碍物位移来估算速度，再向该 Chrome 窗口发送空格或下方向键。它不读取 DOM、`Runner.instance_`、Canvas 内部数据或游戏速度。浏览器协议只用于截图和键盘输入。

## 安装

需要 macOS、Google Chrome、Python 3.9+ 和与 Chrome 兼容的 ChromeDriver。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Selenium 会优先使用系统 PATH 或本机 Selenium 缓存中的 ChromeDriver。也可在启动时通过 `--chromedriver /path/to/chromedriver` 指定。ChromeDriver 可从 Chrome for Testing 的官方渠道下载。

## 运行

```bash
python dino_cv_official.py
```

程序会打开独立的 Chrome 窗口进入 `chrome://dino`，自动开始、跳跃、下蹲，并在碰撞后重新开始。按 `Ctrl-C` 停止。窗口焦点不影响它发送游戏按键。首次打开官方离线游戏时，ChromeDriver 可能报告 `ERR_INTERNET_DISCONNECTED`，程序会识别这个预期响应并继续截图。

调试选项：

```bash
python dino_cv_official.py --debug --duration 60
```

`--debug` 每半秒覆盖保存一张 `dino_debug.png`：绿色框是恐龙，青色框是障碍，红色框是当前威胁，蓝线是地面。`--duration` 可限定运行时间；`--headless` 可隐藏测试用 Chrome 窗口。`--event-dir /tmp/dino-birds` 会保存首次识别每只翼龙时的原始画面。`python dino_cv_official.py --help` 查看全部参数。

## 控制现有 Chrome 窗口

[dino_cv.py](dino_cv.py) 是纯桌面截图版，用 `mss` 读取屏幕和 `PyAutoGUI` 发送按键。该模式需要 macOS 的 **屏幕录制**、**辅助功能**权限，以及 Chrome 窗口保持前台。它受窗口位置和焦点影响；要稳定运行官方游戏，推荐上面的浏览器截图版。

[dino_selenium.py](dino_selenium.py) 是旧的游戏状态读取版，不属于 CV 实现。
