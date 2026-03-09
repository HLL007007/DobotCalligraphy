这是一份为您量身定制的 `README.md` 文件。根据您提供的代码和项目结构，我为您总结了项目的核心功能、运行环境、使用方法和注意事项。您可以直接将其复制并保存为 `README.md`。

***

# 🖋️ Dobot Magician 书法机器人 (Dobot Calligraphy)

本项目基于越疆科技 (Dobot) Magician 桌面级机械臂，实现了一套**高精度、高拟真**的中国书法书写算法。区别于简单的 2D 轨迹描边，本算法内建了**笔触物理引擎**与**结构修正模型**，通过机械臂空间坐标与 Z 轴深度的动态插补，真实还原人类书写时的运笔发力过程。

## ✨ 核心特性 (Features)

本项目包含两套核心算法 (`calligraphy.py` 与 `mycalligraphy.py`)，具备以下高级书写特性：

*   **🌊 动态压感 (Z轴呼吸)：** 根据笔画行进状态（起、行、收），动态调整 Z 轴下压深度，模拟毛笔铺毫与提按。
*   **🗡️ 笔锋模拟引擎：**
    *   **横画**：方笔切入、逆锋回收。
    *   **撇画**：极速加速、弹射出锋。
    *   **捺画**：一波三折、铺毫顿笔、水平拖拽。
    *   **折画**：顿挫转折、护锋。
    *   **钩画**：深按蓄势、左上飞出。
*   **💨 惯性与虚位模拟：** 运用空间贝塞尔曲线生成空中取势轨迹，利用目标点延长线（虚位算法）欺骗机械臂形成锐利尖锋。
*   **🔧 结构修正补丁：** 自动识别笔画曲率与交点，解决“口”字缺口、封口不严等常见字体排版问题。
*   **📐 逆运动学控制 (仅 `mycalligraphy.py`)：** 引入严谨的 Dobot 物理空间逆运动学 (IK) 算法，直接下发关节角 (Joint) 控制指令，避免笛卡尔空间奇异点，并加入了防爆库的分段指令下发机制。

## 📁 目录结构 (Directory Structure)

```text
📦 DobotCalligraphy
 ┣ 📂 data/                 # 存放汉字笔画的源数据 (需要 .json 格式)
 ┣ 📂 images/               # 存放相关的图片或生成预览
 ┣ 📜 calligraphy.py        # 核心主程序 (基于直线插补的笛卡尔坐标控制)
 ┣ 📜 mycalligraphy.py      # 进阶主程序 (包含逆运动学解算与分段指令防溢出)
 ┣ 📜 test.py               # Dobot 正逆运动学 (FK/IK) 验证测试脚本
 ┣ 📜 DobotControl.py       # Dobot 基础运动控制示例
 ┣ 📜 DobotDllType.py       # Dobot 官方 Python API 封装层
 ┣ 📜 DobotDll.dll 等       # Dobot 底层 C++ 动态链接库文件
 ┗ 📜 README.md             # 项目说明文档
```

## 🛠️ 环境依赖 (Requirements)

1.  **硬件：** Dobot Magician 机械臂（末端需安装毛笔夹具）。
2.  **操作系统：** Windows (因底层依赖 `.dll` 动态库)。
3.  **Python 版本：** 需与提供的 `.dll` 位数一致（通常要求 **Python 64位**）。
4.  **第三方库：** 
    ```bash
    pip install numpy
    ```

## 🚀 快速开始 (Quick Start)

### 1. 准备笔画数据
确保 `data/` 目录下有你想书写的汉字的 `.json` 数据文件（例如 `data/中.json`）。JSON 文件需包含笔画中轴线提取后的坐标点列 `medians`。

### 2. 机械臂物理校准 ⚠️ **非常重要**
在运行程序前，**必须**根据您当前的桌面高度、毛笔长度修改配置！打开 `calligraphy.py` 或 `mycalligraphy.py`，找到 `Config` 类：

```python
class Config:
    COM_PORT = "COM3"         # 修改为您的实际串口号 (在设备管理器中查看)
    BASE_Z   = 16.0           # ⚠️ 基准高度：请通过DobotStudio测出笔尖刚好轻触纸面的 Z 轴高度
    LIFT_Z   = 5.0            # 提笔高度：写完一笔后抬起的高度
    PRESS_Z  = -5.0           # 顿笔深度：最重一笔向下压的深度 (负值代表在 BASE_Z 基础上下降)
    START_X  = 220.0          # 起笔 X 坐标
    START_Y  = 0.0            # 起笔 Y 坐标
```

### 3. 运行程序
确保机械臂已开机并解除急停，运行以下命令：

```bash
# 推荐使用进阶版本 (稳定性更好)
python mycalligraphy.py
```
程序运行后，在终端输入你想书写的汉字（需存在对应的数据文件），机械臂将自动开始书写。

## 🧪 运动学测试 (`test.py`)

如果您修改了机械臂的末端执行器尺寸（例如换了更长的毛笔），可以运行 `test.py` 来验证运动学逆解是否准确：

```bash
python test.py
```
该脚本会输出正解与逆解的推算结果及误差(mm)，确保算法系安全可靠。

## ⚠️ 注意事项 (Notes & Safety)

1. **防碰撞警告：** 首次运行时，强烈建议将纸张垫高或把毛笔拿掉，**悬空运行一次**，观察 `BASE_Z` 和下压深度是否合适，防止扎坏毛笔或损坏机械臂电机。
2. **串口被占用：** 运行 Python 脚本前，请确保已**关闭 DobotStudio** 等其他占用串口的软件。
3. **指令溢出报警：** 书法轨迹生成的密集点位极多，如果机械臂亮红灯或停滞，请使用 `mycalligraphy.py`，其内部实现了 `execute_trajectory` 分段下发逻辑（检测缓存小于 200 条时再下发）。
4. **字迹抖动：** 算法内建了 `jitter_xy` (0.1mm) 和 `noise_z` 来模拟人手轻微震颤产生的墨迹边缘效果。如果觉得机械臂抖动过大，可以在 `BrushEngine` 类中将其设为 0。

## 📄 开源协议 (License)
本项目遵循 [MIT License](LICENSE) 协议 (详见 LICENSE 文件)。

---
*Created with ❤️ for Dobot Calligraphy Art.*