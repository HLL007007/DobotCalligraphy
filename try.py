# -*- coding: utf-8 -*-
"""
Dobot Magician 书法算法 (Ultimate Fusion Edition - 终极融合版)
杀手锏：JSON 完美笔顺 + TTF 视觉动态压感 (时空融合)
"""
import time
import math
import os
import json
import cv2
import numpy as np
import DobotDllType as dType
from PIL import Image, ImageDraw, ImageFont

# =================================================================================
# 1. 全局配置中心
# =================================================================================
class Config:
    COM_PORT = "COM3"         
    DATA_DIR = "./data"       # JSON 完美笔画库
    FONT_FILE = "FZChuSLKSJW.TTF" # 提取粗细的真实字体库
    
    # --- 空间坐标参数 (单位: mm) ---
    BASE_Z   = -3.0          # 纸面高度
    LIFT_Z   = 5.0            # 抬笔高度
    SAFE_Z   = 25.0           # 安全高度
    
    # --- 笔触物理参数 ---
    PRESS_Z  = -3.0           # 顿笔最大深度 (可根据你的墨水和毛笔微调: -2.0 到 -4.0)
    
    # --- 运动与排版参数 ---
    GLOBAL_SPEED = 45        # 行笔速度 (适中，防止洇墨)
    FONT_SCALE   = 0.04      # 字号缩放 (约 6x6 厘米的大字，展现细节)
    START_X      = 200.0      
    START_Y      = 0.0        
    SPACING      = 70.0       # 字间距
    USE_JOINT_MODE = True     

# =================================================================================
# 2. 终极融合引擎 (Data Fusion Engine)
# =================================================================================
class FusionEngine:
    def __init__(self, config):
        self.cfg = config
        self.canvas_size = 1024 # 与 JSON 坐标系严格对齐

    def _get_width_map(self, char):
        """生成物理宽度场 (Distance Map)"""
        # 1. 渲染文字 (留出边界，防止字贴边)
        font = ImageFont.truetype(self.cfg.FONT_FILE, int(self.canvas_size * 0.8))
        img = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        draw = ImageDraw.Draw(img)
        
        # 居中对齐
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (self.canvas_size - (bbox[2] - bbox[0])) / 2 - bbox[0]
        y = (self.canvas_size - (bbox[3] - bbox[1])) / 2 - bbox[1]
        draw.text((x, y), char, font=font, fill=255)
        
        img_array = np.array(img)
        
        # 2. 计算宽度场 (点到边缘的距离)
        dist_transform = cv2.distanceTransform(img_array, cv2.DIST_L2, 5)
        return dist_transform

    def _resample_stroke(self, stroke, num_samples=60):
        """将 JSON 轨迹重采样为均匀的密集点阵，保证机械臂运行顺滑"""
        pts = np.array(stroke)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        cumulative_dist = np.concatenate(([0], np.cumsum(dist)))
        total_dist = cumulative_dist[-1]
        if total_dist == 0: return pts
        
        samples = np.linspace(0, total_dist, num_samples)
        new_pts = np.zeros((num_samples, 2))
        for i in range(2): 
            new_pts[:, i] = np.interp(samples, cumulative_dist, pts[:, i])
        return new_pts

    def generate_waypoints(self, char, offset_x, offset_y):
        """融合生成最终三维轨迹"""
        # 1. 读取 JSON 灵魂 (轨迹与笔顺)
        json_path = os.path.join(self.cfg.DATA_DIR, f"{char}.json")
        if not os.path.exists(json_path):
            print(f"找不到 {char} 的 JSON 数据，跳过...")
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            medians = json.load(f).get("medians", [])

        # 2. 读取 TTF 血肉 (真实物理粗细)
        width_map = self._get_width_map(char)
        max_global_width = np.max(width_map)
        if max_global_width == 0: max_global_width = 1.0

        all_waypoints = []
        
        for stroke in medians:
            if len(stroke) < 2: continue
            
            # 轨迹重采样，确保速度均匀
            smooth_stroke = self._resample_stroke(stroke, num_samples=50)
            
            # --- Z 轴深度查表与计算 ---
            z_array = []
            for pt in smooth_stroke:
                # 坐标系对齐 (防止越界)
                px = int(np.clip(pt[1], 0, self.canvas_size - 1))
                py = int(np.clip(pt[0], 0, self.canvas_size - 1))
                
                # 杀手锏：局部宽容搜索 (Local Max Search)
                # 因为 JSON 和 字体可能不是完全 100% 对齐，我们在周围 20x20 像素内找最粗的点
                y_min, y_max = max(0, py-10), min(self.canvas_size, py+10)
                x_min, x_max = max(0, px-10), min(self.canvas_size, px+10)
                local_w = np.max(width_map[y_min:y_max, x_min:x_max])
                
                # 宽度映射为 Z 轴深度 (采用 1.5 次方，让笔锋更锐利)
                normalized_w = local_w / max_global_width
                z = self.cfg.BASE_Z - abs(self.cfg.PRESS_Z) * (normalized_w ** 1.5)
                z_array.append(z)
                
            # 一维高斯平滑 (确保 Z 轴电机不会因为图像噪点而剧烈抖动)
            z_array = np.convolve(z_array, np.ones(5)/5, mode='same')
            
            # --- 生成物理运动指令 ---
            # 1. 空中移至起笔点
            start_pt = smooth_stroke[0]
            rx = offset_x + (self.canvas_size - start_pt[1]) * self.cfg.FONT_SCALE
            ry = offset_y + start_pt[0] * self.cfg.FONT_SCALE
            all_waypoints.append({'x': rx, 'y': ry, 'z': self.cfg.BASE_Z + self.cfg.LIFT_Z, 'pause': 0, 'velocity': self.cfg.GLOBAL_SPEED})
            
            # 2. 落下行笔 (携带动态 Z 轴)
            for i, pt in enumerate(smooth_stroke):
                rx = offset_x + (self.canvas_size - pt[1]) * self.cfg.FONT_SCALE
                ry = offset_y + pt[0] * self.cfg.FONT_SCALE
                all_waypoints.append({'x': rx, 'y': ry, 'z': z_array[i], 'pause': 0, 'velocity': self.cfg.GLOBAL_SPEED * 0.8})
                
            # 3. 提笔收锋
            last_pt = all_waypoints[-1]
            all_waypoints.append({'x': last_pt['x'], 'y': last_pt['y'], 'z': self.cfg.BASE_Z + self.cfg.LIFT_Z, 'pause': 0, 'velocity': self.cfg.GLOBAL_SPEED})
            
        return all_waypoints

# =================================================================================
# 3. 逆运动学 (IK) 与 硬件驱动 (Driver) 保持原样，极其稳定
# =================================================================================
class DobotInverseKinematics:
    def __init__(self):
        self.L2, self.L3, self.TOOL_X = 135.0, 147.0, 61.0

    def cartesian_to_joint(self, x, y, z):
        j1 = math.degrees(math.atan2(y, x))
        r_target = math.sqrt(x**2 + y**2) - self.TOOL_X
        dist_sq = r_target**2 + z**2
        dist = math.sqrt(dist_sq)
        if dist > (self.L2 + self.L3) or dist < abs(self.L2 - self.L3):
            raise ValueError("目标坐标超出物理触达范围")
        cos_alpha = (self.L2**2 + dist_sq - self.L3**2) / (2 * self.L2 * dist)
        alpha = math.degrees(math.acos(max(-1, min(1, cos_alpha))))
        cos_gamma = (self.L2**2 + self.L3**2 - dist_sq) / (2 * self.L2 * self.L3)
        gamma = math.degrees(math.acos(max(-1, min(1, cos_gamma))))
        phi_from_z = math.degrees(math.atan2(r_target, z))
        j2 = phi_from_z - alpha
        j3 = 90.0 + j2 - gamma
        
        if not (-90 <= j1 <= 90): raise ValueError(f"J1超出限位")
        if not (0 <= j2 <= 85): raise ValueError(f"J2超出限位")
        if not (-10 <= j3 <= 95): raise ValueError(f"J3超出限位")
        return [round(j1, 4), round(j2, 4), round(j3, 4), 0.0]

class DobotDriver:
    def __init__(self, port):
        self.api = dType.load()
        self.port = port
        self.connected = False

    def connect(self):
        state = dType.ConnectDobot(self.api, self.port, 115200)[0]
        if state == dType.DobotConnect.DobotConnect_NoError:
            print(f">>> 机械臂连接成功 [{self.port}]")
            dType.SetQueuedCmdClear(self.api)
            self.connected = True
            return True
        return False

    def execute_trajectory(self, waypoints):
        if not self.connected: return 0
        ik = DobotInverseKinematics()
        last_id = 0
        for i, pt in enumerate(waypoints):
            if i % 10 == 0:
                while True:
                    curr_id = dType.GetQueuedCmdCurrentIndex(self.api)[0]
                    if (last_id - curr_id) < 200: break
                    time.sleep(0.05)
            try:
                joints = ik.cartesian_to_joint(pt['x'], pt['y'], pt['z'])
                dType.SetPTPCommonParams(self.api, pt['velocity'], pt['velocity'], isQueued=1)
                last_id = dType.SetPTPCmd(self.api, 5, joints[0], joints[1], joints[2], joints[3], isQueued=1)[0]
            except ValueError as e:
                continue
        return last_id

    def wait_finish(self, last_id):
        try:
            while True:
                if dType.GetQueuedCmdCurrentIndex(self.api)[0] >= last_id: break
        except KeyboardInterrupt:
            dType.SetQueuedCmdStopExec(self.api)

    def close(self):
        dType.DisconnectDobot(self.api)
        print(">>> 连接已断开")

# =================================================================================
# 4. 主程序入口
# =================================================================================
def main():
    driver = DobotDriver(Config.COM_PORT)
    if not driver.connect(): return
    
    engine = FusionEngine(Config)
    text = input("请输入要写的汉字(JSON笔顺 + TTF压感融合): ")
    full_trajectory = []
    
    for index, char in enumerate(text):
        print(f"正在抽取 {char} 的JSON骨架，并查表注入TTF压感...")
        off_y = Config.START_Y + (index * Config.SPACING)
        waypoints = engine.generate_waypoints(char, Config.START_X, off_y)
        full_trajectory.extend(waypoints)

    if full_trajectory:
        print(f">>> 开始执行大师级书法任务，总点数: {len(full_trajectory)}")
        f_pt = full_trajectory[0]
        dType.SetPTPCmd(driver.api, 1, f_pt['x'], f_pt['y'], Config.BASE_Z + Config.SAFE_Z, 0, isQueued=0)
        dType.SetQueuedCmdStartExec(driver.api) 
        last_id = driver.execute_trajectory(full_trajectory)
        dType.SetPTPCmd(driver.api, 1, f_pt['x'], f_pt['y'], Config.BASE_Z + Config.SAFE_Z, 0, isQueued=1)
        
        print(">>> 指令已全部传输，等待神作诞生...")
        driver.wait_finish(last_id)
        dType.SetQueuedCmdStopExec(driver.api)
    
    driver.close()
    print(">>> 任务圆满完成")

if __name__ == "__main__":
    main()