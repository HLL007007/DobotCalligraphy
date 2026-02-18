"""
Dobot Magician 书法算法核心控制程序 (Pro Edition)
功能：基于二阶物理模型(Mass-Spring-Damper)的拟真毛笔控制
更新特性：
  - 引入虚拟毛笔物理层，模拟笔毫的滞后与回弹
  - 使用 Hermite 平滑插值替代线性插值
  - 速度-流体压力耦合控制
"""

import json
import os
import sys
import time
import numpy as np
import DobotDllType as dType

# =================================================================================
# 1. 物理数学库 (Physics Math Library) [NEW]
# =================================================================================
class PhysicsMath:
    """提供高阶插值与物理模拟函数"""
    
    @staticmethod
    def smoothstep(edge0, edge1, x):
        """S形平滑插值 (Hermite Interpolation)"""
        # 归一化 x 到 [0, 1]
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        # 3t^2 - 2t^3 曲线，导数在两端为0，实现平滑过渡
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def smooth_interp(t, t_points, v_points):
        """
        基于 Smoothstep 的多段插值替代 np.interp
        实现非线性的参数过渡，消除机械顿挫感
        """
        if t <= t_points[0]: return v_points[0]
        if t >= t_points[-1]: return v_points[-1]
        
        # 寻找 t 所在的区间
        for i in range(len(t_points) - 1):
            if t_points[i] <= t <= t_points[i+1]:
                # 在区间内进行 S 形插值
                local_t = (t - t_points[i]) / (t_points[i+1] - t_points[i])
                factor = local_t * local_t * (3.0 - 2.0 * local_t)
                return v_points[i] * (1 - factor) + v_points[i+1] * factor
        return v_points[-1]

# =================================================================================
# 2. 虚拟毛笔模型 (Virtual Brush Physics) [NEW]
# =================================================================================
class VirtualBrush:
    """
    二阶弹簧阻尼系统 (Mass-Spring-Damper System)
    输入：机械臂刚性坐标 (Target Z)
    输出：笔毫软体坐标 (Actual Z)
    """
    def __init__(self, stiffness=0.15, damping=0.6, mass=1.0):
        self.k = stiffness  # 刚度系数 (越大越硬，回弹越快)
        self.c = damping    # 阻尼系数 (越大越粘滞，震荡越小)
        self.m = mass       # 虚拟质量
        
        self.pos = 0.0      # 当前实际位置
        self.vel = 0.0      # 当前实际速度
    
    def reset(self, initial_z):
        self.pos = initial_z
        self.vel = 0.0

    def update(self, target_z, dt=0.016):
        """
        计算下一帧的物理位置
        F = ma = -k(x - target) - c*v
        """
        # 弹力 (虎克定律) + 阻尼力
        force = -self.k * (self.pos - target_z) - self.c * self.vel
        acc = force / self.m
        
        # 半隐式欧拉积分 (Semi-implicit Euler)
        self.vel += acc * dt
        self.pos += self.vel * dt
        
        return self.pos

# =================================================================================
# 3. 全局配置中心 (Configuration)
# =================================================================================
class Config:
    # --- 基础设置 ---
    DATA_DIR = "./data"       # 笔画数据路径
    COM_PORT = "COM3"         # 机械臂端口
    
    # --- 空间坐标参数 (单位: mm) ---
    BASE_Z   = 12.0           # 基准高度 (笔尖刚好触碰纸面的高度)
    LIFT_Z   = 5.0            # 提笔高度
    SAFE_Z   = 25.0           # 安全回零高度
    
    # --- 笔触深度参数 ---
    PRESS_Z  = -5.0           # 顿笔深度
    STRK_Z   = -2.0           # 行笔深度
    
    # --- 运动参数 ---
    GLOBAL_SPEED = 40         # 全局速度基准 (mm/s)
    FONT_SCALE   = 0.045      # 字体缩放系数
    
    # --- 物理风格参数 [NEW] ---
    BRUSH_STIFFNESS = 0.4     # 笔毫硬度 (0.1软 - 1.0硬)
    BRUSH_DAMPING   = 0.7     # 笔毫阻尼 (防止过分震荡)

    # --- 排版参数 ---
    START_X      = 220.0
    START_Y      = 0.0
    SPACING      = 50.0

# =================================================================================
# 4. 笔画分析器 (Stroke Analyzer)
# =================================================================================
class StrokeAnalyzer:
    """分析笔画几何特征"""
    
    @staticmethod
    def _resample_stroke(stroke, num_samples=50):
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

    @staticmethod
    def get_curvature_profile(stroke):
        pts = StrokeAnalyzer._resample_stroke(stroke)
        vectors = np.diff(pts, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        curvature = np.abs(angle_diffs)
        return np.convolve(curvature, np.ones(3)/3, mode='same')

    @staticmethod
    def analyze(stroke):
        num_pts = len(stroke)
        if num_pts < 5: return "NORMAL"

        curvatures = StrokeAnalyzer.get_curvature_profile(stroke)
        max_curv_idx = np.argmax(curvatures)
        max_curv_val = curvatures[max_curv_idx]
        t_pivot = max_curv_idx / len(curvatures)
        
        if max_curv_val > 0.7:
            if 0.10 < t_pivot < 0.75: return "ZHE"
            elif t_pivot >= 0.75:     return "GOU"

        p_start, p_end = np.array(stroke[0]), np.array(stroke[-1])
        vec = p_end - p_start
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        
        if -20 < angle < 20:   return "HENG"
        if 20 <= angle < 65:   return "NA"
        if 65 <= angle < 125:  return "SHU"
        if 125 <= angle < 175: return "PIE"
        
        return "NORMAL"

    @staticmethod
    def get_pivot_t(stroke):
        curvatures = StrokeAnalyzer.get_curvature_profile(stroke)
        return np.argmax(curvatures) / len(curvatures)

# =================================================================================
# 5. 笔触物理引擎 (Brush Engine) [UPDATED]
# =================================================================================
class BrushEngine:
    """
    核心类：结合物理模型生成自然轨迹
    """
    def __init__(self, config):
        self.cfg = config
        self.last_pos = None 
        # 初始化物理笔尖对象
        self.virtual_brush = VirtualBrush(stiffness=config.BRUSH_STIFFNESS, damping=config.BRUSH_DAMPING)

    def _generate_bezier_curve(self, start_pt, end_pt, num_points=12):
        """生成空中过渡轨迹 (S形变速)"""
        p1 = {'x': start_pt['x'], 'y': start_pt['y'], 'z': start_pt['z'] + 10}
        p2 = {'x': end_pt['x'], 'y': end_pt['y'], 'z': end_pt['z'] + 2.0}
        curve = []
        for t_lin in np.linspace(0, 1, num_points):
            # 使用 smoothstep 优化空中移动速度，起落更轻柔
            t = PhysicsMath.smoothstep(0, 1, t_lin)
            it = 1 - t
            bx = it**3*start_pt['x'] + 3*t*it**2*p1['x'] + 3*t**2*it*p2['x'] + t**3*end_pt['x']
            by = it**3*start_pt['y'] + 3*t*it**2*p1['y'] + 3*t**2*it*p2['y'] + t**3*end_pt['y']
            bz = it**3*start_pt['z'] + 3*t*it**2*p1['z'] + 3*t**2*it*p2['z'] + t**3*end_pt['z']
            curve.append({'x': bx, 'y': by, 'z': bz, 'pause': 0, 'velocity': self.cfg.GLOBAL_SPEED})
        return curve

    def generate_waypoints(self, medians, offset_x, offset_y):
        all_waypoints = []
        self.last_pos = None 

        for stroke in medians:
            num_pts = len(stroke)
            if num_pts < 2: continue
            
            # 1. 分析特征
            s_type = StrokeAnalyzer.analyze(stroke)
            t_pivot = StrokeAnalyzer.get_pivot_t(stroke) if s_type in ["ZHE", "GOU"] else 0.5
            
            # 2. 计算向量
            p_start, p_end = np.array(stroke[0]), np.array(stroke[-1])
            vec = p_end - p_start
            if s_type == "ZHE":
                pivot_idx = int(t_pivot * (num_pts - 1))
                if pivot_idx < num_pts - 1:
                    vec = p_end - np.array(stroke[pivot_idx])
            vec_len = np.linalg.norm(vec)
            unit_vec = vec / vec_len if vec_len > 0 else np.array([0, 0])

            # 3. 空中移动
            s_raw = stroke[0]
            tgt_x = offset_x + (1024 - s_raw[1]) * self.cfg.FONT_SCALE
            tgt_y = offset_y + s_raw[0] * self.cfg.FONT_SCALE
            tgt_z = self.cfg.BASE_Z + self.cfg.LIFT_Z 
            
            if self.last_pos:
                all_waypoints.extend(self._generate_bezier_curve(self.last_pos, {'x': tgt_x, 'y': tgt_y, 'z': tgt_z}))
                self.virtual_brush.reset(tgt_z)
            else:
                self.virtual_brush.reset(tgt_z)

            # 4. 笔画轨迹计算 (两阶段：几何生成 -> 物理模拟)
            sample_count = 60 
            
            # --- 阶段A: 几何轨迹预计算 (Structure Layer) ---
            geo_trajectory = []
            for i in range(sample_count):
                t = i / (sample_count - 1)
                idx = t * (num_pts - 1)
                idx_f, idx_c = int(np.floor(idx)), int(np.ceil(idx))
                alpha = idx - idx_f
                curr_p = np.array(stroke[idx_f]) * (1-alpha) + np.array(stroke[idx_c]) * alpha
                
                # 结构修正：折画延长
                if s_type == "ZHE" and t > 0.8:
                    extend_len = 50.0 * (t - 0.8) / 0.2 / self.cfg.FONT_SCALE
                    curr_p += unit_vec * extend_len
                
                geo_trajectory.append(curr_p)

            # --- 阶段B: 物理模拟与样式应用 (Physics Layer) ---
            for i in range(sample_count):
                t = i / (sample_count - 1)
                curr_p = geo_trajectory[i]

                # 初始化目标参数
                target_z = self.cfg.BASE_Z + self.cfg.STRK_Z
                pause = 0
                v_ratio = 1.0
                
                # ------------------- 笔法逻辑 (使用 Smooth Interp) -------------------
                
                # === [横 HENG] ===
                if s_type == "HENG":
                    if t < 0.15: # 逆锋切入
                        target_z = PhysicsMath.smooth_interp(t, [0, 0.15], [self.cfg.BASE_Z + 3.0, self.cfg.BASE_Z + self.cfg.PRESS_Z])
                        v_ratio = PhysicsMath.smooth_interp(t, [0, 0.15], [0.8, 0.4])
                    elif t > 0.85: # 顿笔回锋
                        if t <= 0.92:
                            target_z = PhysicsMath.smooth_interp(t, [0.85, 0.92], [self.cfg.BASE_Z + self.cfg.STRK_Z, self.cfg.BASE_Z + self.cfg.PRESS_Z])
                            v_ratio = 0.2
                            if t >= 0.9: pause = 0.15
                        else:
                            target_z = PhysicsMath.smooth_interp(t, [0.92, 1.0], [self.cfg.BASE_Z + self.cfg.PRESS_Z, self.cfg.BASE_Z + self.cfg.LIFT_Z])
                            curr_p -= unit_vec * (0.6 * (t - 0.92) / 0.08 / self.cfg.FONT_SCALE)
                            v_ratio = 0.45
                    else:
                        # 中段下垂模拟重力
                        arch = 0.5 * np.sin(t * np.pi) 
                        target_z = self.cfg.BASE_Z + self.cfg.STRK_Z + 0.5 - arch
                        v_ratio = 0.75

                # === [撇 PIE] ===
                elif s_type == "PIE":
                    if t < 0.2:
                        target_z = PhysicsMath.smooth_interp(t, [0, 0.2], [self.cfg.BASE_Z+self.cfg.LIFT_Z, self.cfg.BASE_Z+self.cfg.STRK_Z])
                        v_ratio = 0.5
                    else:
                        # 抛物线加速 (甩笔)
                        v_ratio = 0.6 + 2.5 * (t * t) 
                        if t < 0.7: 
                            target_z = self.cfg.BASE_Z + self.cfg.STRK_Z
                        else:
                            # 抬笔并冲出虚位
                            target_z = PhysicsMath.smooth_interp(t, [0.7, 1.0], [self.cfg.BASE_Z + self.cfg.STRK_Z, self.cfg.BASE_Z + self.cfg.LIFT_Z])
                            curr_p += unit_vec * (6.0 * (t - 0.7) / 0.3 / self.cfg.FONT_SCALE)

                # === [捺 NA] ===
                elif s_type == "NA":
                    if t < 0.15:
                         target_z = PhysicsMath.smooth_interp(t, [0, 0.15], [self.cfg.BASE_Z+self.cfg.LIFT_Z, self.cfg.BASE_Z+self.cfg.STRK_Z])
                    elif t > 0.75:
                        if t <= 0.9: # 铺毫
                            target_z = PhysicsMath.smooth_interp(t, [0.75, 0.9], [self.cfg.BASE_Z+self.cfg.STRK_Z, self.cfg.BASE_Z + self.cfg.PRESS_Z + 1.0])
                            v_ratio = 0.2
                            # 震颤模拟笔毫散开
                            curr_p[0] += np.random.uniform(-0.2, 0.2) / self.cfg.FONT_SCALE
                            if t >= 0.88: pause = 0.2
                        else: # 缓慢出锋
                            target_z = PhysicsMath.smooth_interp(t, [0.9, 1.0], [self.cfg.BASE_Z + self.cfg.PRESS_Z + 1.0, self.cfg.BASE_Z+self.cfg.LIFT_Z])
                            curr_p += unit_vec * (4.0 * (t - 0.9) / 0.1 / self.cfg.FONT_SCALE)
                    else:
                        # 波折
                        wave = 0.8 * np.sin(t * np.pi * 1.5) 
                        target_z = self.cfg.BASE_Z + self.cfg.STRK_Z + wave
                        v_ratio = 0.6

                # === [钩 GOU] ===
                elif s_type == "GOU":
                    if abs(t - t_pivot) <= 0.05:
                        target_z = self.cfg.BASE_Z + self.cfg.PRESS_Z - 0.5
                        pause, v_ratio = 0.15, 0.1
                    elif t > t_pivot:
                        rescale_t = (t - t_pivot) / (1.0 - t_pivot)
                        if rescale_t < 0.4:
                            # 蓄势阶段
                            target_z = self.cfg.BASE_Z + self.cfg.PRESS_Z
                            v_ratio = 0.15
                        else:
                            # 弹射阶段
                            flick_t = PhysicsMath.smoothstep(0.4, 1.0, rescale_t)
                            target_z = (self.cfg.BASE_Z + self.cfg.PRESS_Z) + (self.cfg.LIFT_Z - self.cfg.PRESS_Z) * flick_t
                            v_ratio = 1.0 + flick_t * 5.0 
                            offset_mag = 5.0 * flick_t / self.cfg.FONT_SCALE
                            curr_p += np.array([-0.8, -1.0]) * offset_mag
                
                else:
                    # 默认笔画
                    if t < 0.1: target_z = PhysicsMath.smooth_interp(t, [0, 0.1], [self.cfg.BASE_Z+self.cfg.LIFT_Z, self.cfg.BASE_Z+self.cfg.STRK_Z])
                    elif t > 0.9: target_z = PhysicsMath.smooth_interp(t, [0.9, 1.0], [self.cfg.BASE_Z+self.cfg.STRK_Z, self.cfg.BASE_Z+self.cfg.LIFT_Z])
                    else: target_z = self.cfg.BASE_Z + self.cfg.STRK_Z

                # ------------------- 物理滤波核心 (Physics Filter) -------------------
                
                # 1. 速度-压力耦合修正 (流体力学效应)
                # 速度越快，笔尖受流体升力越大，Z轴自动略微抬起，使快笔画更尖锐
                lift_correction = 0.5 * (v_ratio - 1.0) if v_ratio > 1.0 else 0
                target_z += lift_correction

                # 2. 弹簧阻尼系统计算
                # 将机械臂的目标位置(target_z)转化为软笔尖的实际位置(actual_z)
                actual_z = self.virtual_brush.update(target_z, dt=0.02)
                
                # 3. 最终坐标变换
                rx = offset_x + (1024 - curr_p[1]) * self.cfg.FONT_SCALE
                ry = offset_y + curr_p[0] * self.cfg.FONT_SCALE
                
                # 4. 微量随机纹理
                rx += np.random.uniform(-0.05, 0.05)
                ry += np.random.uniform(-0.05, 0.05)

                all_waypoints.append({
                    'x': rx, 'y': ry, 
                    'z': actual_z,
                    'pause': pause,
                    'velocity': self.cfg.GLOBAL_SPEED * v_ratio
                })
            
            self.last_pos = all_waypoints[-1]
            
        return all_waypoints

# =================================================================================
# 6. 硬件驱动层 (Hardware Driver)
# =================================================================================
class DobotDriver:
    """处理与 Dobot Magician 的底层通信"""
    def __init__(self, port):
        self.api = dType.load()
        self.port = port
        self.connected = False

    def connect(self):
        state = dType.ConnectDobot(self.api, self.port, 115200)[0]
        if state == dType.DobotConnect.DobotConnect_NoError:
            print(f">>> 机械臂连接成功 [{self.port}]")
            dType.SetQueuedCmdClear(self.api)
            # 设置 PTP 运动参数
            dType.SetPTPCommonParams(self.api, Config.GLOBAL_SPEED, Config.GLOBAL_SPEED, isQueued=1)
            self.connected = True
            return True
        print(f"!!! 连接失败: 无法连接到 {self.port}")
        return False

    def execute(self, waypoints):
        if not self.connected: return 0
        cmd_id = 0
        for wp in waypoints:
            # 动态调整速度
            dType.SetPTPCommonParams(self.api, wp['velocity'], wp['velocity'], isQueued=1)
            # PTPMOVLXYZMode: 直线运动
            res = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, wp['x'], wp['y'], wp['z'], 0, isQueued=1)
            cmd_id = res[0]
            if wp.get('pause', 0) > 0:
                # 插入驻笔等待
                cmd_id = dType.SetWAITCmd(self.api, int(wp['pause'] * 1000), isQueued=1)[0]
        return cmd_id

    def wait_finish(self, last_id):
        """阻塞等待直到指令队列执行完毕"""
        try:
            while True:
                curr_id = dType.GetQueuedCmdCurrentIndex(self.api)[0]
                if curr_id >= last_id: break
                time.sleep(0.1)
        except KeyboardInterrupt:
            dType.SetQueuedCmdStopExec(self.api)
            print("\n!!! 紧急停止 !!!")

    def close(self):
        dType.DisconnectDobot(self.api)
        print(">>> 机械臂连接已断开")

# =================================================================================
# 7. 主程序入口 (Main)
# =================================================================================
def main():
    # 1. 初始化驱动
    driver = DobotDriver(Config.COM_PORT)
    if not driver.connect(): return
    
    # 2. 初始化引擎
    engine = BrushEngine(Config)
    
    # 3. 获取输入
    text = input("请输入汉字 (确保data目录下有对应的json文件): ")
    
    full_trajectory = []
    
    # 4. 生成轨迹
    for index, char in enumerate(text):
        path = os.path.join(Config.DATA_DIR, f"{char}.json")
        if os.path.exists(path):
            print(f"正在处理字符: {char} ...")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    medians = json.load(f).get("medians", [])
                
                # 计算字距偏移
                off_y = Config.START_Y + (index * Config.SPACING)
                # 生成单个字的轨迹
                full_trajectory.extend(engine.generate_waypoints(medians, Config.START_X, off_y))
            except Exception as e:
                print(f"读取或解析 {char}.json 失败: {e}")
        else:
            print(f"未找到笔画数据: {path}")

    # 5. 发送指令
    if full_trajectory:
        print(f">>> 生成指令数: {len(full_trajectory)}，开始传输...")
        dType.SetQueuedCmdStartExec(driver.api) 
        last_id = driver.execute(full_trajectory)
        
        # 结束后抬笔归位
        f_pt = full_trajectory[-1]
        last_id = dType.SetPTPCmd(driver.api, dType.PTPMode.PTPMOVLXYZMode, 
                                 f_pt['x'], f_pt['y'], Config.BASE_Z + Config.SAFE_Z, 0, isQueued=1)[0]
        
        print(">>> 正在书写，请勿触碰机械臂...")
        driver.wait_finish(last_id)
    
    driver.close()
    print(">>> 任务圆满完成")

if __name__ == "__main__":
    main()