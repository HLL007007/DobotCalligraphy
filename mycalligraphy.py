# -*- coding: utf-8 -*-
"""
Dobot Magician 书法算法核心控制程序 (Master Edition)
功能：基于笔锋物理模型的轨迹生成与机械臂控制
包含特性：
  - 动态压感 (Z轴呼吸)
  - 笔锋模拟 (切锋、回锋、护锋)
  - 惯性模拟 (空中虚位、弹射出锋)
  - 结构修正 (自动封口、垂露)
  - 完美官方逆运动学映射 (100% 精准坐标)
"""
import time
import math
import json
import os
import sys
import numpy as np
import DobotDllType as dType

# =================================================================================
# 1. 全局配置中心 (Configuration)
# =================================================================================
class Config:
    # --- 基础设置 ---
    DATA_DIR = "./data"       # 笔画数据路径
    COM_PORT = "COM3"         # 机械臂端口
    
    # --- 空间坐标参数 (单位: mm) ---
    BASE_Z   = 0.0            # 基准高度 (笔尖刚好触碰纸面的高度，需根据实际校准)
    LIFT_Z   = 5.0            # 提笔高度 (空中移动的高度)
    SAFE_Z   = 25.0           # 安全回零高度
    
    # --- 笔触深度参数 (相对于 BASE_Z) ---
    # 注意：负值越大，下压越深
    PRESS_Z  = -5.0           # 顿笔深度 (最重，用于捺脚、转折)
    STRK_Z   = -2.0           # 行笔深度 (标准，用于横竖行进)
    
    # --- 运动参数 ---
    GLOBAL_SPEED = 40         # 全局速度基准 (mm/s)
    FONT_SCALE   = 0.030      # 字体缩放系数 (将 1024x1024 坐标映射到机械臂空间)
    
    # --- 排版参数 ---
    START_X      = 200.0      # 起始点 X 坐标
    START_Y      = 0.0        # 起始点 Y 坐标
    SPACING      = 30.0       # 字间距

    USE_JOINT_MODE = True  # 开启关节空间控制模式
    
# =================================================================================
# 2. 笔画分析器 (Stroke Analyzer)
# =================================================================================
class StrokeAnalyzer:
    """
    负责对笔画原始数据进行几何分析，识别笔画类型（横、竖、撇、捺、折、钩）
    """
    
    @staticmethod
    def _resample_stroke(stroke, num_samples=50):
        """数据重采样：将疏密不均的点转换为等间距点"""
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
        """计算曲率谱：分析笔画中哪一点弯曲最剧烈"""
        pts = StrokeAnalyzer._resample_stroke(stroke)
        vectors = np.diff(pts, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.diff(angles)
        # 处理 -pi 到 pi 的跳变
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        curvature = np.abs(angle_diffs)
        # 简单平滑处理
        return np.convolve(curvature, np.ones(3)/3, mode='same')

    @staticmethod
    def analyze(stroke):
        """核心分类逻辑"""
        num_pts = len(stroke)
        if num_pts < 5: return "NORMAL"

        # 1. 基于曲率判断复杂笔画 (折、钩)
        curvatures = StrokeAnalyzer.get_curvature_profile(stroke)
        max_curv_idx = np.argmax(curvatures)
        max_curv_val = curvatures[max_curv_idx]
        t_pivot = max_curv_idx / len(curvatures)
        
        if max_curv_val > 0.7:
            if 0.10 < t_pivot < 0.75: return "ZHE" # 中间转折 -> 折
            elif t_pivot >= 0.75:     return "GOU" # 尾部转折 -> 钩

        # 2. 基于首尾角度判断基础笔画
        p_start, p_end = np.array(stroke[0]), np.array(stroke[-1])
        vec = p_end - p_start
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        
        if -20 < angle < 20:   return "HENG" # 横
        if 20 <= angle < 65:   return "NA"   # 捺
        if 65 <= angle < 125:  return "SHU"  # 竖
        if 125 <= angle < 175: return "PIE"  # 撇
        
        return "NORMAL"

    @staticmethod
    def get_pivot_t(stroke):
        """获取转折点在笔画中的百分比位置 (0.0 - 1.0)"""
        curvatures = StrokeAnalyzer.get_curvature_profile(stroke)
        return np.argmax(curvatures) / len(curvatures)

# =================================================================================
# 3. 笔触物理引擎 (Brush Physics Engine)
# =================================================================================
class BrushEngine:
    """
    核心类：将几何坐标转换为机械臂的运动轨迹 (Waypoints)
    包含所有书法技法的物理模拟：起笔切锋、行笔涩势、收笔回锋等
    """
    def __init__(self, config):
        self.cfg = config
        self.last_pos = None 

    def _generate_bezier_curve(self, start_pt, end_pt, num_points=12):
        """生成空中过渡轨迹 (模拟手腕的空中取势)"""
        p1 = {'x': start_pt['x'], 'y': start_pt['y'], 'z': start_pt['z'] + 10}
        p2 = {'x': end_pt['x'], 'y': end_pt['y'], 'z': end_pt['z'] + 2.0}
        curve = []
        for t_lin in np.linspace(0, 1, num_points):
            t = 1 - (1 - t_lin)**2
            it = 1 - t
            # 三阶贝塞尔插值
            bx = it**3*start_pt['x'] + 3*t*it**2*p1['x'] + 3*t**2*it*p2['x'] + t**3*end_pt['x']
            by = it**3*start_pt['y'] + 3*t*it**2*p1['y'] + 3*t**2*it*p2['y'] + t**3*end_pt['y']
            bz = it**3*start_pt['z'] + 3*t*it**2*p1['z'] + 3*t**2*it*p2['z'] + t**3*end_pt['z']
            curve.append({'x': bx, 'y': by, 'z': bz, 'pause': 0, 'velocity': self.cfg.GLOBAL_SPEED})
        return curve

    def generate_waypoints(self, medians, offset_x, offset_y):
        """主生成函数：输入笔画数据，输出机械臂控制点列表"""
        all_waypoints = []
        self.last_pos = None 

        for stroke in medians:
            num_pts = len(stroke)
            if num_pts < 2: continue
            
            # 1. 分析笔画特征
            s_type = StrokeAnalyzer.analyze(stroke)
            t_pivot = StrokeAnalyzer.get_pivot_t(stroke) if s_type in ["ZHE", "GOU"] else 0.5
            
            # 2. 计算方向向量 (用于虚位算法与回锋)
            p_start, p_end = np.array(stroke[0]), np.array(stroke[-1])
            vec = p_end - p_start
            
            # 特殊处理：折画取后半段向量，确保垂直延长
            if s_type == "ZHE":
                pivot_idx = int(t_pivot * (num_pts - 1))
                if pivot_idx < num_pts - 1:
                    vec = p_end - np.array(stroke[pivot_idx])
            
            vec_len = np.linalg.norm(vec)
            unit_vec = vec / vec_len if vec_len > 0 else np.array([0, 0])

            # 3. 生成空中移动轨迹 (移动到起笔点上方)
            s_raw = stroke[0]
            tgt_x = offset_x + (1024 - s_raw[1]) * self.cfg.FONT_SCALE
            tgt_y = offset_y + s_raw[0] * self.cfg.FONT_SCALE
            tgt_z = self.cfg.BASE_Z + self.cfg.LIFT_Z 
            
            if self.last_pos:
                all_waypoints.extend(self._generate_bezier_curve(self.last_pos, {'x': tgt_x, 'y': tgt_y, 'z': tgt_z}))
            
            # 4. 笔画点插值生成 (Sampling)
            sample_count = 60 
            for i in range(sample_count):
                t = i / (sample_count - 1) # 进度 0.0 -> 1.0
                
                # --- 基础插值坐标 ---
                idx = t * (num_pts - 1)
                idx_f, idx_c = int(np.floor(idx)), int(np.ceil(idx))
                alpha = idx - idx_f
                curr_p = np.array(stroke[idx_f]) * (1-alpha) + np.array(stroke[idx_c]) * alpha
                
                # =========================================================
                # 结构修正层 (Structure Patch)
                # 解决“口”字缺口、封口不严等结构问题
                # =========================================================
                
                # 折画 (ZHE) 右竖：大幅垂直延长 50mm，确保底横能接住
                if s_type == "ZHE" and t > 0.8:
                    extend_len = 50.0 * (t - 0.8) / 0.2 / self.cfg.FONT_SCALE
                    curr_p += unit_vec * extend_len
                
                # =========================================================
                # 物理模拟层 (Physics Simulation)
                # =========================================================
                
                # 初始化变量 (防止未定义错误)
                z = self.cfg.BASE_Z + self.cfg.STRK_Z
                pause = 0
                v_ratio = 1.0 # 速度倍率
                
                # 全局随机扰动 (模拟手部肌肉震颤与纸张纹理)
                noise_z = np.random.uniform(-0.15, 0.15)
                jitter_xy = 0.1

                # ------------------- 笔画特异性逻辑 -------------------
                
                # === [横 HENG]：方笔切入，逆锋回收 ===
                if s_type == "HENG":
                    if t < 0.15: # 起笔
                        # 空中切入：从 +3.0mm 快速切到 PRESS_Z 附近
                        z = np.interp(t, [0, 0.15], [self.cfg.BASE_Z + 3.0, self.cfg.BASE_Z + self.cfg.PRESS_Z + 1.0])
                        v_ratio = np.interp(t, [0, 0.15], [0.8, 0.4]) 
                    elif t > 0.85: # 收笔
                        if t <= 0.92: # 顿笔
                            z = np.interp(t, [0.85, 0.92], [self.cfg.BASE_Z + self.cfg.STRK_Z + 1.5, self.cfg.BASE_Z + self.cfg.PRESS_Z])
                            v_ratio = 0.2 
                            if t >= 0.9: pause = 0.15 
                        else: # 回锋
                            z = np.interp(t, [0.92, 1.0], [self.cfg.BASE_Z + self.cfg.PRESS_Z, self.cfg.BASE_Z + self.cfg.LIFT_Z])
                            # 坐标回缩算法：逆向量移动 0.6mm
                            recoil_dist = 0.6 * (t - 0.92) / 0.08 / self.cfg.FONT_SCALE
                            curr_p -= unit_vec * recoil_dist 
                            v_ratio = 0.45
                    else: # 行笔
                        z = self.cfg.BASE_Z + self.cfg.STRK_Z + 1.0 #抬起一些，防止过于沉重
                        v_ratio = 0.75

                # === [撇 PIE]：空中虚位，弹射出锋 ===
                elif s_type == "PIE":
                    if t < 0.2: 
                        z = np.interp(t, [0, 0.2], [self.cfg.BASE_Z+self.cfg.LIFT_Z, self.cfg.BASE_Z+self.cfg.STRK_Z])
                        v_ratio = 0.5 
                    else: 
                        # 后段快速提起
                        if t < 0.8: z = self.cfg.BASE_Z + self.cfg.STRK_Z
                        else:       z = np.interp(t, [0.8, 1.0], [self.cfg.BASE_Z + self.cfg.STRK_Z, self.cfg.BASE_Z + self.cfg.LIFT_Z + 2.0])
                        
                        # 虚位算法：目标点延伸到空中 6.0mm 处
                        # 欺骗机械臂冲向延长线，形成锐利尖锋
                        if t > 0.7:
                            virtual_dist = 6.0 * (t - 0.7) / 0.3 / self.cfg.FONT_SCALE
                            curr_p += unit_vec * virtual_dist
                        
                        # 极速加速
                        v_ratio = 0.6 + 1.8 * ((t-0.2)/0.8)**2 

                # === [捺 NA]：铺毫顿笔，水平拖拽 ===
                elif s_type == "NA":
                    if t < 0.15: # 蚕头 (轻灵)
                        z = np.interp(t, [0, 0.15], [self.cfg.BASE_Z+self.cfg.LIFT_Z, self.cfg.BASE_Z+self.cfg.STRK_Z + 1.0])
                        v_ratio = 0.45
                    elif t > 0.75: # 燕尾
                        if t <= 0.9: # 铺毫
                            target_deep = self.cfg.BASE_Z + self.cfg.PRESS_Z + 1.5 # 稍微克制，不按死
                            z = np.interp(t, [0.75, 0.9], [self.cfg.BASE_Z+self.cfg.STRK_Z + 0.8, target_deep])
                            v_ratio = 0.2 
                            # 增加大幅度抖动，模拟笔毫散开
                            curr_p[0] += np.random.uniform(-0.3, 0.3) / self.cfg.FONT_SCALE 
                            if t >= 0.88: pause = 0.2 
                        else: # 拖拽
                            target_deep = self.cfg.BASE_Z + self.cfg.PRESS_Z + 1.5
                            z = np.interp(t, [0.9, 1.0], [target_deep, self.cfg.BASE_Z+self.cfg.LIFT_Z])
                            # 虚位拖拽：水平延伸 4.0mm
                            virtual_drag = 4.0 * (t - 0.9) / 0.1 / self.cfg.FONT_SCALE
                            curr_p += unit_vec * virtual_drag
                            v_ratio = 0.35 
                    else: # 波折
                        z = self.cfg.BASE_Z + self.cfg.STRK_Z + 0.8 + 0.5 * np.sin(t * np.pi) + noise_z
                        v_ratio = 0.6 + 0.3 * np.sin(t * np.pi)

                # === [竖 SHU]：中锋铺毫 ===
                elif s_type == "SHU":
                    if t > 0.5:
                        shu_extend = 100.0 * (t - 0.5) / 0.5 / self.cfg.FONT_SCALE
                        curr_p += unit_vec * shu_extend
                    if t < 0.15:
                         z = np.interp(t, [0, 0.15], [self.cfg.BASE_Z + self.cfg.LIFT_Z, self.cfg.BASE_Z + self.cfg.PRESS_Z])
                         v_ratio = 0.4
                    elif t > 0.99:
                        z = np.interp(t, [0.99, 1.0], [self.cfg.BASE_Z + self.cfg.PRESS_Z + 1.0, self.cfg.BASE_Z + self.cfg.LIFT_Z])
                        v_ratio = 0.3
                    else:
                        z = self.cfg.BASE_Z + self.cfg.STRK_Z - 1.0 # 竖画略粗
                        v_ratio = 0.65

                # === [折 ZHE]：顿挫转折 ===
                elif s_type == "ZHE":
                    if t_pivot - 0.08 <= t < t_pivot:
                        z = self.cfg.BASE_Z + self.cfg.STRK_Z +  0.5 * np.sin(t * np.pi) + noise_z
                        v_ratio = 0.6 
                    elif abs(t - t_pivot) <= 0.02:
                        z = self.cfg.BASE_Z + self.cfg.PRESS_Z - 1.0 + noise_z
                        pause = 0.15 
                        v_ratio = 0.2 
                    elif t_pivot < t <= t_pivot + 0.1:
                        z = self.cfg.BASE_Z + self.cfg.STRK_Z
                        v_ratio = 0.5 
                    else:
                        z = self.cfg.BASE_Z + self.cfg.STRK_Z - 0.5 * np.sin(t * np.pi) + noise_z
                        v_ratio = 0.65
                
                # === [钩 GOU]：蓄势弹起 (修改版：停顿+左上飞出) ===
                elif s_type == "GOU":
                    # 1. 转折点：深按并停顿
                    if abs(t - t_pivot) <= 0.05:
                        z = self.cfg.BASE_Z + self.cfg.PRESS_Z - 0.5 
                        pause, v_ratio = 0.15, 0.1 # 增加停顿时间，降低速度
                    
                    # 2. 出钩阶段
                    elif t > t_pivot:
                        rescale_t = (t - t_pivot) / (1.0 - t_pivot)
                        
                        # Phase A: 蓄势 (响应"最后部分先停顿一下")
                        # 在出钩的前 40% 行程，保持深按，速度极慢
                        if rescale_t < 0.4:
                            z = self.cfg.BASE_Z + self.cfg.PRESS_Z
                            v_ratio = 0.15 
                        
                        # Phase B: 弹射 (响应"向左上方迅速抬笔")
                        else:
                            flick_t = (rescale_t - 0.4) / 0.6 # 归一化弹射进度
                            
                            # Z轴：线性抬起
                            z = (self.cfg.BASE_Z + self.cfg.PRESS_Z) + (self.cfg.LIFT_Z - self.cfg.PRESS_Z) * flick_t
                            
                            # 速度：爆发式加速
                            v_ratio = 2.0 + flick_t * 4.0 
                            
                            # 坐标修正：强制向左上 (Image: x减小, y减小)
                            # 模拟手腕翻转动作，偏移量随抬笔逐渐增大
                            offset_mag = 5.0 * flick_t / self.cfg.FONT_SCALE # 偏移 5mm
                            # 向量 [-0.8, -1.0] 代表左上方 (偏上)
                            curr_p += np.array([-0.8, -1.0]) * offset_mag
                            
                # === [默认 NORMAL/DIAN] ===
                else: 
                    z = np.interp(t, [0, 0.1, 0.9, 1.0], [self.cfg.BASE_Z+self.cfg.LIFT_Z, self.cfg.BASE_Z+self.cfg.STRK_Z, self.cfg.BASE_Z+self.cfg.STRK_Z, self.cfg.BASE_Z+self.cfg.LIFT_Z])
                    v_ratio = 1.0

                # 5. 最终坐标变换与抖动叠加
                rx = offset_x + (1024 - curr_p[1]) * self.cfg.FONT_SCALE
                ry = offset_y + curr_p[0] * self.cfg.FONT_SCALE

                rx += np.random.uniform(-jitter_xy, jitter_xy)
                ry += np.random.uniform(-jitter_xy, jitter_xy)

                all_waypoints.append({
                    'x': rx, 'y': ry, 'z': z, 
                    'pause': pause,
                    'velocity': self.cfg.GLOBAL_SPEED * v_ratio
                })
            
            self.last_pos = all_waypoints[-1]
            
        return all_waypoints

# =================================================================================
# 4. 逆运动学（IK）算法 (全新升级：完美匹配官方底层逻辑)
# =================================================================================
class DobotInverseKinematics:
    """
    将 (X, Y, Z) 完美映射为关节角 (J1, J2, J3)
    内部逻辑已与 Dobot Studio 官方坐标系完全同步
    """
    def __init__(self):
        # 官方写死的物理几何参数 (mm)
        self.L2 = 135.0  # 大臂长度
        self.L3 = 147.0  # 小臂长度
        self.TOOL_X = 61.0  # 官方默认的末端工具水平延伸量
        # 注意：在官方算法中，Z轴不需要加上底座高度 L1

    def cartesian_to_joint(self, x, y, z):
        """
        输入: 笛卡尔坐标 (x, y, z)
        输出: 包含 [j1, j2, j3, r] 的列表 (单位: 度)
        """
        # 1. 计算 J1 (底座旋转角度)
        j1 = math.degrees(math.atan2(y, x))
        
        # 2. 计算目标点在 R-Z 平面的相对位置 (减去工具的水平延伸量)
        r_target = math.sqrt(x**2 + y**2) - self.TOOL_X
        
        # 3. 求解三角形 (两边为 L2, L3，对角线为 dist)
        dist_sq = r_target**2 + z**2
        dist = math.sqrt(dist_sq)
        
        if dist > (self.L2 + self.L3) or dist < abs(self.L2 - self.L3):
            raise ValueError("目标坐标超出机械臂物理触达范围")

        # alpha 是目标连线与大臂的夹角
        cos_alpha = (self.L2**2 + dist_sq - self.L3**2) / (2 * self.L2 * dist)
        alpha = math.degrees(math.acos(max(-1, min(1, cos_alpha))))
        
        # gamma 是肘部内部夹角
        cos_gamma = (self.L2**2 + self.L3**2 - dist_sq) / (2 * self.L2 * self.L3)
        gamma = math.degrees(math.acos(max(-1, min(1, cos_gamma))))
        
        # 4. 完美映射回官方 UI 坐标系
        # phi_from_z 是目标连线相对于垂直 Z 轴的角度
        phi_from_z = math.degrees(math.atan2(r_target, z))
        
        # J2：大臂相对于垂直轴的角度
        j2 = phi_from_z - alpha
        
        # J3：小臂角度
        j3 = 90.0 + j2 - gamma
        
        # ⬇️！！！关键限位保护！！！⬇️
        # 防止算法输出极端角度导致机械臂硬件损坏
        if not (-90 <= j1 <= 90):
            raise ValueError(f"J1(底座) 角度 {j1:.1f} 超出限位")
        if not (0 <= j2 <= 85):
            raise ValueError(f"J2(大臂) 角度 {j2:.1f} 超出限位")
        if not (-10 <= j3 <= 95):
            raise ValueError(f"J3(小臂) 角度 {j3:.1f} 超出限位")
        
        # 最终返回 (四位列表，第4位是R轴设为0.0)
        return [round(j1, 4), round(j2, 4), round(j3, 4), 0.0]

# =================================================================================
# 5. 硬件驱动层 (Hardware Driver)
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
            # 设置 PTP 运动参数 (跳跃模式参数等)
            dType.SetPTPCommonParams(self.api, Config.GLOBAL_SPEED, Config.GLOBAL_SPEED, isQueued=1)
            self.connected = True
            return True
        print(f"!!! 连接失败: 无法连接到 {self.port}")
        return False

    def execute_trajectory(self, waypoints):
        """
        分段下发逻辑：监控机械臂缓存，防止指令溢出
        """
        if not self.connected: return 0

        ik = DobotInverseKinematics()
        last_id = 0
        QUEUE_THRESHOLD = 200 # 当积压指令超过200条时，暂停下发

        for i, pt in enumerate(waypoints):
            # --- 分段监控核心 ---
            if i % 10 == 0: # 每10条指令查一次岗
                while True:
                    curr_id = dType.GetQueuedCmdCurrentIndex(self.api)[0]
                    # 计算队列中还没跑完的指令数
                    if (last_id - curr_id) < QUEUE_THRESHOLD:
                        break
                    time.sleep(0.05) # 队列太满，歇 50 毫秒

            # --- 下发指令 ---
            if Config.USE_JOINT_MODE:
                try:
                    # 使用完美匹配官方逻辑的 IK 进行解算
                    joints = ik.cartesian_to_joint(pt['x'], pt['y'], pt['z'])
                    dType.SetPTPCommonParams(self.api, pt['velocity'], pt['velocity'], isQueued=1)
                    # Mode 5: 关节空间直线插补
                    last_id = dType.SetPTPCmd(self.api, 5, joints[0], joints[1], joints[2], joints[3], isQueued=1)[0]
                except ValueError as e:
                    print(f"坐标跳过 ({pt['x']:.1f}, {pt['y']:.1f}, {pt['z']:.1f}): {e}")
                    continue
            else:
                dType.SetPTPCommonParams(self.api, pt['velocity'], pt['velocity'], isQueued=1)
                last_id = dType.SetPTPCmd(self.api, 2, pt['x'], pt['y'], pt['z'], 0, isQueued=1)[0]

            if pt.get('pause', 0) > 0:
                last_id = dType.SetWAITCmd(self.api, int(pt['pause'] * 1000), isQueued=1)[0]
            
        return last_id

    def wait_finish(self, last_id):
        """阻塞等待直到指令队列执行完毕"""
        try:
            while True:
                curr_id = dType.GetQueuedCmdCurrentIndex(self.api)[0]
                if curr_id >= last_id: break
        except KeyboardInterrupt:
            dType.SetQueuedCmdStopExec(self.api)
            print("\n!!! 紧急停止 !!!")

    def close(self):
        dType.DisconnectDobot(self.api)
        print(">>> 机械臂连接已断开")

# =================================================================================
# 6. 主程序入口 (Main)
# =================================================================================
def main():
    # 1. 初始化驱动
    driver = DobotDriver(Config.COM_PORT)
    if not driver.connect(): return
    
    # 2. 初始化引擎
    engine = BrushEngine(Config)
    
    # 3. 获取输入并生成轨迹
    text = input("请输入汉字: ")
    full_trajectory = []
    
    for index, char in enumerate(text):
        path = os.path.join(Config.DATA_DIR, f"{char}.json")
        if os.path.exists(path):
            print(f"正在处理字符: {char} ...")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    medians = json.load(f).get("medians", [])
                off_y = Config.START_Y + (index * Config.SPACING)
                full_trajectory.extend(engine.generate_waypoints(medians, Config.START_X, off_y))
            except Exception as e:
                print(f"数据解析失败: {e}")
        else:
            print(f"未找到笔画数据: {path}")

    # 4. 指令有序下发
    if full_trajectory:
        print(f">>> 开始执行书法任务，总点数: {len(full_trajectory)}")
        
        # 1. 预位移动 (不进入队列)
        f_pt = full_trajectory[0]
        dType.SetPTPCmd(driver.api, 1, f_pt['x'], f_pt['y'], Config.BASE_Z + Config.SAFE_Z, 0, isQueued=0)
        
        # 2. 启动队列执行（必须在发送前启动，否则分段下发的 GetQueuedCmdCurrentIndex 不会更新）
        dType.SetQueuedCmdStartExec(driver.api) 
        
        # 3. 开始下发（内部包含分段等待逻辑）
        last_id = driver.execute_trajectory(full_trajectory)
        
        # 4. 结束收笔动作
        last_id = dType.SetPTPCmd(driver.api, 1, f_pt['x'], f_pt['y'], Config.BASE_Z + Config.SAFE_Z, 0, isQueued=1)[0]
        
        print(">>> 指令已全部传输至机械臂，等待书写完成...")
        driver.wait_finish(last_id)
        
        # 5. 停止并归位
        dType.SetQueuedCmdStopExec(driver.api)
    
    driver.close()
    print(">>> 任务圆满完成")

if __name__ == "__main__":
    main()