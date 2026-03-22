# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class TTFExtractor:
    def __init__(self, font_path, image_size=512):
        self.font_path = font_path
        self.image_size = image_size

    def render_char(self, char):
        """将汉字渲染为黑底白字的图像"""
        # 留白边缘，防止字贴边
        padding = 40
        actual_size = self.image_size - padding * 2
        font = ImageFont.truetype(self.font_path, actual_size)
        
        img = Image.new('L', (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(img)
        
        # 居中绘制
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (self.image_size - text_w) / 2 - bbox[0]
        y = (self.image_size - text_h) / 2 - bbox[1]
        
        draw.text((x, y), char, font=font, fill=255)
        return np.array(img)

    def extract_skeleton_and_width(self, img_array):
        """核心视觉算法：提取骨架和宽度"""
        # 1. 距离变换：计算白色区域内每个点到黑色边缘的最短距离 (这就代表了笔画的粗细/宽度)
        dist_transform = cv2.distanceTransform(img_array, cv2.DIST_L2, 5)
        
        # 2. 骨架提取：使用 skimage 获取单像素宽度的中心线
        binary = img_array > 127
        skeleton = morphology.skeletonize(binary)
        
        return skeleton, dist_transform

    def route_strokes(self, skeleton, dist_transform, jump_threshold=15.0):
        """
        轨迹规划：将散乱的骨架像素点连成连续的笔画
        使用贪心最近邻算法 (Greedy Nearest Neighbor)
        """
        # 获取所有骨架点的坐标 (y, x)
        points_y, points_x = np.where(skeleton)
        if len(points_x) == 0: return []
        
        points = np.column_stack((points_x, points_y)) # Shape: (N, 2)
        unvisited = set(range(len(points)))
        strokes = []
        current_stroke = []
        
        # 启发式规则：汉字通常从左上角开始写
        # 找到 Y+X 最小的点作为第一笔的起点
        start_idx = np.argmin(points[:, 0] + points[:, 1])
        curr_idx = start_idx
        unvisited.remove(curr_idx)
        current_stroke.append(curr_idx)
        
        while unvisited:
            curr_pt = points[curr_idx]
            
            # 向量化寻找最近的未访问点 (加速计算)
            unvisited_list = list(unvisited)
            unvisited_pts = points[unvisited_list]
            dists = np.linalg.norm(unvisited_pts - curr_pt, axis=1)
            
            min_i = np.argmin(dists)
            best_idx = unvisited_list[min_i]
            min_dist = dists[min_i]
            
            # 如果最近的点距离大于阈值，说明需要“抬笔”写下一画了
            if min_dist > jump_threshold:
                strokes.append(current_stroke)
                current_stroke = [best_idx]
            else:
                current_stroke.append(best_idx)
                
            curr_idx = best_idx
            unvisited.remove(curr_idx)
            
        if current_stroke:
            strokes.append(current_stroke)
            
        # 按照笔画的起始点高度 (Y坐标) 对笔画进行粗略排序 (从上到下)
        strokes.sort(key=lambda s: points[s[0]][1])
            
        # 组合为含有坐标和宽度的字典列表
        final_strokes_data = []
        for stroke in strokes:
            stroke_data = []
            for idx in stroke:
                x, y = points[idx]
                w = dist_transform[y, x] # 获取该点的实际笔画宽度
                stroke_data.append({'x': x, 'y': y, 'w': w})
            final_strokes_data.append(stroke_data)
            
        return final_strokes_data

    def map_to_dobot_waypoints(self, strokes_data, config):
        """将图像坐标 (X, Y, W) 降维打击映射到机械臂真实物理空间 (X, Y, Z)"""
        waypoints = []
        
        # 寻找全局最大宽度，用于归一化 Z 轴深度
        max_w = max([pt['w'] for stroke in strokes_data for pt in stroke])
        if max_w == 0: max_w = 1.0
        
        for stroke in strokes_data:
            if len(stroke) < 3: continue # 过滤噪点
            
            # 1. 抬笔移动到起点上方 (加入虚位)
            start_pt = stroke[0]
            # 坐标系映射：图像左上角(0,0) -> 机械臂起始点
            # 根据你原代码的映射规则进行了适配
            rob_x = config.START_X + (self.image_size - start_pt['y']) * config.FONT_SCALE
            rob_y = config.START_Y + start_pt['x'] * config.FONT_SCALE
            
            waypoints.append({
                'x': rob_x, 'y': rob_y, 
                'z': config.BASE_Z + config.LIFT_Z, 
                'pause': 0, 'velocity': config.GLOBAL_SPEED
            })
            
            # 2. 连续落下书写 (精髓：Z轴深度与宽度完全成正比！)
            for pt in stroke:
                rob_x = config.START_X + (self.image_size - pt['y']) * config.FONT_SCALE
                rob_y = config.START_Y + pt['x'] * config.FONT_SCALE
                
                # *** 灵魂公式 ***
                # 宽度 W 越大，Z 轴越深（接近 PRESS_Z），线条越粗！
                # 宽度 W 越小，Z 轴越浅（接近 BASE_Z），线条越细！
                normalized_w = pt['w'] / max_w
                # 利用非线性映射（平方），让尖锋更锐利，顿笔更沉重
                z_depth = config.BASE_Z - abs(config.PRESS_Z) * (normalized_w ** 1.5)
                
                waypoints.append({
                    'x': rob_x, 'y': rob_y, 'z': z_depth, 
                    'pause': 0, 'velocity': config.GLOBAL_SPEED * 0.7 # 行笔稍慢
                })
                
            # 3. 笔画结束，抬笔
            last_pt = waypoints[-1]
            waypoints.append({
                'x': last_pt['x'], 'y': last_pt['y'], 
                'z': config.BASE_Z + config.LIFT_Z, 
                'pause': 0, 'velocity': config.GLOBAL_SPEED
            })
            
        return waypoints

# ================= 模拟你的 Config =================
class MockConfig:
    START_X = 200.0
    START_Y = 0.0
    FONT_SCALE = 0.15   # 映射比例 (根据 image_size=512 调整)
    BASE_Z = 0.0
    LIFT_Z = 5.0
    PRESS_Z = -5.0      # 最深下压5毫米
    GLOBAL_SPEED = 40

# ================= 测试运行 =================
if __name__ == "__main__":
    # 替换为你下载的毛笔字体路径
    FONT_PATH = "maobi.ttf" 
    CHAR = "永"  # 测试汉字
    
    extractor = TTFExtractor(FONT_PATH, image_size=512)
    
    # 1. 渲染图像
    print("1. 正在渲染图像...")
    img = extractor.render_char(CHAR)
    
    # 2. 提取骨架与粗细
    print("2. 正在提取距离场(宽度)与骨架...")
    skeleton, dist_transform = extractor.extract_skeleton_and_width(img)
    
    # 3. 规划连贯笔画
    print("3. 正在进行路径规划...")
    strokes_data = extractor.route_strokes(skeleton, dist_transform)
    
    # 4. 生成机械臂代码
    print("4. 正在生成带有 Z 轴呼吸的机械臂轨迹...")
    waypoints = extractor.map_to_dobot_waypoints(strokes_data, MockConfig())
    print(f"总计生成 {len(waypoints)} 个运动指令点。")
    
    # ================= 可视化结果 (强烈建议查看) =================
    plt.figure(figsize=(15, 5))
    
    # 子图1：原图渲染
    plt.subplot(131)
    plt.title("Rendered TrueType Font")
    plt.imshow(img, cmap='gray')
    
    # 子图2：宽度场 (越亮表示笔画越粗)
    plt.subplot(132)
    plt.title("Distance Transform (Stroke Width/Pressure)")
    plt.imshow(dist_transform, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 子图3：提取的笔顺轨迹和提按深度
    plt.subplot(133)
    plt.title("Extracted Trajectory (Color = Z Depth)")
    # 绘制背景
    plt.imshow(img, cmap='gray', alpha=0.3)
    
    max_w = max([pt['w'] for stroke in strokes_data for pt in stroke])
    colors = plt.cm.jet(np.linspace(0, 1, len(strokes_data)))
    
    for i, stroke in enumerate(strokes_data):
        xs = [pt['x'] for pt in stroke]
        ys = [pt['y'] for pt in stroke]
        ws = [pt['w'] * 3 for pt in stroke] # 放大宽度用于显示点的大小
        
        # 用点的大小和颜色深浅来表示机械臂下压的力度
        plt.scatter(xs, ys, s=ws, c=[colors[i]], label=f'Stroke {i+1}')
        # 连线
        plt.plot(xs, ys, c=colors[i], linewidth=1)
        
    plt.gca().invert_yaxis()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.show()