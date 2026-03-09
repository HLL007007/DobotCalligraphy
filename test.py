import math

class DobotValidator:
    def __init__(self):
        # 严格按照 Dobot Magician 物理参数 (mm)
        self.L1 = 138.0  # 基座高度
        self.L2 = 135.0  # 大臂长度
        self.L3 = 147.0  # 小臂长度

    def forward_kinematics(self, j1, j2, j3):
        """
        正运动学：输入角度(deg)，输出末端坐标(x, y, z)
        注意：Dobot 的 J3 是小臂相对于水平面的角度（受平行四边形机构约束）
        """
        # 转为弧度
        r_j1, r_j2, r_j3 = map(math.radians, [j1, j2, j3])
        
        # 计算在 R-Z 平面的投影长度
        r = self.L2 * math.cos(r_j2) + self.L3 * math.cos(r_j3)
        
        # 映射回 3D 空间
        x = r * math.cos(r_j1)
        y = r * math.sin(r_j1)
        z = self.L1 + self.L2 * math.sin(r_j2) + self.L3 * math.sin(r_j3)
        
        return round(x, 4), round(y, 4), round(z, 4)

    def inverse_kinematics(self, x, y, z):
        """
        逆运动学：输入末端坐标(x, y, z)，输出角度(j1, j2, j3)
        """
        # 1. 计算 J1
        j1 = math.degrees(math.atan2(y, x))
        
        # 2. 几何平面降维
        r = math.sqrt(x**2 + y**2)
        z_rel = z - self.L1  # 相对于第一个关节的高度
        
        # 3. 余弦定理求解
        dist_sq = r**2 + z_rel**2
        dist = math.sqrt(dist_sq)
        
        if dist > (self.L2 + self.L3) or dist < abs(self.L2 - self.L3):
            raise ValueError("Target out of reach")

        cos_alpha = (self.L2**2 + dist_sq - self.L3**2) / (2 * self.L2 * dist)
        alpha = math.acos(max(-1, min(1, cos_alpha)))
        
        cos_gamma = (self.L2**2 + self.L3**2 - dist_sq) / (2 * self.L2 * self.L3)
        gamma = math.acos(max(-1, min(1, cos_gamma)))
        
        beta = math.atan2(z_rel, r)
        
        # 4. 得到结果
        j2 = math.degrees(beta + alpha)
        j3 = math.degrees(beta + alpha - (math.pi - gamma))
        
        return round(j1, 4), round(j2, 4), round(j3, 4)

# --- 验证环节 ---
validator = DobotValidator()

# 选取几个典型的书法坐标点 (x, y, z)
test_points = [
    (200.0, 0.0, 50.0),    # 正前方
    (150.0, 150.0, 20.0),  # 侧方低位
    (180.0, -50.0, 80.0),  # 侧方高位
]

print(f"{'原始坐标 (X,Y,Z)':<25} | {'逆解角度 (J1,J2,J3)':<25} | {'正解还原 (X,Y,Z)':<25} | {'误差(mm)'}")
print("-" * 100)

for p in test_points:
    try:
        # 1. 逆解算
        j1, j2, j3 = validator.inverse_kinematics(*p)
        # 2. 正解验证
        p_new = validator.forward_kinematics(j1, j2, j3)
        # 3. 计算欧氏距离误差
        error = math.sqrt(sum([(a - b)**2 for a, b in zip(p, p_new)]))
        
        print(f"{str(p):<25} | {str((j1,j2,j3)):<25} | {str(p_new):<25} | {error:.6f}")
    except Exception as e:
        print(f"坐标 {p} 验证失败: {e}")