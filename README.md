# Dobot Magician 机械臂书法系统
### 🌟 总体导览：

要让机械臂写书法，一共需要分五步，对应代码里的五大模块：
1. **工具箱 (`Imports`)**：准备好需要的零件和翻译官。
2. **大脑参数设定 (`Config`)**：规定好纸在哪、笔要压多深、字写多大。
3. **火眼金睛 (`StrokeAnalyzer`)**：看着字库里的点，认出这是“横”还是“撇”。
4. **肌肉记忆 (`BrushEngine`)**：**最核心的部分！** 教机器人起笔怎么切锋，收笔怎么回锋。
5. **底层神经 (`DobotDriver` & `main`)**：把想好的动作，变成电流信号发给真实的机械臂。


### 第一部分：准备工具箱 (Imports)

```python
import json             # 工具1：用来读取字库文件（因为字库是.json格式的）
import os               # 工具2：用来在电脑硬盘里找文件、拼凑文件路径
import sys              # 工具3：用来和电脑操作系统打交道（比如遇到严重错误时强制退出）
import numpy as np      # 工具4：超级数学计算器！算角度、算距离、算曲线全靠它（简称np）
import DobotDllType as dType  # 工具5：越疆机械臂的官方“翻译字典”，能把Python翻译成机器懂的指令
```

---

### 第二部分：大脑参数设定 (Config)

```python
class Config:
    # --- 基础设置 ---
    DATA_DIR = "./data"       # 告诉程序：笔画的数据文件都在旁边叫 "data" 的文件夹里
    COM_PORT = "COM3"         # 告诉程序：机械臂的USB线插在电脑的 "COM3" 接口上
    
    # --- 空间坐标参数 (单位: mm) ---
    BASE_Z   = 16.0           # 基准线：笔尖刚好轻轻碰到纸面的高度（16毫米）
    LIFT_Z   = 5.0            # 抬笔：写完一笔后，笔尖抬高 5 毫米，在空中移动，防止蹭脏纸
    SAFE_Z   = 25.0           # 绝对安全：全部写完后，笔尖高高抬起 25 毫米
    
    # --- 笔触深度参数 (控制毛笔的粗细) ---
    # 注意：Z往下走是负数，负得越多，笔毛压得越扁，字越粗
    PRESS_Z  = -5.0           # 猛压：往下压 5 毫米。用于顿笔、捺脚、拐折处（最粗）
    STRK_Z   = -2.0           # 正常走笔：稍微压下去 2 毫米。用于横、竖的中间部分（正常粗细）
    
    # --- 运动参数 ---
    GLOBAL_SPEED = 40         # 机器人移动的全局基础速度（每秒 40 毫米）
    FONT_SCALE   = 0.045      # 缩放魔法：电脑里的字库超级大（1024x1024），乘以 0.045 把它缩成能在纸上写的小字
    
    # --- 排版参数 ---
    START_X      = 220.0      # 第一个字在纸上的前后位置
    START_Y      = 0.0        # 第一个字在纸上的左右位置
    SPACING      = 50.0       # 字间距：写完一个字，下一个字往旁边挪 50 毫米
```

---

### 第三部分：火眼金睛 (StrokeAnalyzer)

**【总述】** 机器人拿到的一堆坐标点，它本身不知道这是啥。这个模块负责通过计算**弯曲程度**和**走向**，识别出这到底是个“横”、“撇”还是“钩”。

```python
class StrokeAnalyzer:
    
    @staticmethod
    def _resample_stroke(stroke, num_samples=50):
        # 【理顺线条】原始数据有的地方点很密，有的很稀。
        # 这个函数用数学方法，把一条线重新切分成 50 个距离相等的点，方便后面计算。
        # (里面用到了求两点距离、累加距离、插值等数学魔法)
        ...

    @staticmethod
    def get_curvature_profile(stroke):
        # 【找转折点】计算这条线段哪里最弯（曲率谱）。
        # 它通过算相邻两个点的角度差，差值越大，说明拐弯越急。
        ...

    @staticmethod
    def analyze(stroke):
        # 【核心认字逻辑】
        num_pts = len(stroke)
        if num_pts < 5: return "NORMAL" # 如果点太少，就当普通短点处理

        # 1. 先看它弯不弯
        curvatures = StrokeAnalyzer.get_curvature_profile(stroke)
        max_curv_idx = np.argmax(curvatures) # 找到最弯的那个点
        max_curv_val = curvatures[max_curv_idx] # 看看它到底有多弯
        t_pivot = max_curv_idx / len(curvatures) # 看看这个弯出现在整条线的哪个阶段（百分比）
        
        # 如果弯曲度大于0.7（说明是个急转弯！）
        if max_curv_val > 0.7:
            if 0.10 < t_pivot < 0.75: return "ZHE" # 弯在中间，说明是个“折” (比如口字的右上角)
            elif t_pivot >= 0.75:     return "GOU" # 弯在尾巴，说明是个“钩” (比如竖钩的底下)

        # 2. 如果不怎么弯，就是直的，那就看方向角度
        p_start, p_end = np.array(stroke[0]), np.array(stroke[-1]) # 拿出起点和终点
        vec = p_end - p_start # 连成一条线
        angle = np.degrees(np.arctan2(vec[1], vec[0])) # 算出这条线的角度
        
        if -20 < angle < 20:   return "HENG" # 差不多平着走 -> 横
        if 20 <= angle < 65:   return "NA"   # 往右下角走 -> 捺
        if 65 <= angle < 125:  return "SHU"  # 直直往下走 -> 竖
        if 125 <= angle < 175: return "PIE"  # 往左下角走 -> 撇
        
        return "NORMAL" # 认不出来就当普通笔画写
```

---

### 第四部分：肌肉记忆引擎 (BrushEngine) - 🌟全篇最牛的地方

**【总述】** 这里是教机器人具体的书法招式。不同笔画，力度怎么变？速度怎么变？

#### 4.1 空中动作与基础准备
```python
class BrushEngine:
    def __init__(self, config):
        self.cfg = config # 把第二部分的设置记住
        self.last_pos = None # 记录上一笔停在哪了

    def _generate_bezier_curve(self, start_pt, end_pt, num_points=12):
        # 【空中取势】上一笔写完，到下一笔开始，不能傻乎乎走直角。
        # 这里用“三阶贝塞尔曲线”公式，让机械臂在空中划出一道优美的抛物线弧度。
        ...

    def generate_waypoints(self, medians, offset_x, offset_y):
        all_waypoints = [] # 记录接下来每一步的坐标
        
        for stroke in medians: # 把字拆开，一笔一笔地处理
            # 1. 认出这是什么笔画？
            s_type = StrokeAnalyzer.analyze(stroke) 
            
            # 2. 算出这笔往哪边指 (unit_vec，方向向量，后面回锋要用到)
            ... 
            
            # 3. 笔尖飞到这笔的起点正上方 (加上之前算好的字间距 offset_x, offset_y)
            # 如果有上一笔，就在空中画条弧线飞过去
            ...
```

#### 4.2 书法魔法：物理模拟层（开始写纸上的部分）
**【总述】** 这里把一笔切成了 60 个小段（`sample_count = 60`），用 `t` 表示进度（0 代表刚下笔，1 代表写完了）。

```python
            for i in range(sample_count):
                t = i / (sample_count - 1) # t 就是进度条 (0.0 到 1.0)
                curr_p = ... # 算出当前走到哪个坐标点了

                # 【补丁魔法】如果是个“折”，到了80%以后，强制笔画往下多走一点（延长50mm）。
                # 为什么？因为很多字帖的“口”字，右下角是不封口的。这样一拉长，字就封口了！
                if s_type == "ZHE" and t > 0.8: ...

                # --- 准备魔法变量 ---
                z = self.cfg.BASE_Z + self.cfg.STRK_Z # 默认高度是“正常走笔”
                pause = 0 # 默认不停顿
                v_ratio = 1.0 # 默认速度是 1 倍速
                
                # 给机器人的手加点“帕金森”：
                noise_z = np.random.uniform(-0.15, 0.15) # 高度随机微微抖动
                # 这样做是为了模拟宣纸粗糙的质感，让墨迹边缘有自然的毛糙感，不像打印机印出来的！
```

#### 4.3 各类笔画绝招！（代码拆解）
```python
                # 💥 绝招一：【横】
                if s_type == "HENG":
                    if t < 0.15: # 刚下笔
                        # 笔尖从空中斜切下去，力度变重，速度变慢
                        z = 越来越深; v_ratio = 越来越慢
                    elif t > 0.85: # 快写完了
                        if t <= 0.92: # 重重顿一下，发呆0.15秒
                            z = 猛压; pause = 0.15
                        else: # 【回锋魔法】
                            z = 抬起
                            # 逆着来的方向，倒退 0.6 毫米！把笔尖藏起来！
                            curr_p -= unit_vec * 倒退距离 
                            
                # 💥 绝招二：【撇】
                elif s_type == "PIE":
                    # 前面正常写...
                    if t > 0.7: # 写到尾巴了
                        # 【虚位弹射魔法】
                        z = 快速抬起
                        v_ratio = 疯狂加速 (最高飙到2倍多速度)
                        # 故意把目标点设在离纸很远的半空中，欺骗机器人猛冲过去，
                        # 就能甩出刀刃一样锋利的撇尖！
                        curr_p += unit_vec * 延伸距离
                        
                # 💥 绝招三：【捺】
                elif s_type == "NA":
                    if t > 0.75: # 到了捺脚（最粗的地方）
                        if t <= 0.9: # 狠狠压下去，并随机左右抖动（铺毫：把毛笔搓开）
                            z = 猛压; pause = 0.2
                            curr_p[0] += 左右抖动
                        else: # 【水平拖拽魔法】笔尖不抬起来，平着往外拖，拉出燕尾！
                            ...
                            
                # 💥 绝招四：【钩】
                elif s_type == "GOU":
                    if abs(t - t_pivot) <= 0.05: # 到了拐角处
                        z = 猛压; pause = 0.15 # 停顿蓄势
                    elif t > t_pivot: # 出钩啦！
                        if rescale_t < 0.4: # 先闷着头压一会
                            z = 猛压
                        else: # 突然向左上方爆发！
                            v_ratio = 极速飙升
                            # 强制修改坐标，往左上方(-0.8, -1.0)猛提！
                            curr_p += np.array([-0.8, -1.0]) * 偏移量
```

```python
                # 把上面加了魔法的坐标，换算成真实的机械臂坐标
                rx = ...; ry = ...
                # 存进路线本子里
                all_waypoints.append({'x': rx, 'y': ry, 'z': z, 'pause': pause, 'velocity': 速度})
```

---

### 第五部分：底层神经系统 (DobotDriver)

**【总述】** 算出了成千上万个坐标点，现在要把它们变成命令发送给机器人。

```python
class DobotDriver:
    def __init__(self, port):
        self.api = dType.load() # 拿出说明书
        self.port = port # 记住插在哪个接口

    def connect(self):
        # 尝试连接机器人，成功就说“连接成功”，失败就报错
        ...

    def execute(self, waypoints):
        # 【派发任务】
        cmd_id = 0
        for wp in waypoints: # 遍历本子里的每一个点
            # 1. 告诉机器人去这个点的速度
            dType.SetPTPCommonParams(...) 
            # 2. 告诉机器人去这个 x,y,z 坐标
            dType.SetPTPCmd(self.api, 模式, wp['x'], wp['y'], wp['z'], ...)
            # 3. 如果这个点需要停顿(顿笔)，就发一条等待指令
            if wp.get('pause', 0) > 0:
                dType.SetWAITCmd(self.api, 等待多少毫秒)
        return cmd_id # 返回最后一个任务的编号

    def wait_finish(self, last_id):
        # 【死等】因为指令发得快，机器人动得慢。
        # 这个函数就是死死盯着机器人，直到它做到最后一个任务编号为止。
        ...
```

---

### 第六部分：总司令部 (main)

**【总述】** 程序的入口。你一点运行，电脑就是从这里开始执行的。串联了上面所有的功能。

```python
def main():
    # 1. 唤醒并连接机器人
    driver = DobotDriver(Config.COM_PORT)
    if not driver.connect(): return # 连不上就拉倒
    
    # 2. 唤醒书法大脑
    engine = BrushEngine(Config)
    
    # 3. 问人类你想写什么字？
    text = input("请输入汉字 (确保data目录下有对应的json文件): ")
    
    full_trajectory = [] # 拿出一个全新的空白本子
    
    # 4. 把字排好队，一个个算路线
    for index, char in enumerate(text):
        # 去 data 文件夹里找对应汉字的 json 图纸
        path = os.path.join(Config.DATA_DIR, f"{char}.json")
        ...
        # 算出要挪动多少距离（字间距）
        off_y = Config.START_Y + (index * Config.SPACING)
        # 把图纸扔给大脑，把算好的绝招坐标，全抄到大本子上
        full_trajectory.extend(engine.generate_waypoints(medians, ...))

    # 5. 开始干活！
    if full_trajectory:
        driver.execute(full_trajectory) # 把一厚本坐标甩给机器人
        
        # 写完后，加一条命令：笔尖抬离纸面回到安全高度，免得弄脏
        f_pt = full_trajectory[-1]
        dType.SetPTPCmd(..., Config.BASE_Z + Config.SAFE_Z, ...)
        
        print(">>> 正在书写，请勿触碰机械臂...")
        driver.wait_finish(last_id) # 挂机等待它写完
    
    # 6. 打完收工
    driver.close()
    print(">>> 任务圆满完成")

# 这是Python的潜规则：代表只要你双击这个文件，就启动 main() 
if __name__ == "__main__":
    main()
```
