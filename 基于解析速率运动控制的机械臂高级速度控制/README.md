# 基于解析速率运动控制的机械臂高级速度控制

**学校：** 帝国理工学院（Imperial College London）  
**合作单位：** MathWorks 高级研究与技术办公室  
**导师：** Dr. Giordano Scarciotti  
**MathWorks 指导：** Roberto G. Valenti, George Amarantidis, Rory Adams  

---

## 项目概述
本项目针对 UR5 协作机械臂，基于 **解析速率运动控制（Resolved-Rate Motion Control, RRMC）** 在 MATLAB Simulink 中开发了一个 **高级速度控制框架**。  
研究重点在于解决 **运动学奇异点附近的不稳定问题**，并比较了 **Moore–Penrose 伪逆 (MP)** 与 **阻尼最小二乘法 (DLS)** 两种逆运动学方法的鲁棒性。

主要目标：  
- 实现 **平滑且高精度的轨迹跟踪**。  
- 搭建 **模块化、可复用的 Simulink 控制框架**，便于科研和工业应用。  
- 对比 **MP 与 DLS** 的性能，验证在奇异点附近的稳定性。  

---

## 仓库结构
基于解析速率运动控制的机械臂高级速度控制/
│
├─ 模型/ # 主要 Simulink 实现
│ └─ Integration.slx
│
├─ 报告/ # 项目文档
│ ├─ Chen_Guan-Sheng_[EXTERNAL]_MathWorks_Project.pdf
│ └─ Chen_Guan-Sheng_Poster.pdf
|
└─ README_CN.md # 中文版说明文件


---

## 功能特点
- **轨迹生成**：基于梯形速度曲线。  
- **正/逆微分运动学**：实现任务空间与关节空间的映射。  
- **逆运动学方法：**  
  - Moore–Penrose 伪逆 (MP)  
  - 阻尼最小二乘法 (DLS)，支持阻尼系数 λ 调整  
- **关节空间 PID 控制**：将期望速度转化为力矩命令。  
- **安全机制**：包括速率限制、速度与力矩饱和，便于向真实硬件移植。  

---

## 关键成果
- **DLS 在奇异点附近表现优于 MP**，轨迹执行更稳定。  
- 在仿真中实现 **亚厘米级末端定位精度**。  
- 找到最佳阻尼系数 **λ ≈ 0.03**，在鲁棒性与精度间达到平衡。  
- 提出 **分组 Kp 分配策略**，提升多关节复杂运动的平滑性。  

---

## 技术栈
- **工具：** MATLAB Simulink, Robotics System Toolbox, Simulation 3D  
- **方法：** RRMC, DLS 逆运动学, PID 控制  
- **机器人：** UR5 协作机械臂（仿真）  

---

## 文件说明
- **Integration.slx** – 完整的 RRMC 框架 Simulink 模型  
- **Chen_Guan-Sheng_[EXTERNAL]_MathWorks_Project.pdf** – 技术报告  
- **Chen_Guan-Sheng_Poster.pdf** – 项目海报  

---

## 应用方向
- **工业机器人与智能制造**  
- **康复与辅助机器人**  
- **控制系统科研与教学**  

---
