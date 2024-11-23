import tkinter as tk
import time

import airbot

# 初始化机械臂对象
arm = airbot.create_agent(end_mode="gripper")
pos, quat = arm.get_current_pose()  # [t1, t2, t3, x, y, z, w]

def move_forward(event=None):
    """控制机械臂前进"""
    print("Moving forward...")
    pos[0] += 0.01
    arm.set_target_pose(pos, quat, vel=0.05)
    update_position()

def move_backward(event=None):
    """控制机械臂后退"""
    print("Moving backward...")
    pos[0] -= 0.01
    arm.set_target_pose(pos, quat, vel=0.05)
    update_position()

def move_left(event=None):
    """控制机械臂向左"""
    print("Moving left...")
    pos[1] += 0.01
    arm.set_target_pose(pos, quat, vel=0.05)
    update_position()

def move_right(event=None):
    """控制机械臂向右"""
    print("Moving right...")
    pos[1] -= 0.01
    arm.set_target_pose(pos, quat, vel=0.05)
    update_position()

def move_up(event=None):
    """控制机械臂上升"""
    print("Moving up...")
    pos[2] += 0.01
    arm.set_target_pose(pos, quat, vel=0.05)
    update_position()

def move_down(event=None):
    """控制机械臂下降"""
    print("Moving down...")
    pos[2] -= 0.01
    arm.set_target_pose(pos, quat, vel=0.05)
    update_position()

def close_gripper(event=None):
    """控制机械臂闭合抓手"""
    print("Closing gripper...")
    arm.set_target_end(0)
    time.sleep(0.5)
    update_position()

def open_gripper(event=None):
    """控制机械臂打开抓手"""
    print("Opening gripper...")
    arm.set_target_end(1)
    time.sleep(0.5)
    update_position()

def update_position():
    """实时更新抓手的位置"""
    cur_end = arm.get_current_end()
    pos_now, _ = arm.get_current_pose()
    # 仅保留两位小数
    pos_now = [round(p, 2) for p in pos_now]
    position = f"{cur_end}" + "\n" + f"{pos_now}"
    position_label.config(text=f"Gripper Position: {position}")

def main():
    # 创建主窗口
    root = tk.Tk()
    root.title("Robot Arm Controller")

    # 绑定键盘事件
    root.bind("<w>", move_forward)
    root.bind("<s>", move_backward)
    root.bind("<q>", move_up)
    root.bind("<e>", move_down)
    root.bind("<a>", move_left)
    root.bind("<d>", move_right)
    root.bind("<j>", close_gripper)
    root.bind("<k>", open_gripper)

    # 显示抓手的位置
    global position_label
    position_label = tk.Label(root, text="Gripper Position: Unknown", font=("Arial", 16), fg="blue")
    position_label.pack(pady=25)

    # 说明标签
    instruction_label = tk.Label(root, text="Use W/A/S/D/Q/E to move, J to close gripper, K to open gripper.", font=("Arial", 14))
    instruction_label.pack(pady=15)

    # 开始主循环
    root.mainloop()

if __name__ == "__main__":
    main()
