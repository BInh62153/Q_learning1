#/env/driving_env.py
import numpy as np

class DrivingEnvironment:
    def __init__(self, difficulty=1):
        self.difficulty = difficulty
        self.state = None


#========================================================
#Phương thức này được gọi khi bắt đầu một episode mới.
#Nó khởi tạo state là mảng 3 phần tử, tất cả bằng 0: [0,0,0].
#Trả về trạng thái ban đầu để agent có thể bắt đầu học.
#=============================================================
    def reset(self):
        self.state = np.zeros(3)
        return self.state


#==========================================================================================
#Đây là bước chính của môi trường: agent thực hiện một action và môi trường trả về kết quả.
#reward: phần thưởng ngẫu nhiên, có giảm đi một chút theo độ khó (-0.1 * difficulty).
#done: xác suất kết thúc episode, càng khó thì càng dễ kết thúc (0.1 * difficulty).
#state: trạng thái mới, được tạo ngẫu nhiên mỗi lần step (np.random.randn(3)).
#{}: thông tin thêm (ở đây không dùng)
#=================================================================================
    def step(self, action):
        reward = np.random.randn() - 0.1 * self.difficulty
        done = np.random.rand() < 0.1 * self.difficulty
        self.state = np.random.randn(3)
        return self.state, reward, done, {}

    def render(self):
        pass

