import numpy as np
import matplotlib.pyplot as plt
from utils.logger import TrainingLogger


class Evaluator:
    def __init__(self, logger: TrainingLogger):
        self.logger = logger

    # ------------------------------------------------------------------
    def plot_learning_curve(self):
        """Vẽ biểu đồ reward trung bình theo episode"""
        rewards = self.logger.episode_rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Reward mỗi episode', alpha=0.5)
        if len(rewards) > 50:
            avg = np.convolve(rewards, np.ones(50) / 50, mode='valid')
            plt.plot(avg, label='Trung bình 50 episode', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Biểu đồ quá trình học của Q-learning Agent")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    def plot_success_rate(self):
        """Vẽ tỉ lệ thành công nếu có"""
        if not hasattr(self.logger, "success_log") or len(self.logger.success_log) == 0:
            print("Logger chưa có success_log — bỏ qua.")
            return
        success = np.array(self.logger.success_log)
        window = 50
        rolling_rate = np.convolve(success, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(rolling_rate, label='Tỉ lệ thành công trung bình (50 ep)')
        plt.xlabel("Episode")
        plt.ylabel("Tỉ lệ thành công")
        plt.title("Tỉ lệ thành công của Agent theo thời gian")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    def plot_epsilon_decay(self):
        """Vẽ quá trình giảm epsilon"""
        if not hasattr(self.logger, "epsilon_log") or len(self.logger.epsilon_log) == 0:
            print("Logger chưa ghi epsilon_log — bỏ qua.")
            return
        epsilons = self.logger.epsilon_log
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, label='Epsilon', color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon (mức độ thăm dò)")
        plt.title("Quá trình giảm Epsilon (Exploration Decay)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    def plot_loss_curve(self):
        """Vẽ biểu đồ Q-loss theo episode"""
        if not hasattr(self.logger, "loss_log") or len(self.logger.loss_log) == 0:
            print("Logger chưa ghi loss_log — bỏ qua.")
            return
        losses = self.logger.loss_log
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Loss mỗi episode', alpha=0.6)
        if len(losses) > 50:
            avg = np.convolve(losses, np.ones(50) / 50, mode='valid')
            plt.plot(avg, label='Trung bình 50 episode', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Biểu đồ Q-loss của Agent")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    def compare_agents(self, logger_list, names):
        """So sánh nhiều agent (nhiều logger khác nhau)"""
        plt.figure(figsize=(10, 5))
        for logger, name in zip(logger_list, names):
            rewards = logger.episode_rewards
            if len(rewards) > 50:
                avg = np.convolve(rewards, np.ones(50) / 50, mode='valid')
                plt.plot(avg, label=name)
            else:
                plt.plot(rewards, label=name)
        plt.xlabel("Episode")
        plt.ylabel("Reward trung bình (50 ep)")
        plt.title("So sánh hiệu suất giữa các Agent")
        plt.legend()
        plt.grid(True)
        plt.show()
