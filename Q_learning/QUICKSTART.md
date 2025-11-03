#  Quick Start Guide

## Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài từng package:
```bash
pip install pygame numpy torch matplotlib
```

## Bước 2: Test setup

```bash
python test_setup.py
```

Nếu tất cả , bạn đã sẵn sàng!

## Bước 3: Thử chơi thủ công

```bash
python main.py play
```

**Điều khiển:**
- `←` : Sang lane trái
- `→` : Sang lane phải  
- Tránh vật đỏ, ăn vật xanh!

## Bước 4: Train AI

### Option 1: Quick Train (Recommended cho lần đầu)
```bash
python main.py quick
```
→ Train 1000 episodes với cấu hình tốt nhất

### Option 2: Custom Training
```bash
# DQN mode (mạnh nhất)
python main.py train --mode dqn --episodes 2000

# Tabular Q-Learning mode (nhanh hơn, đơn giản hơn)
python main.py train --mode tabular --episodes 3000

# Training không hiển thị (nhanh gấp 3-5 lần)
python main.py train --mode dqn --episodes 5000 --no-render
```

## Bước 5: Xem AI chơi

```bash
python main.py continue models/checkpoint_ep500.pkl --episodes 1000
```

## Bước 6: Đánh giá performance

```bash
python main.py eval models/best_score_model.pkl --eval-episodes 20
```

---

##  Hiểu kết quả training

### Metrics quan trọng:

- **Score**: Số goals (xanh) đã thu thập → càng cao càng tốt
- **Reward**: Tổng reward trong episode → càng cao càng tốt  
- **Success Rate**: % episodes không bị collision → > 30% là tốt
- **Epsilon (ε)**: Tỉ lệ explore → giảm dần từ 1.0 → 0.01

### Training tốt trông như thế nào?

```
Ep   100 | Reward:   -8.45 | Score:   0 | ε: 0.606
Ep   200 | Reward:    2.31 | Score:   1 | ε: 0.367  ← Bắt đầu học!
Ep   500 | Reward:   15.67 | Score:   4 | ε: 0.081
Ep  1000 | Reward:   45.32 | Score:  12 | ε: 0.010  ← Đã giỏi!
```

### Training không tốt (cần fix):

```
Ep  1000 | Reward:  -15.00 | Score:   0 | ε: 0.010  ← Không học được gì
```

**Cách fix:**
- Giảm `epsilon_decay` từ 0.995 → 0.99 (explore nhiều hơn)
- Tăng `learning_rate` từ 1e-3 → 5e-3
- Thử `--hybrid` mode (dùng A*)

---

##  Tips để train tốt hơn

### 1. Để agent học nhanh:
```bash
python main.py train --mode dqn --episodes 3000 --hybrid
```
→ Hybrid mode dùng A* path planning để bootstrap learning

### 2. Để train nhanh (không cần xem):
```bash
python main.py train --mode dqn --episodes 5000 --no-render
```
→ Nhanh gấp 3-5 lần!

### 3. Continue từ checkpoint tốt:
```bash
python main.py continue models/checkpoint_ep500.pkl --episodes 1000
```

### 4. So sánh Tabular vs DQN:
```bash
# Train tabular
python main.py train --mode tabular --episodes 3000

# Train DQN  
python main.py train --mode dqn --episodes 2000

# Compare
python main.py eval models/best_score_model.pkl
```

---

##  Troubleshooting

### Lỗi: "No module named 'env'"
```bash
# Chạy từ root directory của project
cd /path/to/project
python main.py play
```

### Lỗi: "CUDA out of memory"
```bash
# Trong advanced_agent.py, thêm dòng này:
device = 'cpu'  # Force CPU

# Hoặc giảm batch_size:
python main.py train --mode dqn --episodes 2000
# Sửa trong trainer: batch_size=32 thay vì 64
```

### Agent không học được gì:
1. Check epsilon - phải giảm dần: 1.0 → 0.01
2. Check reward - phải tăng dần qua các episodes
3. Thử `--hybrid` mode
4. Giảm epsilon_decay

### Training quá chậm:
1. Dùng `--no-render`
2. Giảm `--render-interval`  
3. Dùng GPU nếu có (DQN mode)
4. Giảm `batch_size`

---

##  File structure sau khi train

```
project/
├── models/
│   ├── best_score_model.pkl      ← Model tốt nhất (score)
│   ├── best_reward_model.pkl     ← Model tốt nhất (reward)
│   ├── checkpoint_ep100.pkl      ← Checkpoints
│   ├── checkpoint_ep200.pkl
│   └── final_model.pkl           ← Model cuối cùng
│
├── logs/
│   ├── q_table_ep100.png         ← Q-table visualization
│   ├── value_map_ep100.png       ← Value map (DQN)
│   └── training_plots.png        ← Training curves
│
└── ...
```

---







##  Next Steps ?????

1. **Improve performance**:
   - Tune hyperparameters
   - Try longer training (5000+ episodes)
   - Experiment with reward function

2. **Add features**:
   - Multiple goal types
   - Power-ups
   - Different obstacle patterns
   - Difficulty levels

3. **Try other algorithms**:
   - PPO (Proximal Policy Optimization)
   - A3C (Asynchronous Actor-Critic)
   - SAC (Soft Actor-Critic)

---

**Good luck training! **