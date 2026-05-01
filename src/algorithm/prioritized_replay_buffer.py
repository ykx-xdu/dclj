import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        # 循环覆盖存储，维持固定容量
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6):
        # α 控制“抽样时有多偏爱高 TD 误差样本”，而 β 控制“更新网络时要不要、以及在多大程度上抵消这种偏爱带来的估计偏差”。
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def add(self, transition):
        # 将新经验以当前最大优先级插入，确保新经验被采样到
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probs = np.array(priorities) / self.tree.total()
        # 按照优先级计算重要性采样权重，beta逐渐增大抵消偏差
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()
        # 更新 beta，使采样偏差校正随着训练逐步增大
        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch, idxs, is_weights

    def update_priorities(self, idxs, priorities):
        # 根据新的TD误差更新对应经验的优先级
        priorities = np.power(priorities + 1e-5, self.alpha)
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)
            # 动态维护max_priority
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self):
        """返回当前存储的经验数量"""
        return self.tree.n_entries
