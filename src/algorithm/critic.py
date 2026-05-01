import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, num_agents, hidden_dim=1024, dropout_rate=0.1):
        super().__init__()
        self.num_agents = num_agents
        input_dim = (total_state_dim // num_agents) + (total_action_dim // num_agents)
        # 编码器：逐智能体编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
        )

        self.dropout = nn.Dropout(dropout_rate)

        # 多头自注意力
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.ln_attn = nn.LayerNorm(hidden_dim)
        # 注意力聚合层
        self.attn_agg = nn.Linear(hidden_dim, 1)
        # 全连接 Q 输出
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        # states/actions: (batch, num_agents, state_dim/action_dim)
        x = torch.cat([states, actions], dim=-1)  # (batch, num_agents, state+action)
        encoded = self.encoder(x)  # (batch, num_agents, hidden_dim)
        encoded = self.dropout(encoded)  # 添加Dropout
        # 自注意力(Q=K=V)
        attn_out, _ = self.attn(encoded, encoded, encoded)
        # 残差+归一化
        attn_out = self.ln_attn(encoded + attn_out)
        # 加权求和
        scores = self.attn_agg(attn_out)       # (batch, num_agents, 1)
        weights = torch.softmax(scores, dim=1) # 对智能体维度归一化
        aggregated = (weights * attn_out).sum(dim=1)  # (batch, hidden_dim)
        # 输出全局 Q 值
        q_value = self.fc_final(aggregated)   # (batch, 1)
        return q_value
