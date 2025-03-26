import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LinearCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.activation = lambda x: F.elu(x) + 1

    def forward(self, q, k, v):
        Q = self.activation(self.query_proj(q))
        K = self.activation(self.key_proj(k))
        V = self.value_proj(v)

        kv = torch.einsum('bld,blh->bdh', K, V)
        z = torch.einsum('bld,bd->bl', Q, K.sum(dim=1)) + 1e-6
        out = torch.einsum('bld,bdh->blh', Q, kv)
        return out / z.unsqueeze(-1)

class BiModalClassifier(nn.Module):
    def __init__(self, dim=64, len_a=10, len_b=5, use_attention=True):
        super().__init__()
        self.dim = dim
        self.use_attention = use_attention

        self.embed_a = nn.Linear(10, dim)
        self.embed_b = nn.Linear(6, dim)

        self.pos_a = nn.Parameter(torch.randn(1, len_a, dim))
        self.pos_b = nn.Parameter(torch.randn(1, len_b, dim))

        if use_attention:
            self.attn_ab = LinearCrossAttention(dim)
            self.attn_ba = LinearCrossAttention(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, a, b):
        a_embed = self.embed_a(a) + self.pos_a
        b_embed = self.embed_b(b) + self.pos_b

        if self.use_attention:
            ab = self.attn_ab(a_embed, b_embed, b_embed).mean(dim=1)  # A attends to B
            ba = self.attn_ba(b_embed, a_embed, a_embed).mean(dim=1)  # B attends to A
        else:
            ab = a_embed.mean(dim=1)
            ba = b_embed.mean(dim=1)

        joint = torch.cat([ab, ba], dim=-1)
        return torch.sigmoid(self.classifier(joint)).squeeze(-1)



def generate_batch(batch_size=32, len_a=10, len_b=5, labels=None):
    """
    Задача: есть две последовательности: a, b
    Если метка батча 1, то последовательность b содержит в себе зашумленную копию a
    Если метка батча 0, последовательности никак не связаны

    :param batch_size:
    :param len_a:
    :param len_b:
    :return:
    """
    a = torch.randn(batch_size, len_a, 10)
    b = torch.randn(batch_size, len_b, 6)
    if not labels:
        labels = torch.randint(0, 2, (batch_size,)).float()
    else:
        labels = torch.tensor(labels)

    for i in range(batch_size):
        if labels[i] == 1:
            indices = torch.randint(0, len_a, (2,))
            b[i, 0] = a[i, indices[0], :6] + 0.01 * torch.randn(6)
            b[i, 1] = a[i, indices[1], :6] + 0.01 * torch.randn(6)
        else:
            b[i] = torch.randn(len_b, 6)

    return a, b, labels

def train_model(use_attention=True, steps=20000):
    model = BiModalClassifier(use_attention=use_attention)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    losses, accuracies = [], []

    for step in range(steps):
        a, b, y = generate_batch()
        pred = model(a, b)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = ((pred > 0.5) == y).float().mean()
        losses.append(loss.item())
        accuracies.append(acc.item())

        if step % 100 == 0:
            print(f"[{'ATTN' if use_attention else 'BASE'}] Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc.item():.2f}")

    return model, losses, accuracies


def visualize_attention_weights(model, a, b):
    model.eval()

    a_embed = model.embed_a(a) + model.pos_a  # (1, La, D)
    b_embed = model.embed_b(b) + model.pos_b  # (1, Lb, D)

    phi = lambda x: F.elu(x) + 1
    Q = phi(model.attn_ab.query_proj(a_embed))  # (1, La, D)
    K = phi(model.attn_ab.key_proj(b_embed))    # (1, Lb, D)

    attn_weights = torch.einsum('lid,ljd->lij', Q, K).squeeze(0)  # (La, Lb)
    attn_norm = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_norm.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("Approximate Linear Cross-Attention Weights (A attends to B)")
    plt.xlabel("B tokens (keys)")
    plt.ylabel("A tokens (queries)")
    plt.show()

if __name__ == "__main__":

    model, loss_attn, acc_attn = train_model(use_attention=True)
    # model, loss_base, acc_base = train_model(use_attention=False)
    #
    #
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_attn, label='With Attention')
    # plt.plot(loss_base, label='Without Attention')
    # plt.title('Loss')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(acc_attn, label='With Attention')
    # plt.plot(acc_base, label='Without Attention')
    # plt.title('Accuracy')
    # plt.legend()
    #
    # plt.suptitle("Linear Cross-Attention vs Baseline")
    # plt.show()

    a, b, labels = generate_batch(batch_size=1, labels=[1])
    visualize_attention_weights(model, a, b)

