import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# 1. SETUP
# ============================================================================

class MnistMLP(nn.Module):
    def __init__(self):
        super(MnistMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_params_vec(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params_vec(model, vec):
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data.copy_(vec[pointer:pointer + num_param].view_as(p))
        pointer += num_param

def get_grads_vec(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters()])

# Load Data
print("Loading MNIST...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Prepare Test Data
X_test = mnist_test.data.float() / 255.0 
X_test = (X_test - 0.5) / 0.5
y_test = mnist_test.targets

# Create Agents
N_AGENTS = 20
SAMPLES_PER_AGENT = 50 
agent_data = []
all_indices = np.arange(len(mnist_data))
np.random.shuffle(all_indices)

for i in range(N_AGENTS):
    indices = all_indices[i * SAMPLES_PER_AGENT : (i + 1) * SAMPLES_PER_AGENT]
    imgs = mnist_data.data[indices].float() / 255.0
    imgs = (imgs - 0.5) / 0.5 
    targets = mnist_data.targets[indices]
    agent_data.append((imgs, targets))

# Create Graph
G = nx.watts_strogatz_graph(N_AGENTS, k=5, p=0.3, seed=42)
W = nx.to_numpy_array(G)
D_ii = W.sum(axis=1) 
D_ii[D_ii == 0] = 1.0

# Pre-train Local Models
print("Pre-training local models...")
theta_loc = np.zeros((N_AGENTS, get_params_vec(MnistMLP()).shape[0]))
criterion = nn.CrossEntropyLoss()
m_i = np.array([SAMPLES_PER_AGENT] * N_AGENTS)
c_i = m_i / np.max(m_i)

for i in range(N_AGENTS):
    model = MnistMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    X, y = agent_data[i]
    model.train()
    for epoch in range(50): 
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    theta_loc[i] = get_params_vec(model).numpy()

# ----------------------------------------------------------------------------
# [NEW] CALCULATE LOCAL BASELINE ACCURACY
# ----------------------------------------------------------------------------
print("Calculating Local Baseline Accuracy...")
local_baseline_acc = 0
model_eval = MnistMLP()
with torch.no_grad():
    for i in range(N_AGENTS):
        # Load local parameters
        set_params_vec(model_eval, torch.tensor(theta_loc[i], dtype=torch.float32))
        # Evaluate on Global Test Set
        out = model_eval(X_test)
        _, pred = torch.max(out, 1)
        acc = (pred == y_test).float().mean().item()
        local_baseline_acc += acc

local_baseline_acc /= N_AGENTS
print(f"Average Local Baseline Accuracy: {local_baseline_acc*100:.2f}%")

# ============================================================================
# 2. GAUSSIAN PRIVATE ALGORITHM
# ============================================================================

def run_gaussian_private_algo(mu, epsilon_iter, delta_iter, T, theta_init, W, D_ii, c_i, agent_data, verbose=True):
    theta = theta_init.copy()
    L_loc = 10.0
    alpha = 1.0 / (1.0 + mu * c_i * L_loc)
    
    # Clipping Constant C (L2 Norm bound)
    C_CLIP = 10.0 
    
    history_acc = []
    model = MnistMLP()
    
    for t in range(T):
        i = np.random.randint(0, N_AGENTS)
        X_i, y_i = agent_data[i]
        
        set_params_vec(model, torch.tensor(theta[i], dtype=torch.float32))
        model.zero_grad()
        
        output = model(X_i)
        loss = criterion(output, y_i)
        loss.backward()
        
        # --- GAUSSIAN PRIVACY MECHANISM ---
        
        # 1. L2 Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), C_CLIP, norm_type=2.0)
        
        grad_vec = get_grads_vec(model)
        
        # 2. Gaussian Noise Generation
        if epsilon_iter > 0:
            term_delta = np.sqrt(2 * np.log(2.0 / delta_iter))
            scale = (2.0 * C_CLIP * term_delta) / (epsilon_iter * SAMPLES_PER_AGENT)
            
            # Sample from Normal Distribution
            noise = torch.normal(mean=0.0, std=scale, size=grad_vec.shape)
            grad_private = grad_vec + noise
        else:
            grad_private = grad_vec
            
        grad_private_np = grad_private.numpy()
        
        # --- UPDATE RULE ---
        neighbor_term = np.zeros(theta.shape[1])
        neighbors = np.where(W[i] > 0)[0]
        if len(neighbors) > 0:
            for j in neighbors:
                neighbor_term += theta[j] 
            neighbor_term /= D_ii[i]
        else:
            neighbor_term = theta[i]

        term_1 = (1 - alpha[i]) * theta[i]
        term_2 = alpha[i] * neighbor_term
        term_3 = alpha[i] * mu * c_i[i] * grad_private_np
        
        theta[i] = term_1 + term_2 - term_3

        # Monitoring
        if t % 500 == 0 or t == T-1:
            acc = 0
            with torch.no_grad():
                for ag in range(N_AGENTS):
                    set_params_vec(model, torch.tensor(theta[ag], dtype=torch.float32))
                    out = model(X_test)
                    _, pred = torch.max(out, 1)
                    acc += (pred == y_test).float().mean().item()
            acc /= N_AGENTS
            history_acc.append(acc)
            
            if verbose and t % 500 == 0:
                e_str = f"ε={epsilon_iter}" if epsilon_iter else "Non-Priv"
                sys.stdout.write(f"\r {e_str} | Iter {t}/{T} | Acc: {acc*100:.2f}%")
                sys.stdout.flush()

    if verbose: print()
    return theta, history_acc[-1], history_acc

# ============================================================================
# 3. RUN EXPERIMENTS
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT: GAUSSIAN MECHANISM (delta > 0)")
print("="*80)

MU = 0.1
ITERATIONS = 5000
DELTA = 1e-5 

configs = [
    {'eps': 0.0, 'delta': 0.0,   'label': 'Non-Private'},
    {'eps': 5.0, 'delta': DELTA, 'label': f'(ε=5.0, δ={DELTA})'},
    {'eps': 1.0, 'delta': DELTA, 'label': f'(ε=1.0, δ={DELTA})'},
]

results = []

for conf in configs:
    theta_final, acc, hist = run_gaussian_private_algo(
        mu=MU,
        epsilon_iter=conf['eps'],
        delta_iter=conf['delta'],
        T=ITERATIONS,
        theta_init=theta_loc,
        W=W, D_ii=D_ii, c_i=c_i,
        agent_data=agent_data
    )
    results.append({'label': conf['label'], 'hist': hist})
    print(f"✓ {conf['label']} -> Final Acc: {acc*100:.2f}%")

# ============================================================================
# 4. PLOTTING WITH LOCAL BASELINE
# ============================================================================
plt.figure(figsize=(10, 6))

# Plot Experiment Results
for res in results:
    x_axis = np.linspace(0, ITERATIONS, len(res['hist']))
    plt.plot(x_axis, res['hist'], linewidth=2, label=res['label'])

# [NEW] Plot Local Baseline
plt.axhline(y=local_baseline_acc, color='red', linestyle='--', linewidth=2, 
            label=f'Local Baseline ({local_baseline_acc*100:.1f}%)')

plt.title('Decentralized MNIST Classification', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()