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

# Pre-train Local Models (Warm Start)
print("Pre-training local models (Warm Start)...")
temp_model = MnistMLP()
param_dim = get_params_vec(temp_model).shape[0]
theta_loc = np.zeros((N_AGENTS, param_dim))
criterion = nn.CrossEntropyLoss()
m_i = np.array([SAMPLES_PER_AGENT] * N_AGENTS)
c_i = m_i / np.max(m_i) # Normalized confidence

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

# ============================================================================
# 2. PRIVATE ALGORITHM IMPLEMENTATION 
# ============================================================================

def run_private_bellet_algo(mu, epsilon_iter, T, theta_init, W, D_ii, c_i, agent_data, verbose=True):
    """
    epsilon_iter: Privacy budget per iteration (determines noise scale)
    """
    theta = theta_init.copy()
    L_loc = 10.0 # Estimated local Lipschitz constant for alpha calculation
    
    # Precompute alpha (Step size)
    alpha = 1.0 / (1.0 + mu * c_i * L_loc)
    
    # Clipping Constant C (from Supplementary Material D.2)
    #
    C_CLIP = 10.0 
    
    history_acc = []
    model = MnistMLP()
    
    for t in range(T):

        i = np.random.randint(0, N_AGENTS)
       
        X_i, y_i = agent_data[i]
        
        # Load params
        set_params_vec(model, torch.tensor(theta[i], dtype=torch.float32))
        model.zero_grad()
        
        # Forward/Backward
        output = model(X_i)
        loss = criterion(output, y_i)
        loss.backward()
        
        
        # We clip the gradient norm to C_CLIP to ensure sensitivity is bounded
        torch.nn.utils.clip_grad_norm_(model.parameters(), C_CLIP, norm_type=1.0)
        
        # Get flattened gradient
        grad_vec = get_grads_vec(model)
        
        # B. Noise Generation (Theorem 1)
        # Scale s_i = 2 * L0 / (epsilon * m_i)
        if epsilon_iter > 0:
            scale = (2 * C_CLIP) / (epsilon_iter * SAMPLES_PER_AGENT)
            noise = torch.distributions.Laplace(0, scale).sample(grad_vec.shape)
            
            #Add Noise to Gradient
            
            grad_private = grad_vec + noise
        else:
            grad_private = grad_vec 
            
        grad_private_np = grad_private.numpy()
        
       

        # Consensus Term (Neighbor Average)
        neighbor_term = np.zeros(param_dim)
        neighbors = np.where(W[i] > 0)[0]
        
        if len(neighbors) > 0:
            for j in neighbors:
                neighbor_term += theta[j] 
            neighbor_term /= D_ii[i]
        else:
            neighbor_term = theta[i]

        # 4. Update Rule (
        # Θ(t+1) = (1 - α)Θ(t) + α * ( Neighbor_Avg - μ * c_i * (∇L + η) )
        
        term_1 = (1 - alpha[i]) * theta[i]
        term_2 = alpha[i] * neighbor_term
        term_3 = alpha[i] * mu * c_i[i] * grad_private_np
        
        theta[i] = term_1 + term_2 - term_3

        # Monitoring
        if t % 500 == 0 or t == T-1:
            acc = 0
            # Helper to eval
            with torch.no_grad():
                for ag in range(N_AGENTS):
                    set_params_vec(model, torch.tensor(theta[ag], dtype=torch.float32))
                    out = model(X_test)
                    _, pred = torch.max(out, 1)
                    acc += (pred == y_test).float().mean().item()
            acc /= N_AGENTS
            history_acc.append(acc)
            
            if verbose and t % 1000 == 0:
                e_str = f"ε={epsilon_iter:.2f}" if epsilon_iter else "Non-Priv"
                sys.stdout.write(f"\r {e_str} | Iter {t}/{T} | Acc: {acc*100:.2f}%")
                sys.stdout.flush()

    if verbose: print()
    return theta, history_acc[-1], history_acc

# ============================================================================
# 3. RUN EXPERIMENTS: 
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT: PRIVACY-UTILITY TRADE-OFF")
print("="*80)

MU = 0.1
ITERATIONS = 5000


configs = [
    {'eps': 0.0,   'label': 'Non-Private (Baseline)'},
    {'eps': 5.0,   'label': 'ε=5.0 (Low Noise)'},
    {'eps': 1.0,   'label': 'ε=1.0 (Med Noise)'},
    {'eps': 0.2,   'label': 'ε=0.2 (High Noise)'} 
]

results = []

for conf in configs:
    theta_final, acc, hist = run_private_bellet_algo(
        mu=MU,
        epsilon_iter=conf['eps'],
        T=ITERATIONS,
        theta_init=theta_loc,
        W=W, D_ii=D_ii, c_i=c_i,
        agent_data=agent_data
    )
    results.append({'label': conf['label'], 'hist': hist})
    print(f"✓ {conf['label']} -> Final Acc: {acc*100:.2f}%")

# ============================================================================
# 4. VISUALIZATION
# ============================================================================

plt.figure(figsize=(10, 6))


for res in results:
    x_axis = np.linspace(0, ITERATIONS, len(res['hist']))
    plt.plot(x_axis, res['hist'], linewidth=2, label=res['label'])


acc_isolated = 0
with torch.no_grad():
    temp_m = MnistMLP()
    for i in range(N_AGENTS):
        set_params_vec(temp_m, torch.tensor(theta_loc[i], dtype=torch.float32))
        out = temp_m(X_test)
        _, pred = torch.max(out, 1)
        acc_isolated += (pred == y_test).float().mean().item()
acc_isolated /= N_AGENTS

plt.axhline(y=acc_isolated, color='black', linestyle='--', label='Isolated (No Collab)')
plt.title('Privacy-Utility Trade-off (Bellet et al. Algo)', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bellet_privacy_tradeoff.png')
plt.show()