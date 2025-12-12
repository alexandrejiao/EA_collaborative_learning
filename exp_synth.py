import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys

# Seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. SETUP (Strict Synthetic Data)
# ============================================================================
print("="*80)
print("GENERATING SYNTHETIC DATA")
print("="*80)

N_AGENTS = 100
DIMENSION = 100 
GAMMA = 0.1 

# Target Models
target_models = np.zeros((N_AGENTS, DIMENSION))
target_models[:, :2] = np.random.randn(N_AGENTS, 2)
target_models /= np.linalg.norm(target_models, axis=1, keepdims=True)

# Graph
cos_sim = np.dot(target_models, target_models.T) 
W = np.exp((cos_sim - 1) / GAMMA)
W[W < 1e-3] = 0
np.fill_diagonal(W, 0)
D_ii = W.sum(axis=1)

# Datasets
train_data = []
test_data = []
for i in range(N_AGENTS):
    m_i = np.random.randint(10, 101)
    X_loc = np.random.randn(m_i, DIMENSION)
    X_loc /= np.linalg.norm(X_loc, axis=1, keepdims=True)
    logits = np.dot(X_loc, target_models[i])
    y_loc = np.sign(logits)
    y_loc[y_loc == 0] = 1 
    flip_indices = np.random.choice(m_i, size=int(0.05 * m_i), replace=False)
    y_loc[flip_indices] *= -1
    train_data.append((X_loc, y_loc))
    
    X_test = np.random.randn(100, DIMENSION)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
    y_test = np.sign(np.dot(X_test, target_models[i]))
    y_test[y_test == 0] = 1
    test_data.append((X_test, y_test))

# ============================================================================
# 2. PRIVACY SOLVER (EXACT THEOREM 1 - Min of 3 Terms)
# ============================================================================

def solve_step_epsilon_theorem1_exact(total_epsilon, k_steps, delta):
    """
    Numerically solves for step_epsilon (eps') such that:
    total_epsilon = min( Term1, Term2, Term3 )
    """
    if total_epsilon is None: return None
    
    def compute_cost(eps):
        
        exp_e = np.exp(eps)
        
        mean_term = k_steps * ( (exp_e - 1) * eps / (exp_e + 1) )
        
       
        sum_2_eps_sq = k_steps * (2 * eps**2)
        sqrt_sum_2_eps_sq = np.sqrt(sum_2_eps_sq)
        
        
        root_sum_eps_sq = np.sqrt(k_steps * eps**2)

        
        term1 = k_steps * eps
        
        
        log_arg_A = np.e + (root_sum_eps_sq / delta)
        term2 = mean_term + np.sqrt(sum_2_eps_sq * np.log(log_arg_A))
        
       
        log_arg_B = 1.0 / delta
        term3 = mean_term + np.sqrt(sum_2_eps_sq * np.log(log_arg_B))
        
        return min(term1, term2, term3)

    # Binary Search
    low = 0.0
    high = total_epsilon 
    
    for _ in range(60):
        mid = (low + high) / 2
        if mid <= 0: continue
        
        cost = compute_cost(mid)
        
        if cost > total_epsilon:
            high = mid
        else:
            low = mid
            
    return low

def sigmoid(z):
    z = np.clip(z, -20, 20)
    return 1 / (1 + np.exp(-z))

# ============================================================================
# 3. EXPERIMENT RUNNER
# ============================================================================

def run_experiment_exact_three_terms(mu, T, epsilon_total, theta_init):
    theta = theta_init.copy()
    
    m_counts = np.array([len(d[0]) for d in train_data])
    c_i = m_counts / np.max(m_counts)
    
    # L1 Clipping (Strict Laplace requirement)
    # d=100 -> L1 <= sqrt(100)*L2 = 10.0
    C_CLIP_L1 = 10.0 
    L_loc = 0.25 
    alphas = 1.0 / (1.0 + mu * c_i * (D_ii + L_loc))

    # --- PRIVACY BUDGETING ---
    MAX_UPDATES = int(T / N_AGENTS)
    agent_counts = np.zeros(N_AGENTS, dtype=int)
    
    noise_scale = None
    if epsilon_total is not None:
        DELTA_FIXED = np.exp(-5)
        
        # Solve for Exact Step Epsilon using all 3 bounds
        step_eps = solve_step_epsilon_theorem1_exact(epsilon_total, MAX_UPDATES, DELTA_FIXED)
        
        # Laplace Scale (Theorem 1): s = 2C / (eps' * m)
        noise_scale = (2.0 * C_CLIP_L1) / (step_eps * m_counts)
        print(f"  > Budget: {epsilon_total} | Steps: {MAX_UPDATES} | Step Eps: {step_eps:.6f}")
    # -------------------------

    history_acc = []
    
    for t in range(T):
        i = np.random.randint(0, N_AGENTS)
        
        # Strict stopping condition
        if epsilon_total is not None and agent_counts[i] >= MAX_UPDATES:
            pass
        else:
            X, y = train_data[i]
            
            # Gradient
            scores = np.dot(X, theta[i])
            sig_term = sigmoid(-y * scores)
            grad_mean = np.dot(sig_term * (-y), X) / len(y)
            grad_reg = (1.0 / len(y)) * theta[i]
            grad_local = grad_mean + grad_reg
            
            # L1 Clipping
            grad_l1 = np.linalg.norm(grad_local, ord=1)
            if grad_l1 > C_CLIP_L1:
                grad_local = grad_local * (C_CLIP_L1 / grad_l1)
            
            # Laplace Noise
            if noise_scale is not None:
                noise = np.random.laplace(0, noise_scale[i], size=DIMENSION)
                grad_final = grad_local + noise
            else:
                grad_final = grad_local
            
            # Update
            neighbors = np.where(W[i] > 0)[0]
            if len(neighbors) > 0:
                avg = np.average(theta[neighbors], axis=0, weights=W[i, neighbors])
            else:
                avg = theta[i]
            
            theta[i] = (1-alphas[i])*theta[i] + alphas[i]*(avg - mu*c_i[i]*grad_final)
            agent_counts[i] += 1
            
        if t % 10 == 0 or t == T-1:
            acc_sum = 0
            idx = np.random.choice(N_AGENTS, 20, replace=False)
            for ag in idx:
                Xt, yt = test_data[ag]
                p = np.sign(np.dot(Xt, theta[ag]))
                p[p==0] = 1
                acc_sum += accuracy_score(yt, p)
            history_acc.append(acc_sum / 20)
            sys.stdout.write(f"\rIter {t}/{T} | Acc: {history_acc[-1]*100:.2f}%")

    print()
    return theta, history_acc

# ============================================================================
# 4. EXECUTION
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: LOCAL INITIALIZATION")
theta_loc = np.zeros((N_AGENTS, DIMENSION))
for i in range(N_AGENTS):
    X, y = train_data[i]
    if len(np.unique(y)) > 1:
        clf = LogisticRegression(fit_intercept=False, C=len(y), solver='lbfgs')
        clf.fit(X, y)
        theta_loc[i] = clf.coef_

acc_init = 0
for i in range(N_AGENTS):
    Xt, yt = test_data[i]
    p = np.sign(np.dot(Xt, theta_loc[i]))
    acc_init += accuracy_score(yt, p)
print(f"Initial Local Accuracy: {acc_init/N_AGENTS*100:.2f}%")

print("\n" + "="*80)
print("PHASE 2: RUNNING WITH EXACT THEOREM 1 (3 TERMS)")
print("="*80)

MU = 0.2
ITERATIONS = 800

configs = [
    {'eps': None, 'label': 'Non-Private'},
    {'eps': 1.05,  'label': 'Private (ε=1.05)'},
    {'eps': 0.55,  'label': 'Private (ε=0.55)'},
    {'eps': 0.15,  'label': 'Private (ε=0.15)'}
]

plt.figure(figsize=(10, 6))

for conf in configs:
    print(f"\nRunning {conf['label']}...")
    theta_f, hist = run_experiment_exact_three_terms(
        mu=MU, T=ITERATIONS, 
        epsilon_total=conf['eps'], 
        theta_init=theta_loc
    )
    plt.plot(np.linspace(0, ITERATIONS, len(hist)), hist, label=conf['label'])

plt.axhline(y=acc_init/N_AGENTS, color='k', linestyle='--', label='Local Baseline')
plt.title('Synthetic Exp: Theorem 1 (Min of 3 Terms)', fontsize=14)
plt.ylabel('Test Accuracy')
plt.xlabel('Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('synthetic_theorem1_strict.png')
plt.show()