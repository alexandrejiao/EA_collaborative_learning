import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys

# Seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. SETUP 
# ============================================================================
print("="*80)
print("GENERATING SYNTHETIC DATA & ADVERSARIAL SETUP")
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
# 2. ROBUST EXPERIMENT RUNNER
# ============================================================================

def sigmoid(z):
    z = np.clip(z, -20, 20)
    return 1 / (1 + np.exp(-z))


def run_adversarial_experiment(mu, T, theta_init, malicious_frac=0.0, use_robust_agg=False):
    """
    Runs the P2P learning loop with potential adversaries and defense.
    """
    theta = theta_init.copy()
    
    m_counts = np.array([len(d[0]) for d in train_data])
    c_i = m_counts / np.max(m_counts)
    L_loc = 0.25 
    alphas = 1.0 / (1.0 + mu * c_i * (D_ii + L_loc))

    # Identify Malicious Agents
    n_malicious = int(N_AGENTS * malicious_frac)
    malicious_indices = np.random.choice(N_AGENTS, n_malicious, replace=False)
    
    print(f" > Setup: {n_malicious} Malicious Agents | Defense: {'MEDIAN' if use_robust_agg else 'MEAN'}")

    history_acc = []
    
    for t in range(T):
        i = np.random.randint(0, N_AGENTS)
        
        # --- MALICIOUS BEHAVIOR ---
        if i in malicious_indices:
            # Set model to random noise
            theta[i] = np.random.randn(DIMENSION) * 5.0
            
            
        # --- HONEST BEHAVIOR ---
        else:
            X, y = train_data[i]
            
            # Gradient Calculation
            scores = np.dot(X, theta[i])
            sig_term = sigmoid(-y * scores)
            grad_mean = np.dot(sig_term * (-y), X) / len(y)
            grad_reg = (1.0 / len(y)) * theta[i]
            grad_final = grad_mean + grad_reg 
            
            # --- AGGREGATION STEP ---
            neighbors = np.where(W[i] > 0)[0]
            
            if len(neighbors) > 0:
                neighbor_thetas = theta[neighbors]
                
                if use_robust_agg:
                   
                    avg = np.median(neighbor_thetas, axis=0)
                else:
                    avg = np.average(neighbor_thetas, axis=0, weights=W[i, neighbors])
            else:
                avg = theta[i]
            
            # Update Rule
            theta[i] = (1-alphas[i])*theta[i] + alphas[i]*(avg - mu*c_i[i]*grad_final)
        
         
        if t % 20 == 0 or t == T-1:
            honest_indices = [x for x in range(N_AGENTS) if x not in malicious_indices]
            if len(honest_indices) == 0:
                history_acc.append(0.0)
            else:
                eval_idx = np.random.choice(honest_indices, min(20, len(honest_indices)), replace=False)
                
                acc_sum = 0
                for ag in eval_idx:
                    Xt, yt = test_data[ag]
                    p = np.sign(np.dot(Xt, theta[ag]))
                    p[p==0] = 1
                    acc_sum += accuracy_score(yt, p)
                
                current_acc = acc_sum / len(eval_idx)
                history_acc.append(current_acc)
                sys.stdout.write(f"\rIter {t}/{T} | Honest Acc: {current_acc*100:.2f}%")

    print()
    return history_acc
# ============================================================================
# 3. EXECUTION
# ============================================================================
print("\n" + "="*80)
print("INITIALIZATION")
theta_loc = np.zeros((N_AGENTS, DIMENSION))
for i in range(N_AGENTS):
    X, y = train_data[i]
    if len(np.unique(y)) > 1:
        clf = LogisticRegression(fit_intercept=False, C=len(y), solver='lbfgs')
        clf.fit(X, y)
        theta_loc[i] = clf.coef_


MU = 0.2
ITERATIONS = 1500

# Experiment 1: Baseline (No Attack)
print("\n--- 1. Baseline (No Attack) ---")
hist_baseline = run_adversarial_experiment(MU, ITERATIONS, theta_loc, 
                                           malicious_frac=0.0, use_robust_agg=False)

# Experiment 2: Under Attack (No Defense)
print("\n--- 2. Under Attack (20% Malicious, Standard Mean) ---")
hist_attack = run_adversarial_experiment(MU, ITERATIONS, theta_loc, 
                                         malicious_frac=0.35, use_robust_agg=False)

# Experiment 3: Under Attack (With Robust Defense)
print("\n--- 3. Under Attack (20% Malicious, Robust Median) ---")
hist_defense = run_adversarial_experiment(MU, ITERATIONS, theta_loc, 
                                          malicious_frac=0.35, use_robust_agg=True)

# ============================================================================
# 4. PLOTTING
# ============================================================================
plt.figure(figsize=(10, 6))

x_vals = np.linspace(0, ITERATIONS, len(hist_baseline))

plt.plot(x_vals, hist_baseline, 'g--', label='Baseline (No Attack)', linewidth=2)
plt.plot(x_vals, hist_attack, 'r-', label='Attack (Standard Mean)', alpha=0.7)
plt.plot(x_vals, hist_defense, 'b-', label='Attack + Defense (Median)', linewidth=2)

plt.title('Adversarial Robustness: Mean vs. Median Aggregation', fontsize=14)
plt.ylabel('Test Accuracy (Honest Agents)')
plt.xlabel('Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('adversarial_robustness.png')
plt.show()