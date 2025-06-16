import gymnasium as gym
import numpy as np

# Criar o ambiente
env = gym.make("FrozenLake-v1", is_slippery=False)  # ambiente determinístico

# Hiperparâmetros
n_episodes = 2000
max_steps = 100
alpha = 0.1    # taxa de aprendizado
gamma = 0.99   # fator de desconto
epsilon = 1.0  # taxa de exploração inicial
eps_decay = 0.995
eps_min = 0.01

# Inicializar a Q‑table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Loop principal
for episode in range(n_episodes):
    state, info = env.reset()
    done = False

    for _ in range(max_steps):
        # Política ε-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Atualização da Q‑value
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state
        if done:
            break

    # Decaimento da exploração
    epsilon = max(epsilon * eps_decay, eps_min)

print("Q‑table final:")
print(Q)

# Testando agente
state, info = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
print("Reward:", reward)