import gymnasium as gym
import numpy as np
import pygame # Importe a biblioteca Pygame

# Criar o ambiente
# Para visualizar o ambiente
#env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

# Hiperparâmetros
n_episodes = 10000 # Aumente este número para ver o aprendizado de forma mais clara
max_steps = 100
alpha = 0.1    # taxa de aprendizado
gamma = 0.99   # fator de desconto
epsilon = 1.0  # taxa de exploração inicial
eps_decay = 0.995
eps_min = 0.01

# Inicializar a Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Loop principal de treinamento
print("Iniciando treinamento...")
for episode in range(n_episodes):
    state, info = env.reset()
    done = False

    for step in range(max_steps): 
        # Renderiza o ambiente para visualização durante o treinamento
        # É importante processar eventos Pygame aqui
        #env.render()
        #for event in pygame.event.get(): # <--- Adição chave para processar eventos Pygame
            #if event.type == pygame.QUIT: # Se o evento for o de fechar a janela
                #done = True # Sinaliza para parar o episódio
                #break # Sai do loop de eventos
        #if done: # Se a janela foi fechada pelo usuário
            #break # Sai do loop de passos

        # Política ε-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Atualização da Q-value
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state
        if done:
            break

    # Decaimento da exploração
    epsilon = max(epsilon * eps_decay, eps_min)
    print(f"Episódio {episode + 1}/{n_episodes} - Epsilon: {epsilon:.4f}")

print("\nTreinamento finalizado.")
print("Q-table final:")
print(Q)

# ---

# Testando o agente treinado
print("\nTestando o agente treinado...")
state, info = env.reset()
done = False
total_reward = 0
while not done:
    #env.render()
    #for event in pygame.event.get(): # <--- Adição chave também para o loop de teste
        #if event.type == pygame.QUIT:
            #done = True
            #break
    #if done:
        #break

    action = np.argmax(Q[state])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward # Acumula a recompensa

print(f"Teste finalizado. Recompensa total: {total_reward}")

env.close() # Garante que o ambiente seja fechado corretamente
print("Ambiente fechado.")

#A ORDEM EH ESQUERDA, BAIXO, DIREITA, CIMA PARA A Q-TABLE