import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import warnings

# Suppress the deprecation warning for seaborn
warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Functions ---
def clip(value, min_val, max_val):
    """Clips a value to be within a specified range."""
    return max(min_val, min(value, max_val))

def calculate_ucb_bonus(t, visits):
    """Calculates the UCB exploration bonus for an arm."""
    if visits == 0:
        return float('inf') 
    return np.sqrt(2 * np.log(max(1, t)) / visits)

# --- Agent Class ---
class Agent:
    def __init__(self, agent_id, num_arms, initial_trust, assigned_bias, network_in, true_arm_means, use_trust_weighting):
        """
        Initializes an agent with its properties and parameters.
        The assigned_bias is a direct value.
        """
        self.id = agent_id
        self.num_arms = num_arms
        self.initial_trust = initial_trust
        self.true_arm_means = true_arm_means 

        self.local_visits = np.zeros(num_arms)
        self.local_rewards_sum = np.zeros(num_arms)

        self.bias = assigned_bias 
        self.trust_scores = defaultdict(lambda: self.initial_trust) 
        self.in_neighbors = network_in.get(agent_id, [])
        self.last_pull_info = {
            'arm': -1, 
            'reward': 0.5, 
            'perceived_means_snapshot': np.full(num_arms, 0.5), 
            'local_visits_at_pull': 0 
        } 
        self.local_means = np.full(self.num_arms, 0.5)
        self.current_perceived_means = np.full(self.num_arms, 0.5)
        self.cumulative_regret = 0.0
        self.use_trust_weighting = use_trust_weighting

    def aggregate_and_perceive_beliefs(self, all_agents_last_pull_info, social_influence_weight):
        """
        Aggregates local and social beliefs, then applies agent's perception bias.
        """
        own_means = np.zeros(self.num_arms)
        for k in range(self.num_arms):
            if self.local_visits[k] > 0:
                own_means[k] = self.local_rewards_sum[k] / self.local_visits[k]
            else:
                own_means[k] = 0.5
        
        self.local_means = own_means.copy()

        social_signal_numerator = np.zeros(self.num_arms)
        social_signal_denominator = np.zeros(self.num_arms)
        any_social_data_received = False 

        for neighbor_id in self.in_neighbors:
            if neighbor_id in all_agents_last_pull_info:
                neighbor_info = all_agents_last_pull_info[neighbor_id]
                
                if 'perceived_means_snapshot' in neighbor_info and neighbor_info['perceived_means_snapshot'] is not None:
                    any_social_data_received = True
                    for k in range(self.num_arms): 
                        neighbor_perceived_mean_for_arm_k = neighbor_info['perceived_means_snapshot'][k]
                        weight = 0.0
                        if self.use_trust_weighting:
                            trust = self.trust_scores[neighbor_id]
                            weight = trust 
                        else:
                            # For unweighted, every neighbor has a weight of 1
                            weight = 1.0
                        social_signal_numerator[k] += weight * neighbor_perceived_mean_for_arm_k 
                        social_signal_denominator[k] += weight 
            
        social_mean_estimates = np.zeros(self.num_arms)
        for k in range(self.num_arms):
            if social_signal_denominator[k] > 0:
                social_mean_estimates[k] = social_signal_numerator[k] / social_signal_denominator[k]
            else:
                social_mean_estimates[k] = 0.5

        for k in range(self.num_arms):
            combined_mean_no_bias = 0.0
            own_data_available = self.local_visits[k] > 0
            
            if own_data_available and social_signal_denominator[k] > 0 and any_social_data_received:
                combined_mean_no_bias = (1 - social_influence_weight) * self.local_means[k] + \
                                        social_influence_weight * social_mean_estimates[k]
            elif own_data_available:
                combined_mean_no_bias = self.local_means[k]
            elif social_signal_denominator[k] > 0 and any_social_data_received:
                combined_mean_no_bias = social_mean_estimates[k]
            else:
                combined_mean_no_bias = 0.5

            self.current_perceived_means[k] = clip(combined_mean_no_bias + self.bias[k], 0, 1)

    def select_arm(self, current_timestep):
        """Selects an arm using the UCB criterion based on perceived means."""
        ucb_values = np.zeros(self.num_arms)
        for k in range(self.num_arms):
            ucb_bonus = calculate_ucb_bonus(current_timestep, self.local_visits[k])
            ucb_values[k] = self.current_perceived_means[k] + ucb_bonus
        
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update_local_statistics_and_share_info(self, chosen_arm, reward):
        """
        Updates local statistics and shares the agent's pure local belief state from *before* the current pull.
        """
        self.local_visits[chosen_arm] += 1
        self.local_rewards_sum[chosen_arm] += reward
        
        self.last_pull_info = {
            'arm': chosen_arm, 
            'reward': reward, 
            'perceived_means_snapshot': self.current_perceived_means.copy(), 
            'local_visits_at_pull': self.local_visits[chosen_arm] 
        }

    def update_trust_scores(self, chosen_arm, actual_reward, learning_rate_trust, all_agents_last_pull_info):
        """
        Updates trust scores for incoming neighbors based on how well THEIR shared *perceived* belief
        for the chosen arm predicted the actual reward.
        """
        if self.use_trust_weighting:
            for neighbor_id in self.in_neighbors:
                if neighbor_id in all_agents_last_pull_info:
                    neighbor_info = all_agents_last_pull_info[neighbor_id]
                    
                    if 'perceived_means_snapshot' in neighbor_info and neighbor_info['perceived_means_snapshot'] is not None:
                        neighbor_perceived_mean_for_chosen_arm = neighbor_info['perceived_means_snapshot'][chosen_arm]
                        
                        error = abs(actual_reward - neighbor_perceived_mean_for_chosen_arm)
                        feedback_signal = 1 - error 
                        
                        self.trust_scores[neighbor_id] = (1 - learning_rate_trust) * self.trust_scores[neighbor_id] + learning_rate_trust * feedback_signal
                        self.trust_scores[neighbor_id] = clip(self.trust_scores[neighbor_id], 0, 1)

    def update_regret(self, chosen_arm, actual_reward, optimal_arm_mean):
        """Updates the agent's cumulative regret."""
        self.cumulative_regret += (optimal_arm_mean - actual_reward)

# --- Environment / Simulation Setup ---
def setup_network(num_agents, topology_type="fully_connected"):
    """Sets up the network topology for the simulation."""
    in_neighbors = defaultdict(list)
    out_neighbors = defaultdict(list)

    if topology_type == "fully_connected":
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    in_neighbors[i].append(j)
                    out_neighbors[i].append(j)
    elif topology_type == "directed_star":
        if num_agents > 1:
            for i in range(1, num_agents):
                in_neighbors[i].append(0)
                out_neighbors[0].append(i) 
                in_neighbors[0].append(i)   
                out_neighbors[i].append(0)  
    elif topology_type == "directed_path":
        for i in range(num_agents):
            if i > 0:
                in_neighbors[i].append(i - 1)
                out_neighbors[i - 1].append(i)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    return in_neighbors, out_neighbors

def run_simulation(N_AGENTS, K_ARMS, TIMESTEPS, NUM_TRIALS,
                   INITIAL_TRUST_SCORE, FIXED_BIASED_VALUE, LEARNING_RATE_TRUST, REWARD_STD_DEV,
                   NETWORK_TOPOLOGY, SOCIAL_INFLUENCE_WEIGHT, PROPORTION_UNBIASED_AGENTS, USE_TRUST_WEIGHTING, SEED_OFFSET=0):
    """
    Helper function to run a simulation and return regret data.
    """
    all_cumulative_regret_histories = []
    
    for trial in range(NUM_TRIALS):
        trial_seed = SEED_OFFSET + trial if SEED_OFFSET is not None else None
        if trial_seed is not None:
            np.random.seed(trial_seed)
            random.seed(trial_seed)

        true_arm_means = np.random.rand(K_ARMS)
        optimal_arm_mean = np.max(true_arm_means)
        in_neighbors, _ = setup_network(N_AGENTS, NETWORK_TOPOLOGY)

        agents = []
        num_unbiased = max(1, int(N_AGENTS * PROPORTION_UNBIASED_AGENTS))
        current_unbiased_agent_ids = random.sample(range(N_AGENTS), num_unbiased)

        for i in range(N_AGENTS):
            bias_vector = np.zeros(K_ARMS)
            if i not in current_unbiased_agent_ids:
                bias_vector.fill(FIXED_BIASED_VALUE)
            
            agents.append(Agent(i, K_ARMS, INITIAL_TRUST_SCORE, bias_vector, 
                                in_neighbors, true_arm_means, use_trust_weighting=USE_TRUST_WEIGHTING))

        cumulative_regret_history = []
        for t in range(TIMESTEPS):
            all_agents_last_pull_info = {agent.id: agent.last_pull_info for agent in agents}

            for agent_i in agents:
                agent_i.aggregate_and_perceive_beliefs(all_agents_last_pull_info, SOCIAL_INFLUENCE_WEIGHT)
                chosen_arm = agent_i.select_arm(t)
                
                actual_reward = np.random.normal(true_arm_means[chosen_arm], REWARD_STD_DEV)
                actual_reward = clip(actual_reward, 0, 1) 
                
                agent_i.update_local_statistics_and_share_info(chosen_arm, actual_reward)
                agent_i.update_trust_scores(chosen_arm, actual_reward, LEARNING_RATE_TRUST, all_agents_last_pull_info)
                agent_i.update_regret(chosen_arm, actual_reward, optimal_arm_mean)
            
            cumulative_regret_history.append(sum(a.cumulative_regret for a in agents))
        
        all_cumulative_regret_histories.append(cumulative_regret_history)
    
    return np.mean(all_cumulative_regret_histories, axis=0), np.std(all_cumulative_regret_histories, axis=0)


def run_and_compare_weighted_vs_unweighted(N_AGENTS, K_ARMS, TIMESTEPS, NUM_TRIALS,
                                           INITIAL_TRUST_SCORE, FIXED_BIASED_VALUE, LEARNING_RATE_TRUST, REWARD_STD_DEV,
                                           NETWORK_TOPOLOGY, SOCIAL_INFLUENCE_WEIGHT, PROPORTION_UNBIASED_AGENTS, SEED=None):
    """
    Runs two simulations and plots the results to compare weighted vs. unweighted social learning.
    """
    print(f"Running weighted (trust-based) simulation...")
    mean_regret_weighted, std_regret_weighted = run_simulation(
        N_AGENTS=N_AGENTS, K_ARMS=K_ARMS, TIMESTEPS=TIMESTEPS, NUM_TRIALS=NUM_TRIALS,
        INITIAL_TRUST_SCORE=INITIAL_TRUST_SCORE, FIXED_BIASED_VALUE=FIXED_BIASED_VALUE, LEARNING_RATE_TRUST=LEARNING_RATE_TRUST, REWARD_STD_DEV=REWARD_STD_DEV,
        NETWORK_TOPOLOGY=NETWORK_TOPOLOGY, SOCIAL_INFLUENCE_WEIGHT=SOCIAL_INFLUENCE_WEIGHT, PROPORTION_UNBIASED_AGENTS=PROPORTION_UNBIASED_AGENTS,
        USE_TRUST_WEIGHTING=True,
        SEED_OFFSET=SEED if SEED is not None else 0
    )

    print(f"Running unweighted (naive) simulation...")
    mean_regret_unweighted, std_regret_unweighted = run_simulation(
        N_AGENTS=N_AGENTS, K_ARMS=K_ARMS, TIMESTEPS=TIMESTEPS, NUM_TRIALS=NUM_TRIALS,
        INITIAL_TRUST_SCORE=INITIAL_TRUST_SCORE, FIXED_BIASED_VALUE=FIXED_BIASED_VALUE, LEARNING_RATE_TRUST=LEARNING_RATE_TRUST, REWARD_STD_DEV=REWARD_STD_DEV,
        NETWORK_TOPOLOGY=NETWORK_TOPOLOGY, SOCIAL_INFLUENCE_WEIGHT=SOCIAL_INFLUENCE_WEIGHT, PROPORTION_UNBIASED_AGENTS=PROPORTION_UNBIASED_AGENTS,
        USE_TRUST_WEIGHTING=False,
        SEED_OFFSET=SEED + NUM_TRIALS if SEED is not None else NUM_TRIALS
    )
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    
    plt.figure(figsize=(12, 7))
    time_steps = np.arange(TIMESTEPS)
    
    plt.plot(time_steps, mean_regret_weighted, label='Weighted (Trust-Based)', color='darkorange', linewidth=2)
    plt.fill_between(time_steps, mean_regret_weighted - std_regret_weighted, mean_regret_weighted + std_regret_weighted, color='darkorange', alpha=0.2)
    
    plt.plot(time_steps, mean_regret_unweighted, label='Unweighted (Naive)', color='green', linewidth=2)
    plt.fill_between(time_steps, mean_regret_unweighted - std_regret_unweighted, mean_regret_unweighted + std_regret_unweighted, color='green', alpha=0.2)
    
    plt.title(f'Cumulative Regret: Weighted vs. Unweighted Social Learning\n'
              f'(4 Agents, $p_{{unbiased}} = {PROPORTION_UNBIASED_AGENTS}$, $\lambda = {SOCIAL_INFLUENCE_WEIGHT}$)', fontsize=16)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- SIMULATION PARAMETERS ---
N_AGENTS = 4
K_ARMS = 10      
TIMESTEPS = 2000 
NUM_TRIALS = 300 # Increased number of trials for a smoother plot
INITIAL_TRUST_SCORE = 0.5
LEARNING_RATE_TRUST = 0.2
FIXED_BIASED_VALUE = -15.0
REWARD_STD_DEV = 0.001          
NETWORK_TOPOLOGY = "fully_connected" 
SOCIAL_INFLUENCE_WEIGHT = 0.7 
PROPORTION_UNBIASED_AGENTS = 0 # Adjusted for a more challenging scenario
SEED = 42

# --- Run the comparison simulation ---
if __name__ == "__main__":
    run_and_compare_weighted_vs_unweighted(
        N_AGENTS, K_ARMS, TIMESTEPS, NUM_TRIALS,
        INITIAL_TRUST_SCORE, FIXED_BIASED_VALUE, LEARNING_RATE_TRUST, REWARD_STD_DEV,
        NETWORK_TOPOLOGY, SOCIAL_INFLUENCE_WEIGHT, PROPORTION_UNBIASED_AGENTS, SEED
    )
