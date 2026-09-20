class_name RLRunner
extends Node

var env = null
var policy = null
var trainer = null

var render_mode := false
var step_delay := 0.0

func configure(p_env, p_policy, p_trainer):
	env = p_env
	policy = p_policy
	trainer = p_trainer

func run_episode() -> float:
	var total_reward = 0.0
	var state = env.reset()
	var done = false
	
	while not done:
		var action = policy.select_action(state)
		# In most RL setups, the policy returns a vector of q-values; we take the argmax
		var discrete_action = 0
		var max_val = -INF
		for i in range(action.size()):
			if action[i] > max_val:
				max_val = action[i]
				discrete_action = i
		
		var result = env.step(discrete_action)
		var next_state = result.next_state
		var reward = result.reward
		var is_done = result.done
		
		# Store in buffer and train if available
		if trainer and trainer.buffer:
			trainer.buffer.store_experience(state, discrete_action, reward, next_state, is_done)
			if trainer.buffer.is_ready(trainer.batch_size):
				var batch = trainer.buffer.sample(trainer.batch_size)
				trainer.learn(batch)
		
		state = next_state
		total_reward += reward
		done = is_done
		
		if step_delay > 0:
			await get_tree().create_timer(step_delay).timeout
			
	return total_reward
