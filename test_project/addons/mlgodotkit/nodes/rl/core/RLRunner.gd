extends Node
class_name RLRunner

signal episode_finished(total_reward)

@export var step_delay := 0.0
@export var render_mode := false

var env: RLEnvironment
var policy: RLPolicy
var trainer

var global_step := 0

func configure(p_env: RLEnvironment, p_policy: RLPolicy, p_trainer):
	env = p_env
	policy = p_policy
	trainer = p_trainer

func run_episode() -> float:
	var state = env.reset()
	var total_reward := 0.0

	for _i in range(env.max_steps):
		var action = policy.act(state)
		var result = env.step(action)

		trainer.observe(
			state,
			action,
			result.reward,
			result.state,
			result.done
		)

		if trainer.should_train(global_step):
			trainer.train_step()

		state = result.state
		total_reward += result.reward
		global_step += 1

		if result.done:
			break

		if render_mode and step_delay > 0.0:
			await get_tree().create_timer(step_delay).timeout

	policy.on_episode_end()
	emit_signal("episode_finished", total_reward)
	return total_reward
