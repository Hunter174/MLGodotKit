class_name EpsilonGreedyPolicy
extends Policy

var epsilon := 0.1

func select_action(state):
	if randf() < epsilon:
		return action_space[randi() % action_space.size()]

	var best_action = null
	var best_value = -INF

	for action in action_space:
		var q = value_function.evaluate(state, action)
		if q > best_value:
			best_value = q
			best_action = action

	return best_action
