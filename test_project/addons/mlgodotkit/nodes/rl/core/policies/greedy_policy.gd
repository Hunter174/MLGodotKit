class_name GreedyPolicy
extends Policy

func select_action(state):
	var best_action = null
	var best_value = -INF

	for action in action_space:
		var q = value_function.evaluate(state, action)
		if q > best_value:
			best_value = q
			best_action = action

	return best_action
