extends AgentNode

const ACTIONS = [
	Vector2.UP,
	Vector2.DOWN,
	Vector2.LEFT,
	Vector2.RIGHT
]

var speed := 100.0
var epsilon := 0.1
var gamma := 0.99
var lr := 0.1

var q_table := {}

func get_observables():
	return {"red_agent_pos" : parent_node.position}

func decide(state_t):
	last_state = process_state(state_t)

	# epsilon-greedy
	if randf() < epsilon:
		action = randi() % ACTIONS.size()
	else:
		action = _best_action(last_state)

func act():
	var direction = ACTIONS[action]
	parent_node.position += direction * speed * get_process_delta_time()

func observe(state_t, state_t1):
	var s = last_state
	var s1 = process_state(state_t1)

	var reward = get_reward(state_t, state_t1)
	var done = get_done(state_t1)

	_update_q(s, action, reward, s1)

func get_reward(state_t, state_t1):
	var target = state_t1["target_pos"]
	var dist = parent_node.position.distance_to(target)
	return -dist

func get_done(state_t1):
	var target = state_t1["target_pos"]
	return parent_node.position.distance_to(target) < 10

func process_state(state):
	var target = state["target_pos"]
	var relative = target - parent_node.position

	# discretize space
	return Vector2(
		sign(relative.x),
		sign(relative.y)
	)

func _best_action(state):
	if not q_table.has(state):
		q_table[state] = [0,0,0,0]

	var values = q_table[state]
	var best = 0
	for i in range(values.size()):
		if values[i] > values[best]:
			best = i
	return best

func _update_q(s, a, r, s1):
	if not q_table.has(s):
		q_table[s] = [0,0,0,0]
	if not q_table.has(s1):
		q_table[s1] = [0,0,0,0]

	var max_q_next = q_table[s1].max()
	var old = q_table[s][a]

	q_table[s][a] = old + lr * (r + gamma * max_q_next - old)
