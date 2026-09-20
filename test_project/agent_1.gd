extends AgentNode

const ACTIONS = [
	Vector2.UP,
	Vector2.DOWN,
	Vector2.LEFT,
	Vector2.RIGHT
]

var speed := 100.0
var debug := false

# -------------------------------------------------
# Initialization
# -------------------------------------------------

func _ready():
	super()

	var nn := NeuralNetworkNode.new()
	nn.set_optimizer("adam")
	nn.set_learning_rate(0.001)

	nn.add_layer(2, 64, "relu")
	nn.add_layer(64, 64, "relu")
	nn.add_layer(64, 4, "linear")
	nn.set_batch_size(64)

	var q = NeuralQFunction.new(nn)

	var target_nn = nn  # must implement clone
	var target_q = NeuralQFunction.new(target_nn)

	policy = EpsilonGreedyPolicy.new([0,1,2,3], q)
	learner = DQNLearner.new(q, target_q)

# -------------------------------------------------
# Observables (what the environment sees)
# -------------------------------------------------

func get_observables():
	return {
		"blue_agent_pos": parent_node.position
	}

# -------------------------------------------------
# Decision Phase
# -------------------------------------------------

func decide(state_t):
	last_state = process_state(state_t)
	action = policy.select_action(last_state)

# -------------------------------------------------
# State Representation (DISCRETE)
# -------------------------------------------------

func process_state(state):
	var target = state["target_pos"]
	var relative = target - parent_node.position
	return [
		relative.x / 1000.0,
		relative.y / 1000.0
	]

# -------------------------------------------------
# Action Phase
# -------------------------------------------------

func act():
	var direction = ACTIONS[action]
	parent_node.position += direction * speed * get_process_delta_time()

# -------------------------------------------------
# Reward Function
# -------------------------------------------------

func get_reward(state_t, state_t1):
	var target_t = state_t["target_pos"]
	var target_t1 = state_t1["target_pos"]

	var prev_pos = state_t["blue_agent_pos"]
	var curr_pos = parent_node.position

	var prev_dist = prev_pos.distance_to(target_t)
	var curr_dist = curr_pos.distance_to(target_t1)

	# Positive if moving closer
	return prev_dist - curr_dist

# -------------------------------------------------
# Done Condition
# -------------------------------------------------

func get_done(state_t1):
	var target = state_t1["target_pos"]
	var dist = parent_node.position.distance_to(target)

	return dist < 10.0
