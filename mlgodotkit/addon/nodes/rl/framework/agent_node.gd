class_name AgentNode extends Node

var last_snapshot : Dictionary
var last_state

var action
var policy
var learner
var pending_decision
var parent_node
var memory_buffer

func _ready():
	parent_node = get_parent()
	add_to_group("rl_agents")
	add_to_group("rl_observables")
	memory_buffer = ReplayBuffer.new(10000)

# --- Core API ---
func decide(state_t):
	last_state = state_t
	action = policy.select_action(state_t)

func act():
	push_error("AgentNode.act() must be overridden")

# Optional override if agent exposes state as observable
func get_observables():
	return {}

# --- Transition Handling ---
func observe(state_t, state_t1):
	var reward = get_reward(state_t, state_t1)
	var done = get_done(state_t1)

	var next_processed = process_state(state_t1)

	memory_buffer.store_experience(
		last_state,
		action,
		reward,
		next_processed,
		done
	)

	if memory_buffer.is_ready(64):
		var batch = memory_buffer.sample(64)
		learner.learn(batch)

# --- Reward / Done Logic (override per agent) ---
func get_reward(_state_t, _state_t1):
	push_error("AgentNode.get_reward() must be overridden")
	return 0.0

func process_state(state):
	push_error("AgentNode.process_state() must be overridden")
	return state

func get_done(_state_t1):
	push_error("AgentNode.get_done() must be overridden")
	return false

func reset():
	last_state = {}
