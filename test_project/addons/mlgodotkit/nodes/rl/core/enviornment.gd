class_name RLEnvironment extends Node

var agents := []
var observables := []
var step_count := 0
var debug := false
var log_interval := 30

func _ready():
	agents = get_tree().get_nodes_in_group("rl_agents")
	observables = get_tree().get_nodes_in_group("rl_observables")

	if debug:
		print("Environment ready.")
		print("Agents found: ", agents.size())
		print("Observables found: ", observables.size())

func _physics_process(delta):
	await step()

func step():
	step_count += 1

	var state_t = get_snapshot()

	if debug and step_count % log_interval == 0:
		print("\n==============================")
		print("STEP: ", step_count)
		print("Agents: ", agents.size())
		print("State_t: ", state_t)

	# --- Decision Phase ---
	for agent in agents:
		if debug and step_count % log_interval == 0:
			print("Decision -> ", agent.name)
		agent.decide(state_t)

	# --- Action Phase ---
	for agent in agents:
		if debug and step_count % log_interval == 0:
			print("Action -> ", agent.name)
		agent.act()

	# --- Resolve World ---
	await get_tree().physics_frame

	var state_t1 = get_snapshot()

	if debug and step_count % log_interval == 0:
		print("State_t1: ", state_t1)

	# --- Observation + Learning Phase ---
	for agent in agents:
		if debug and step_count % log_interval == 0:
			print("Observe -> ", agent.name)
		agent.observe(state_t, state_t1)

	if debug and step_count % log_interval == 0:
		print("==============================\n")

func get_snapshot():
	var snapshot := {}
	for obs in observables:
		var obs_data = obs.get_observables()
		for key in obs_data:
			snapshot[key] = obs_data[key]
	return snapshot

func reset():
	if debug:
		print("Environment reset.")
	step_count = 0
	for agent in agents:
		agent.reset()
		
	for obs in observables:
		obs.reset()
