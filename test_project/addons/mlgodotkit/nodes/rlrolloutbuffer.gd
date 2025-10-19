extends Node
class_name RLRolloutBuffer

## Simple rollout memory for PPO-like algorithms.

var states: Array = []
var actions: Array = []
var rewards: Array = []
var dones: Array = []
var log_probs: Array = []
var values: Array = []

func store(state: Array, action: Array, reward: float, done: bool, log_prob: float = 0.0, value: float = 0.0) -> void:
	states.append(state)
	actions.append(action)
	rewards.append(reward)
	dones.append(done)
	log_probs.append(log_prob)
	values.append(value)

func clear() -> void:
	states.clear()
	actions.clear()
	rewards.clear()
	dones.clear()
	log_probs.clear()
	values.clear()

func size() -> int:
	return states.size()
