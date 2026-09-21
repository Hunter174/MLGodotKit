class_name ObservableNode extends Node
var parent_node

func _ready():
	parent_node = get_parent()
	add_to_group("rl_observables")

# Override to expose values observed by an RL agent.
func get_observables():
	push_error("ObservableNode.get_observables() must be overridden")
	return {}
