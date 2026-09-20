@abstract class_name ObservableNode extends Node
var parent_node

func _ready():
	parent_node = get_parent()
	add_to_group("rl_observables")

@abstract func get_observables() # defines the set of observables gleened from the parent class
