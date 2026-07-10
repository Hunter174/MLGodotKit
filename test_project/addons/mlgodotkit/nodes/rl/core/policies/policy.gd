class_name Policy extends RefCounted

var action_space
var value_function = null

func _init(_action_space, _value_function):
	action_space = _action_space
	value_function = _value_function

func select_action(state):
	push_error("select_action must be implemented by subclass")
	return null
