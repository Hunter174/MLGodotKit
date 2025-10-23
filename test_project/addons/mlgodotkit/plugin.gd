# plugin.gd
@tool
extends EditorPlugin

var nn_plugin
var rl_plugin

func _enter_tree():
	nn_plugin = preload("res://addons/mlgodotkit/plugins/nn_plugin.gd").new()
	rl_plugin = preload("res://addons/mlgodotkit/plugins/rl_plugin.gd").new()
	add_inspector_plugin(nn_plugin)
	add_inspector_plugin(rl_plugin)


func _exit_tree():
	remove_inspector_plugin(nn_plugin)
	remove_inspector_plugin(rl_plugin)
