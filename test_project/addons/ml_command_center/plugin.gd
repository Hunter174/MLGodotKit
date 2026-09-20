@tool
extends EditorPlugin

const TrainingDock = preload("res://addons/ml_command_center/training_dock.gd")

var dock_instance

func _enter_tree():
	# Create the dock instance
	dock_instance = TrainingDock.new()
	# Add the dock to the editor's bottom panel (or a custom slot)
	add_tool_area_dock(dock_instance)
	print("[ML Command Center] Plugin initialized and dock added.")

func _exit_tree():
	# Clean up the dock when the plugin is disabled
	remove_tool_area_dock(dock_instance)
	dock_instance.queue_free()
	print("[ML Command Center] Plugin disabled.")
