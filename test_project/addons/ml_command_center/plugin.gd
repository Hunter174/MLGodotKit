@tool
extends EditorPlugin

const TrainingDock = preload("res://addons/ml_command_center/training_dock.gd")

var dock_instance

func _enter_tree():
	# Create the dock instance
	dock_instance = TrainingDock.new()
	# Add the dock to the editor's main screen area
	add_control_to_container(EditorPlugin.CONTAINER_CANVAS_EDITOR_SIDE_LEFT, dock_instance)

	print("[ML Command Center] Plugin initialized and dock added.")

func _exit_tree():
	# Clean up the dock when the plugin is disabled
	remove_control_from_container(EditorPlugin.CONTAINER_CANVAS_EDITOR_SIDE_LEFT, dock_instance)
	dock_instance.queue_free()
	print("[ML Command Center] Plugin disabled.")
