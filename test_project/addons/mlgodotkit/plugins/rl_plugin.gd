@tool
extends EditorInspectorPlugin

func _can_handle(object: Object) -> bool:
	return object.get_script() and object.get_script().get_global_name() == "RLNode"

func _parse_begin(object: Object) -> void:
	var title := Label.new()
	title.text = "Plugin customization to be added..."
	title.add_theme_color_override("font_color", Color(0.7, 0.85, 1))
	title.add_theme_font_size_override("font_size", 32)
	add_custom_control(title)
	add_custom_control(HSeparator.new())
