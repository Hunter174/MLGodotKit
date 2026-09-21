extends Control

func _ready() -> void:
	var controller := PIDControllerNode.new()
	add_child(controller)
	controller.set_kp(2.0)
	controller.set_ki(0.2)
	controller.set_kd(0.1)
	controller.set_limits(-10.0, 10.0)

	var setpoint := 1.0
	var measurement := 0.0
	var outputs := []
	for _step in range(10):
		var control = controller.update_dt(setpoint, measurement, 0.1)
		measurement += control * 0.1
		outputs.append("%.2f" % measurement)

	_show_result("PID control", [
		"setpoint: %.2f" % setpoint,
		"measurements over time: %s" % [outputs],
		"Tune kp, ki, and kd to change the response.",
	])

func _show_result(title: String, lines: Array) -> void:
	var label := Label.new()
	label.position = Vector2(32, 32)
	label.add_theme_font_size_override("font_size", 20)
	label.text = title + "\n\n" + "\n".join(lines)
	add_child(label)
