extends Control

func _ready() -> void:
	var model := LinearRegressionNode.new()
	add_child(model)

	var inputs := [[0.0], [1.0], [2.0], [3.0], [4.0]]
	var targets := [[1.0], [3.0], [5.0], [7.0], [9.0]]
	model.fit(inputs, targets)
	var predictions = model.predict([[5.0], [6.0]])

	_show_result("Linear regression", [
		"learned relationship: y ≈ 2x + 1",
		"prediction for x=5: %s" % [predictions[0]],
		"prediction for x=6: %s" % [predictions[1]],
	])

func _show_result(title: String, lines: Array) -> void:
	var label := Label.new()
	label.position = Vector2(32, 32)
	label.add_theme_font_size_override("font_size", 20)
	label.text = title + "\n\n" + "\n".join(lines)
	add_child(label)
