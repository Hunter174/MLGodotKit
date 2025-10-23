@tool
extends EditorInspectorPlugin

const ACTIVATIONS := ["relu", "sigmoid", "linear", "leaky_relu"]

func _can_handle(object: Object) -> bool:
	return object.get_class() == "NNNode"

func _parse_property(object, type, name, hint_type, hint_string, usage_flags, wide):
	if name == "layers":
		var layers: Array = object.get("layers")
		var vbox = VBoxContainer.new()

		# Add new layer
		var add_btn = Button.new()
		add_btn.text = "+ Add Layer"
		add_btn.pressed.connect(func():
			layers.append({"input_size": 1, "output_size": 1, "activation": "relu"})
			_sync_layers(object, name, layers)
		)
		vbox.add_child(add_btn)

		# Render layers
		for i in range(layers.size()):
			var row = HBoxContainer.new()

			# Input size – only editable for the first layer
			var in_spin = SpinBox.new()
			in_spin.min_value = 1
			in_spin.value = layers[i].get("input_size", 1)
			in_spin.editable = (i == 0)
			if i == 0:
				in_spin.value_changed.connect(func(val):
					layers[0]["input_size"] = int(val)
					_sync_layers(object, name, layers)
				)
			row.add_child(in_spin)

			# Output size – always editable
			var out_spin = SpinBox.new()
			out_spin.min_value = 1
			out_spin.value = layers[i].get("output_size", 1)
			out_spin.value_changed.connect(func(val):
				layers[i]["output_size"] = int(val)

				# Cascade input sizes forward
				for j in range(i + 1, layers.size()):
					layers[j]["input_size"] = layers[j - 1]["output_size"]

				_sync_layers(object, name, layers)
			)
			row.add_child(out_spin)

			# Activation function
			var act_opt = OptionButton.new()
			for act in ACTIVATIONS:
				act_opt.add_item(act)
			var current_act = str(layers[i].get("activation", "relu"))
			for j in range(act_opt.item_count):
				if act_opt.get_item_text(j) == current_act:
					act_opt.select(j)
					break
			act_opt.item_selected.connect(func(idx):
				layers[i]["activation"] = act_opt.get_item_text(idx)
				_sync_layers(object, name, layers)
			)
			row.add_child(act_opt)

			# Remove layer button
			var rm_btn = Button.new()
			rm_btn.text = "x"
			rm_btn.pressed.connect(func():
				layers.remove_at(i)
				# Cascade fix after removal
				for j in range(1, layers.size()):
					layers[j]["input_size"] = layers[j - 1]["output_size"]
				_sync_layers(object, name, layers)
			)
			row.add_child(rm_btn)

			vbox.add_child(row)

		add_property_editor(name, vbox)
		return true

	return false

func _sync_layers(object: Object, name: String, layers: Array) -> void:
	object.set(name, layers)
	object.notify_property_list_changed()
