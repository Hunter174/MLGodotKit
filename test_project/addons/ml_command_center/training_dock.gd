@tool
extends VBoxContainer

# The Training Dock provides a real-time control interface for ML agents.
# It bridges the gap between the C++ backend and the Godot Editor.

var target_runner = null
var update_timer: Timer

func _init():
	# Setup UI Layout
	set_custom_minimum_size(Vector2(300, 400))
	
	# --- Header ---
	var header = Label.new()
	header.text = "🧠 ML Training Command Center"
	header.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	add_theme_font_override("font_size", 16)
	add_child(header)
	
	add_separator()
	
	# --- Target Selection ---
	var target_label = Label.new()
	target_label.text = "Active RL Runner:"
	add_child(target_label)
	
	var runner_select = OptionButton.new()
	runner_select.custom_minimum_size.y = 30
	runner_select.item_added.connect(_on_runner_selected)
	add_child(runner_select)
	
	# Populate runners from the scene
	refresh_runners(runner_select)
	
	add_separator()
	
	# --- Hyperparameter Controls ---
	var params_container = VBoxContainer.new()
	params_container.name = "ParamsContainer"
	add_child(params_container)
	
	# Example: Learning Rate Slider
	add_hyperparameter_control(params_container, "Learning Rate", 0.001, 0.1, 0.01)
	add_hyperparameter_control(params_container, "Discount Factor", 0.8, 0.999, 0.95)
	add_hyperparameter_control(params_container, "Epsilon (Exploration)", 0.01, 1.0, 0.1)
	
	add_separator()
	
	# --- Training Controls ---
	var ctrl_container = HStackBox.new()
	
	var start_btn = Button.new()
	start_btn.text = "▶ Start"
	start_btn.pressed.connect(_on_start_pressed)
	ctrl_container.add_child(start_btn)
	
	var stop_btn = Button.new()
	stop_btn.text = "⏸ Stop"
	stop_btn.pressed.connect(_on_stop_pressed)
	ctrl_container.add_child(stop_btn)
	
	var reset_btn = Button.new()
	reset_btn.text = "🔄 Reset"
	reset_btn.pressed.connect(_on_reset_pressed)
	ctrl_container.add_child(reset_btn)
	
	add_child(ctrl_container)
	
	# --- Real-time Stats ---
	var stats_label = Label.new()
	stats_label.name = "StatsLabel"
	stats_label.text = "Status: Idle\nLast Reward: 0.0\nStep: 0"
	add_child(stats_label)
	
	# Setup update loop
	update_timer = Timer.new()
	update_timer.wait_time = 0.1
	update_timer.autostart = true
	update_timer.timeout.connect(_update_ui)
	add_child(update_timer)

func add_separator():
	var sep = HSeparator.new()
	add_child(sep)

func add_hyperparameter_control(parent, label_text, min_val, max_val, default):
	var label = Label.new()
	label.text = label_text
	parent.add_child(label)
	
	var slider = HSlider.new()
	slider.min_value = min_val
	slider.max_value = max_val
	slider.step = (max_val - min_val) / 100.0
	slider.value = default
	slider.value_changed.connect(_on_param_changed)
	parent.add_child(slider)
	
	var val_label = Label.new()
	val_label.text = str(default)
	val_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	slider.value_changed.connect(func(v): val_label.text = "%.4f" % v)
	parent.add_child(val_label)

func refresh_runners(select_node):
	select_node.clear()
	# In a real scenario, we search the current scene tree for RLRunner nodes
	# Since we are in the editor, we might search the edited scene
	var scene_root = get_tree().root
	if scene_root:
		# This is a simplified search; in the actual plugin we would 
		# integrate with EditorInterface to get the currently edited scene
		var runners = []
		# Hypothetical search for nodes that have the RLRunner script
		# For now, we add some dummy entries to demonstrate the UI
		select_node.add_item("No Runner Found")
		select_node.add_item("ExampleRunner_1")
		select_node.add_item("ExampleRunner_2")

func _on_runner_selected(index):
	# Logic to link the UI to the actual object in the scene
	print("Selected runner index: ", index)

func _on_param_changed(value):
	if target_runner:
		# Push update to the C++ / GDScript runner
		# target_runner.update_param(value)
		pass

func _on_start_pressed():
	print("Training Started")

func _on_stop_pressed():
	print("Training Stopped")

func _on_reset_pressed():
	print("Training Reset")

func _update_ui():
	# Update stats label from target_runner
	var stats = get_node_or_null("StatsLabel")
	if stats and target_runner:
		stats.text = "Status: Running\nLast Reward: %.2f\nStep: %d" % [target_runner.last_reward, target_runner.step]
