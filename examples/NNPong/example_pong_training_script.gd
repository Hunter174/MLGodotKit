extends Node

@onready var nn = NNNode.new()
@onready var ball = $"../Ball"
@onready var paddle = $"../PaddleAI"

var paddle_speed = 200.0
var training = true

func _ready():
	add_child(nn)
	nn.set_learning_rate(0.05)
	nn.add_layer(2, 4, "relu")
	nn.add_layer(4, 1, "sigmoid")

func _physics_process(delta):
	var input = [
		ball.position.y / 600.0,
		paddle.position.y / 600.0
	]

	# Predict desired movement direction
	var prediction = nn.forward([input])[0]

	# Map sigmoid output to paddle movement
	if prediction < 0.4:
		paddle.position.y -= paddle_speed * delta
	elif prediction > 0.6:
		paddle.position.y += paddle_speed * delta

	paddle.position.y = clamp(paddle.position.y, -200, 200)

	# TRAINING: move toward ball Y-position
	if training:
		var desired_direction = ball.position.y - paddle.position.y
		var target = 0.5  # stay still
		if desired_direction > 10:
			target = 1.0  # move down
		elif desired_direction < -10:
			target = 0.0  # move up

		# Compute error and backprop
		var y_pred = prediction
		var error = y_pred - target
		nn.backward([[2.0 * error]])
