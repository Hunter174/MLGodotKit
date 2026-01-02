extends Node

const EPISODES := 800
const WINDOW := 20
const SUCCESS_THRESHOLD := 100.0
const SEED := 1337

var runner: RLRunner
var policy: DQNPolicy
var trainer: DQNTrainer
var buffer: ReplayBuffer
var env: CartPoleEnvironment

func _ready():
	# Build the needed nodes	
	runner = RLRunner.new()
	policy = DQNPolicy.new()
	trainer = DQNTrainer.new()
	buffer = ReplayBuffer.new()
	env = CartPoleEnvironment.new()
	
	# Make sure we add them to the scene
	add_child(runner)
	add_child(policy)
	add_child(trainer)
	add_child(buffer)
	add_child(env)

	# We need to configure our hyperparams and models params for learning
	_setup_networks()
	_setup_hyperparams()

	runner.configure(env, policy, trainer)
	runner.render_mode = false
	runner.step_delay = 0.0

	
	var rewards := []
	for ep in range(EPISODES):
		var r := await runner.run_episode()
		rewards.append(r)

		if (ep + 1) % WINDOW == 0:
			var avg = _mean(rewards.slice(-WINDOW))
			print("Episode %d | Avg(%d)=%.2f | ε=%.3f" % [ep + 1, WINDOW, avg, policy.epsilon])

	var final_avg = _mean(rewards.slice(-WINDOW))
	print("Final rolling avg:", final_avg)

func _setup_networks():
	var q_online := _build_q_network()
	var q_target := _build_q_network()
	q_target.copy_weights(q_online)

	policy.configure(q_online, 2, 128)
	trainer.configure(q_online, q_target, buffer, 2)

func _setup_hyperparams():
	buffer.capacity = 20000

	policy.epsilon = 1.0
	policy.epsilon_decay = 0.995
	policy.epsilon_min = 0.05

	trainer.gamma = 0.99
	trainer.batch_size = 128
	trainer.warmup = 1000
	trainer.target_update_period = 2000
	trainer.polyak_tau = 0.005

func _build_q_network() -> NeuralNetworkNode:
	var nn := NeuralNetworkNode.new()
	nn.add_layer(4, 64, "leaky_relu")
	nn.add_layer(64, 64, "leaky_relu")
	nn.add_layer(64, 2, "linear")
	nn.set_learning_rate(0.02)
	nn.set_batch_size(128)
	return nn

func _mean(arr):
	var s := 0.0
	for v in arr:
		s += v
	return s / max(1, arr.size())
