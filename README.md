# MLGodotKit

**Empower your Godot projects with the power of machine learning!**  
MLGodotKit is a C++ GDExtension for Godot, enabling seamless integration of AI-driven features into your games and applications. With support for adaptive behaviors and real-time decision-making, it’s designed to inspire innovation and enhance gameplay.

<p align="center">
  <img src="MLGodotKit_logo.png" alt="MLGodotKit Logo" width="500"/>
</p>

---

## Node Extensions in Progress
- **Data Science Tooling**: Tools for preprocessing, analyzing, and visualizing data directly in Godot.
- **Deep Learning Integration**: Simplified nodes for building and deploying neural networks.
- **Cross-Platform Compatibility**: Fully supported on Windows, Linux, and macOS.

---

## Getting Started

Here are quick examples of how to use the three core models in **MLGodotKit**:

> All models are GDExtension nodes and can be used directly in any Godot scene or script.

---

### Linear Regression (`LRNode`)

```gdscript
@onready var lr_model = LRNode.new()

func _ready():
	lr_model.set_learning_rate(0.01)
	lr_model.initialize(1)

	# Simple linear dataset: y = 3x + 5
	var inputs = [[1], [2], [3]]
	var targets = [[8], [11], [14]]

	lr_model.train(inputs, targets, 1000)

	var prediction = lr_model.predict([4])
	print("Predicted y for x=4:", prediction)
```

---

## Descision Tree Classifier (`DTreeNode`)

```gdscript
@onready var tree = DTreeNode.new()

func _ready():
	tree.set_max_depth(5)
	tree.fit([[1], [2], [3], [10], [11], [12]], [0, 0, 0, 1, 1, 1])

	var result = tree.predict([[2], [11]])
	print("Predictions:", result)  # [0, 1]
```

## Neural Network (`NNNode`)

```gdscript
@onready var nn = NNNode.new()

func _ready():
	nn.set_learning_rate(0.1)
	nn.add_layer(2, 4, "relu")
	nn.add_layer(4, 1, "sigmoid")

	var inputs = [[0,0], [0,1], [1,0], [1,1]]
	var targets = [[0], [1], [1], [0]]  # XOR logic

	for i in range(5000):
		for j in range(inputs.size()):
			var y_pred = nn.forward([inputs[j]])[0]
			var error = y_pred - targets[j][0]
			nn.backward([[2.0 * error]])

	print("Test XOR:")
	for i in range(inputs.size()):
		var output = nn.forward([inputs[i]])
		print("Input:", inputs[i], " Predicted:", output, "Expected:", targets[i])

```



## Current Status
MLGodotKit is a **work in progress**, with ongoing development focused on core functionality and cross-platform support.

---

## Credits
Built on the powerful [Eigen C++ library](https://eigen.tuxfamily.org/).


