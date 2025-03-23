extends Node

@onready var tree = DTreeNode.new()

func _ready():
    tree.set_max_depth(3)
    tree.set_min_samples_split(2)

    var inputs = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    var targets = [0, 1]

    tree.fit(inputs, targets)

    var prediction = tree.predict([[1.5, 2.5]])
    assert(prediction.size() == 1, "Prediction result should be a single value")
    assert(prediction[0] == 0, "Expected prediction to be 0")

    print("DTreeNode test passed!")
    get_tree().quit()  # Exit game when done testing
