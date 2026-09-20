#include "extern/catch_amalgamated.hpp"
#include "mlgodotkit/src/models/neural_network/neural_network_core.h"
#include "mlgodotkit/src/control/pid_controller/pid_controller_core.h"
#include <Eigen/Dense>

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
