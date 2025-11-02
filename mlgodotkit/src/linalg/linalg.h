#ifndef MLGODOTKIT_LINALG_H
#define MLGODOTKIT_LINALG_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <Eigen/Dense>
#include "utility/logger.h"
#include "utility/utils.h"

namespace godot {

    class Linalg : public RefCounted {
        GDCLASS(Linalg, RefCounted);

    protected:
        static void _bind_methods();

    public:
        // Core operations
        static Array add(const Array &A, const Array &B);
        static Array mult(const Array &A, const Array &B);
        static Array transpose(const Array &A);
        static Array inv(const Array &A);
    };

} // namespace godot

#endif
