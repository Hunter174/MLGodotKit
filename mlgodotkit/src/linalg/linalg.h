#ifndef LINALG_H
#define LINALG_H

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include "matrix/matrix.h"

namespace godot {

class Linalg : public Object {
    GDCLASS(Linalg, Object);

protected:
    static void _bind_methods();

public:
    static Ref<Matrix> solve(const Ref<Matrix> &A, const Ref<Matrix> &b);
    static Ref<Matrix> least_squares(const Ref<Matrix> &A, const Ref<Matrix> &b);
    static Ref<Matrix> pinv(const Ref<Matrix> &A);

    static Dictionary qr(const Ref<Matrix> &A);
    static Dictionary svd(const Ref<Matrix> &A);
    static Dictionary eig(const Ref<Matrix> &A);
    static Dictionary lu(const Ref<Matrix> &A);
};

} // namespace godot

#endif