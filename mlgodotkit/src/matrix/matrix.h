#ifndef MATRIX_H
#define MATRIX_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector4.hpp>
#include <Eigen/Dense>

namespace godot {

    class Matrix : public RefCounted {
        GDCLASS(Matrix, RefCounted);

    private:
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m;

    protected:
        static void _bind_methods();

    public:
        void _init();
        Matrix();

        static Ref<Matrix> zeros(int rows, int cols);
        static Ref<Matrix> ones(int rows, int cols);
        static Ref<Matrix> identity(int n);
        static Ref<Matrix> from_array(const Array &A);

        static Ref<Matrix> from_vector2(const Vector2 &v, bool column = true);
        static Ref<Matrix> from_vector3(const Vector3 &v, bool column = true);
        static Ref<Matrix> from_vector4(const Vector4 &v, bool column = true);

        Array to_array() const;

        Vector2 to_vector2() const;
        Vector3 to_vector3() const;
        Vector4 to_vector4() const;

        Vector2 mul_vector2(const Vector2 &v) const;
        Vector3 mul_vector3(const Vector3 &v) const;
        Vector4 mul_vector4(const Vector4 &v) const;

        int rows() const;
        int cols() const;

        Ref<Matrix> transpose() const;
        Ref<Matrix> matmul(const Ref<Matrix> &B) const;
        Ref<Matrix> inverse() const;

        float det() const;
        float trace() const;
        float norm() const;

        float get(int i, int j) const;
        void set(int i, int j, float value);

        Ref<Matrix> copy() const;
        bool equals(const Ref<Matrix> &other, float eps = 1e-6f) const;
        Dictionary info() const;

        String _to_string() const;

        using EigenMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const EigenMat &eigen() const { return m; }
        EigenMat &eigen() { return m; }
    };

} // namespace godot

#endif