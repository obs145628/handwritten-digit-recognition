#include "matrix.hh"
#include <algorithm>
#include <cassert>
#include "serial.hxx"

namespace ml
{

    void Matrix::fill(double val)
    {
        std::fill(begin(), end(), val);
    }

    void Matrix::replace(Matrix &temp)
    {
        delete data_;
        rows_ = temp.rows_;
        cols_ = temp.cols_;
        data_ = temp.data_;
        temp.data_ = nullptr;
    }

    Vector Matrix::solve_lower(const Vector &b)
    {
        assert(rows_ == cols_);
        assert(rows_ == b.size());
        std::size_t n = rows_;
        Vector x(n);

        for (std::size_t k = 0; k < n; ++k)
        {
            double val = b[k];
            for (std::size_t i = 0; i < k; ++i)
                val -= at(k, i) * x[i];
            x[k] = val / at(k, k);
        }


        return x;
    }

    Vector Matrix::solve_upper(const Vector &b)
    {
        assert(rows_ == cols_);
        assert(rows_ == b.size());
        std::size_t n = rows_;
        Vector x(n);

        for (std::size_t k = n - 1; k < n; --k)
        {
            double val = b[k];
            for (std::size_t i = k + 1; i < n; ++i)
                val -= at(k, i) * x[i];
            x[k] = val / at(k, k);
        }


        return x;
    }

    Matrix Matrix::transpose() const
    {
        Matrix res(cols(), rows());
        for (std::size_t i = 0; i < cols(); ++i)
            for (std::size_t j = 0; j < rows(); ++j)
                res(i, j) = at(j, i);
        return res;
    }

    Matrix Matrix::inverse() const
    {
        return plu_inverse();
    }

    /** PLU Decomposition
     * http://www.math.unm.edu/~loring/links/linear_s08/LU.pdf
     * http://kaba.hilvi.org/pastel-1.1.0/pastel/math/ludecomposition.htm
     * https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/lup/
     */

    bool Matrix::plu_decomposition(Matrix &p, Matrix& l, Matrix& u,
                                   std::vector<std::size_t> &pv,
                                   bool& even_permuts) const
    {
        assert(rows_ == cols_);
        Matrix res = *this;
        std::size_t n = rows_;
        std::size_t nb_permuts = 0;
        std::vector<double> inv_largest(n);
        std::vector<std::size_t> permuts(n);
        for (std::size_t i = 0; i < n; ++i)
            permuts[i] = i;

        //Find max element in each row
        //If max is 0 => rows of 0 => can't decompose
        for (std::size_t i = 0; i < n; ++i)
        {
            double max = 0;
            for (std::size_t j = 0; j < n; ++j)
            {
                if (std::abs(res(i, j)) > max)
                    max = std::abs(res(i, j));
            }

            if (max == 0)
                return false;
            inv_largest[i] = 1.0 / max;
        }

        // decompose collumn b collumn
        for (std::size_t k = 0; k < n; ++k)
        {


            //find max element among of the collumn at or below current collumn
            //if max is 0 => no pivot => can't decompose
            double max_val = 0;
            std::size_t max_i = k;

            for (std::size_t i = k; i < n; ++i)
            {
                double val = inv_largest[i] * std::abs(res(i, k));
                if (val > max_val)
                {
                    max_val = val;
                    max_i = i;
                }
            }

            if (max_val == 0)
            {
                return false;
            }



            //Swqp current row with the pivot
            if (max_i != k)
            {
                for (std::size_t j = 0; j < n; ++j)
                    std::swap(res(max_i, j), res(k, j));

                std::swap(inv_largest[max_i], inv_largest[k]);
                std::swap(permuts[max_i], permuts[k]);
                ++nb_permuts;
             }


            for (std::size_t i = k + 1; i < n; ++i)
            {
                //Divide all elements in column below the diagonal by the pivot value
                res(i, k) /= res(k, k);

                //Apply Gaussian elimination
                for (std::size_t j = k + 1; j < n; ++j)
                    res(i, j) -= res(i, k) * res(k, j);
            }
        }

        Matrix pout = Matrix::with(n, n, 0);
        for (std::size_t i = 0; i < n; ++i)
            pout(i, permuts[i]) = 1;

        Matrix lout = Matrix::id(n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < i; ++j)
                lout(i, j) = res(i, j);

        Matrix uout = Matrix::with(n, n, 0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j)
                uout(i, j) = res(i, j);

        p.replace(pout);
        l.replace(lout);
        u.replace(uout);
        pv = permuts;
        even_permuts = nb_permuts % 2 == 0;
        return true;
    }

    Vector Matrix::plu_solve(const Vector &b) const
    {
        assert(rows_ == cols_);
        assert(rows_ == b.size());
        std::size_t n = rows_;
        Matrix p;
        Matrix l;
        Matrix u;
        std::vector<std::size_t> pv;
        bool permuts;
        bool ok = plu_decomposition(p, l, u, pv, permuts);
        if (!ok)
            return Vector{};

        Vector b2(n);
        for (std::size_t i = 0; i < n; ++i)
            b2[pv[i]] = b[i];

        Vector y = l.solve_lower(b2);
        Vector x = u.solve_upper(y);
        return x;
    }

    Matrix Matrix::plu_solve(const Matrix &b) const
    {
        assert(rows_ == cols_);
        assert(b.rows_ == b.cols_);
        assert(rows_ == b.rows_);
        std::size_t n = rows_;
        Matrix p;
        Matrix l;
        Matrix u;
        std::vector<std::size_t> pv;
        bool permuts;
        bool ok = plu_decomposition(p, l, u, pv, permuts);
        if (!ok)
            return Matrix{};
        Matrix res(n, n);

        for (std::size_t k = 0; k < n; ++k)
        {
            Vector b2(n);
            for (std::size_t i = 0; i < n; ++i)
                b2[pv[i]] = b(i, k);

            Vector y = l.solve_lower(b2);
            Vector x = u.solve_upper(y);
            for (std::size_t i = 0; i < n; ++i)
                res(i, k) = x[i];
        }

        return res;
    }

    Matrix Matrix::plu_inverse() const
    {
        assert(rows_ == cols_);
        return plu_solve(id(rows_));
    }


    Matrix& Matrix::operator+=(const Matrix& m)
    {
        assert(rows() && m.rows());
        assert(cols() == m.cols());
        const double* in = m.begin();
        for (double* out = begin(); out != end(); ++out)
            *out += *in++;
        return *this;
    }

    Matrix& Matrix::operator-=(const Matrix& m)
    {
        assert(rows() && m.rows());
        assert(cols() == m.cols());
        const double* in = m.begin();
        for (double* out = begin(); out != end(); ++out)
            *out -= *in++;
        return *this;
    }

    Matrix& Matrix::operator+=(double x)
    {
        for (double* out = begin(); out != end(); ++out)
            *out += x;
        return *this;
    }

    Matrix& Matrix::operator-=(double x)
    {
        for (double* out = begin(); out != end(); ++out)
            *out -= x;
        return *this;
    }

    Matrix& Matrix::operator*=(double x)
    {
        for (double* out = begin(); out != end(); ++out)
            *out *= x;
        return *this;
    }

    Matrix& Matrix::operator/=(double x)
    {
        for (double* out = begin(); out != end(); ++out)
            *out /= x;
        return *this;
    }

    Matrix Matrix::id(std::size_t n)
    {
        auto res = Matrix::with(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            res(i, i) = 1;
        return res;
    }

    Matrix Matrix::mul(const Matrix &a, const Matrix &b)
    {
        assert(a.cols_ == b.rows_);
        Matrix res(a.rows_, b.cols_);

        for (std::size_t i = 0; i < res.rows_; ++i)
            for (std::size_t j = 0; j < res.cols_; ++j)
            {
                double val = 0;
                for (std::size_t k = 0; k < a.cols_; ++k)
                    val += a(i, k) * b(k, j);
                res(i, j) = val;
            }

        return res;
    }

    Vector Matrix::mul(const Matrix& a, const Vector& b)
    {
        assert(a.cols_ == b.size());
        Vector res(a.rows_);

        for (std::size_t i = 0; i < res.size(); ++i)
        {
            double val = 0;
            for (std::size_t k = 0; k < a.cols_; ++k)
                val += a(i, k) * b[k];
            res[i] = val;
        }

        return res;
    }

   Vector Matrix::mul(const Vector& a, const Matrix& b)
   {
      assert(a.size() == b.rows());
      Vector res(b.cols());

      for (std::size_t j = 0; j < res.size(); ++j)
        {
            double val = 0;
            for (std::size_t k = 0; k < a.size(); ++k)
               val += a[k] * b(k, j);
            res[j] = val;
        }

        return res;
   }

    void Matrix::mul(const Matrix& a, const Vector& b, Vector& out)
    {
        out.assign(a.rows());

        for (std::size_t i = 0; i < a.rows(); ++i)
        {
            double val = 0;
            for (std::size_t k = 0; k < a.cols_; ++k)
                val += a(i, k) * b[k];
            out[i] = val;
        }
    }

    std::ostream &operator<<(std::ostream &os, const Matrix &m)
    {
        for (std::size_t i = 0; i < m.rows(); ++i)
        {
            os << "|";
            for (std::size_t j = 0; j < m.cols(); ++j)
            {
               os << " " << m(i, j) << " |";
            }
            os << std::endl;
        }

        return os;
    }

    namespace serial
    {
        void write(std::ostream& os, const Matrix& m)
        {
            write_bin(os, m.rows());
            write_bin(os, m.cols());
            write_bin(os, m.begin(), m.end());
        }

        void read(std::istream& is, Matrix& m)
        {
            std::size_t rows;
            std::size_t cols;
            read_bin(is, rows);
            read_bin(is, cols);
            m = Matrix(rows, cols);
            read_bin(is, m.begin(), m.end());
      }
   }


    Matrix operator+(const Matrix& a, const Matrix& b)
    {
        assert(a.rows() && b.rows());
        assert(a.cols() == b.cols());
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        const double* i2 = b.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = *i1++ + *i2++;
        return res;
    }

    Matrix operator-(const Matrix& a, const Matrix& b)
    {
        assert(a.rows() && b.rows());
        assert(a.cols() == b.cols());
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        const double* i2 = b.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = *i1++ - *i2++;
        return res;
    }

    Matrix operator+(const Matrix& a, double b)
    {
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = *i1++ + b;
        return res;
    }

    Matrix operator-(const Matrix& a, double b)
    {
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = *i1++ - b;
        return res;
    }

    Matrix operator*(const Matrix& a, double b)
    {
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = *i1++ * b;
        return res;
    }

    Matrix operator/(const Matrix& a, double b)
    {
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = *i1++ / b;
        return res;
    }

    Matrix operator+(double a, const Matrix& b)
    {
        Matrix res(b.rows(), b.cols());
        const double* i2 = b.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = a + *i2++;
        return res;
    }

    Matrix operator-(double a, const Matrix& b)
    {
        Matrix res(b.rows(), b.cols());
        const double* i2 = b.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = a - *i2++;
        return res;
    }

    Matrix operator*(double a, const Matrix& b)
    {
        Matrix res(b.rows(), b.cols());
        const double* i2 = b.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = a * *i2++;
        return res;
    }

    Matrix operator/(double a, const Matrix& b)
    {
        Matrix res(b.rows(), b.cols());
        const double* i2 = b.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = a / *i2++;
        return res;
    }

    Matrix operator-(const Matrix& a)
    {
        Matrix res(a.rows(), a.cols());
        const double* i1 = a.begin();
        for (double* out = res.begin(); out != res.end(); ++out)
            *out = - *i1++;
        return res;
    }


    Matrix hadamard_product(const Matrix& a, const Matrix& b)
    {
        assert(a.rows() == b.rows());
        assert(a.cols() == b.cols());
        Matrix res(a.rows(), a.cols());
        const double* ai = a.begin();
        const double* bi = b.begin();
        for (double* it = res.begin(); it != res.end(); ++it)
            *it = *ai++ * *bi++;
        return res;
    }

    Matrix outer_product(const Vector& a, const Vector& b)
    {
        Matrix res(a.size(), b.size());
        for (std::size_t i = 0; i < res.rows(); ++i)
            for (std::size_t j = 0; j < res.cols(); ++j)
                res(i, j) = a[i] * b[j];
        return res;
    }

}
