#! /usr/bin/env python3
#
def i4vec_print(n, a, title):

    # *****************************************************************************80
    #
    # i4vec_print() prints an I4VEC.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 August 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer N, the dimension of the vector.
    #
    #    integer A(N), the vector to be printed.
    #
    #    string TITLE, a title.
    #
    print('')
    print(title)
    print('')
    for i in range(0, n):
        print('%6d  %6d' % (i, a[i]))

    return


def mono_between_enum(m, n1, n2):

    # *****************************************************************************80
    #
    # mono_between_enum() enumerates monomials in M dimensions of degrees in a range.
    #
    #  Discussion:
    #
    #    For M = 3, we have the following table:
    #
    #     N2 0  1  2  3  4  5  6   7   8
    #   N1 +----------------------------
    #    0 | 1  4 10 20 35 56 84 120 165
    #    1 | 0  3  9 19 34 55 83 119 164
    #    2 | 0  0  6 16 31 52 80 116 161
    #    3 | 0  0  0 10 25 46 74 110 155
    #    4 | 0  0  0  0 15 36 64 100 145
    #    5 | 0  0  0  0  0 21 49  85 130
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    23 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N1, N2, the minimum and maximum degrees.
    #    0 <= N1 <= N2.
    #
    #  Output:
    #
    #    integer VALUE, the number of monomials in M variables,
    #    of total degree between N1 and N2 inclusive.
    #
    from scipy.special import comb

    n1 = max(n1, 0)

    if (n2 < n1):
        value = 0
        return value

    if (n1 == 0):
        value = comb(n2 + m, n2)
    elif (n1 == n2):
        value = comb(n2 + m - 1, n2)
    else:
        n0 = n1 - 1
        value = comb(n2 + m, n2) - comb(n0 + m, n0)

    return value


def mono_between_enum_test():

    # *****************************************************************************80
    #
    # mono_between_enum_test() tests mono_between_enum().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    23 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    print('')
    print('mono_between_enum_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_between_enum can enumerate the number of monomials')
    print('  in M variables, of total degree between N1 and N2.')

    m = 3
    print('')
    print('  Using spatial dimension M = %d' % (m))
    print('')
    print('   N2:', end='')
    for n2 in range(0, 9):
        print('  %4d' % (n2), end='')
    print('')
    print('  N1 +---------------------------------------------------------------')
    for n1 in range(0, 9):
        print('  %2d |' % (n1), end='')
        for n2 in range(0, 9):
            v = mono_between_enum(m, n1, n2)
            print('  %4d' % (v), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_between_enum_test:')
    print('  Normal end of execution.')
    return


def mono_between_next_grevlex(m, n1, n2, x):

    # *****************************************************************************80
    #
    # mono_between_next_grevlex(): grevlex next monomial, degree between N1 and N2.
    #
    #  Discussion:
    #
    #    We consider all monomials in an M-dimensional space, with total
    #    degree N between N1 and N2, inclusive.
    #
    #    For example:
    #
    #    M = 3
    #    N1 = 2
    #    N2 = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     2        2
    #    2 |  0     1     1        2
    #    3 |  1     0     1        2
    #    4 |  0     2     0        2
    #    5 |  1     1     0        2
    #    6 |  2     0     0        2
    #      |
    #    7 |  0     0     3        3
    #    8 |  0     1     2        3
    #    9 |  1     0     2        3
    #   10 |  0     2     1        3
    #   11 |  1     1     1        3
    #   12 |  2     0     1        3
    #   13 |  0     3     0        3
    #   14 |  1     2     0        3
    #   15 |  2     1     0        3
    #   16 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N1, N2, the minimum and maximum degrees.
    #    0 <= N1 <= N2.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, N1 ].
    #    The last is [ N2, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (n1 < 0):
        print('')
        print('mono_between_next_grevlex - Fatal error!')
        print('  N1 < 0.')
        raise Exception('mono_between_next_grevlex - Fatal error!')

    if (n2 < n1):
        print('')
        print('mono_between_next_grevlex - Fatal error!')
        print('  N2 < N1.')
        raise Exception('mono_between_next_grevlex - Fatal error!')

    if (np.sum(x) < n1):
        print('')
        print('mono_between_next_grevlex - Fatal error!')
        print('  Input X sums to less than N1.')
        raise Exception('mono_between_next_grevlex - Fatal error!')

    if (n2 < np.sum(x)):
        print('')
        print('mono_between_next_grevlex - Fatal error!')
        print('  Input X sums to more than N2.')
        raise Exception('mono_between_next_grevlex - Fatal error!')

    if (n2 == 0):
        return x

    if (x[0] == n2):
        x[0] = 0
        x[m-1] = n1
    else:
        x = mono_next_grevlex(m, x)

    return x


def mono_between_next_grevlex_test():

    # *****************************************************************************80
    #
    # mono_between_next_grevlex_test() tests mono_between_next_grevlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    print('')
    print('mono_between_next_grevlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_between_next_grevlex can list the monomials')
    print('  in M variables, of total degree N between N1 and N2,')
    print('  in grevlex order, one at a time.')
    print('')
    print('  We start the process with (0,0,...,0,N1).')
    print('  The process ends with (N2,0,...,0,0)')

    n1 = 2
    n2 = 3

    print('')
    print('  Let M =   %d' % (m))
    print('      N1 =  %d' % (n1))
    print('      N2 =  %d' % (n2))
    print('')

    x = np.array([0, 0, n1], dtype=np.int32)

    i = 1

    while (True):

        print('  %2d    ' % (i), end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        if (x[0] == n2):
            break

        x = mono_between_next_grevlex(m, n1, n2, x)
        i = i + 1
#
#  Terminate.
#
    print('')
    print('mono_between_next_grevlex_test')
    print('  Normal end of execution.')
    return


def mono_between_next_grlex(m, n1, n2, x):

    # *****************************************************************************80
    #
    # mono_between_next_grlex(): grlex next monomial, degree between N1 and N2.
    #
    #  Discussion:
    #
    #    We consider all monomials in an M-dimensional space, with total
    #    degree N between N1 and N2, inclusive.
    #
    #    For example:
    #
    #    M = 3
    #    N1 = 2
    #    N2 = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     2        2
    #    2 |  0     1     1        2
    #    3 |  0     2     0        2
    #    4 |  1     0     1        2
    #    5 |  1     1     0        2
    #    6 |  2     0     0        2
    #      |
    #    7 |  0     0     3        3
    #    8 |  0     1     2        3
    #    9 |  0     2     1        3
    #   10 |  0     3     0        3
    #   11 |  1     0     2        3
    #   12 |  1     1     1        3
    #   13 |  1     2     0        3
    #   14 |  2     0     1        3
    #   15 |  2     1     0        3
    #   16 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N1, N2, the minimum and maximum degrees.
    #    0 <= N1 <= N2.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, N1 ].
    #    The last is [ N2, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (n1 < 0):
        print('')
        print('mono_between_next_grlex - Fatal error!')
        print('  N1 < 0.')
        raise Exception('mono_between_next_grlex - Fatal error!')

    if (n2 < n1):
        print('')
        print('mono_between_next_grlex - Fatal error!')
        print('  N2 < N1.')
        raise Exception('mono_between_next_grlex - Fatal error!')

    if (np.sum(x) < n1):
        print('')
        print('mono_between_next_grlex - Fatal error!')
        print('  Input X sums to less than N1.')
        raise Exception('mono_between_next_grlex - Fatal error!')

    if (n2 < np.sum(x)):
        print('')
        print('mono_between_next_grlex - Fatal error!')
        print('  Input X sums to more than N2.')
        raise Exception('mono_between_next_grlex - Fatal error!')

    if (n2 == 0):
        return x

    if (x[0] == n2):
        x[0] = 0
        x[m-1] = n1
    else:
        x = mono_next_grlex(m, x)

    return x


def mono_between_next_grlex_test():

    # *****************************************************************************80
    #
    # mono_between_next_grlex_test() tests mono_between_next_grlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    print('')
    print('mono_between_next_grlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_between_next_grlex can list the monomials')
    print('  in M variables, of total degree N between N1 and N2,')
    print('  in grlex order, one at a time.')
    print('')
    print('  We start the process with (0,0,...,0,N1).')
    print('  The process ends with (N2,0,...,0,0)')

    n1 = 2
    n2 = 3

    print('')
    print('  Let M =   %d' % (m))
    print('      N1 =  %d' % (n1))
    print('      N2 =  %d' % (n2))
    print('')

    x = np.array([0, 0, n1], dtype=np.int32)

    i = 1

    while (True):

        print('  %2d    ' % (i), end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        if (x[0] == n2):
            break

        x = mono_between_next_grlex(m, n1, n2, x)
        i = i + 1
#
#  Terminate
#
    print('')
    print('mono_between_next_grlex_test')
    print('  Normal end of execution.')
    return


def mono_between_random(m, n1, n2):

    # *****************************************************************************80
    #
    # mono_between_random(): random monomial with total degree between N1 and N2.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N1, N2, the minimum and maximum degrees.
    #    0 <= N1 <= N2.
    #
    #  Output:
    #
    #    integer X[M], the random monomial.
    #
    #    integer RANK, the rank of the monomial.
    #
    import numpy as np

    n1_copy = max(n1, 0)
    rank_min = mono_upto_enum(m, n1_copy - 1) + 1
    rank_max = mono_upto_enum(m, n2)
    rank = np.random.random_integers(rank_min, rank_max)
    x = mono_unrank_grlex(m, rank)

    return x, rank


def mono_between_random_test():

    # *****************************************************************************80
    #
    # mono_between_random_test() tests mono_between_random().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    m = 3

    print('')
    print('mono_between_random_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_between_random selects at random a monomial')
    print('  in M dimensions of total degree between N1 and N2.')

    n1 = 2
    n2 = 3

    print('')
    print('  Let M =  %d' % (m))
    print('      N1 = %d' % (n1))
    print('      N2 = %d' % (n2))
    print('')

    test_num = 5

    for test in range(0, test_num):
        x, rank = mono_between_random(m, n1, n2)
        print('  %2d    ' % (rank), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_between_random_test:')
    print('  Normal end of execution.')
    return


def monomial_test():

    # *****************************************************************************80
    #
    # monomial_test() tests monomial().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    27 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    print('')
    print('monomial_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  Test monomial().')

    mono_between_enum_test()
    mono_total_enum_test()
    mono_upto_enum_test()
    mono_next_grevlex_test()
    mono_next_grlex_test()
    mono_between_next_grevlex_test()
    mono_between_next_grlex_test()
    mono_total_next_grevlex_test()
    mono_total_next_grlex_test()
    mono_upto_next_grevlex_test()
    mono_upto_next_grlex_test()
    mono_rank_grlex_test()
    mono_unrank_grlex_test()
    mono_between_random_test()
    mono_total_random_test()
    mono_upto_random_test()
    mono_value_test()
    mono_print_test()
#
#  Terminate.
#
    print('')
    print('monomial_test:')
    print('  Normal end of execution.')
    return


def mono_next_grevlex(m, x):

    # *****************************************************************************80
    #
    # mono_next_grevlex(): grevlex next monomial.
    #
    #  Discussion:
    #
    #    Example:
    #
    #    M = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     0        0
    #      |
    #    2 |  0     0     1        1
    #    3 |  0     1     0        1
    #    4 |  1     0     0        1
    #      |
    #    5 |  0     0     2        2
    #    6 |  0     1     1        2
    #    7 |  1     0     1        2
    #    8 |  0     2     0        2
    #    9 |  1     1     0        2
    #   10 |  2     0     0        2
    #      |
    #   11 |  0     0     3        3
    #   12 |  0     1     2        3
    #   13 |  1     0     2        3
    #   14 |  0     2     1        3
    #   15 |  1     1     1        3
    #   16 |  2     0     1        3
    #   17 |  0     3     0        3
    #   18 |  1     2     0        3
    #   19 |  2     1     0        3
    #   20 |  3     0     0        3
    #
    #    Thanks to Stefan Klus for pointing out a discrepancy in a previous
    #    version of this code, 05 February 2015.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    05 February 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (np.sum(x) < 0):
        print('')
        print('mono_next_grevlex - Fatal error!')
        print('  Input X sums to less than 0.')
        raise Exception('mono_next_grevlex - Fatal error!')
#
#  Seek the first index 0 < I for which 0 < X(I).
#
    j = 0

    for i in range(1, m):
        if (0 < x[i]):
            j = i
            break

    if (j == 0):
        t = x[0]
        x[0] = 0
        x[m-1] = t + 1
    elif (j < m - 1):
        x[j] = x[j] - 1
        t = x[0] + 1
        x[0] = 0
        x[j-1] = x[j-1] + t
    elif (j == m - 1):
        t = x[0]
        x[0] = 0
        x[j-1] = t + 1
        x[j] = x[j] - 1

    return x


def mono_next_grevlex_test():

    # *****************************************************************************80
    #
    # mono_next_grevlex_test() tests mono_next_grevlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    05 February 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 4

    print('')
    print('mono_next_grevlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_next_grevlex computes the next monomial')
    print('  in M variables in grevlex order.')
    print('')
    print('  Let M =  %d' % (m))

    k = 0
    x = np.zeros(m)

    while (True):

        d = sum(x)
        print('  %2d  %2d  |' % (k, d), end='')
        for i in range(0, m):
            print('  %2d' % x[i], end='')
        print('')

        if (x[0] == 3):
            break
        k = k + 1
        x = mono_next_grevlex(m, x)
#
#  Terminate.
#
    print('')
    print('mono_next_grevlex_test')
    print('  Normal end of execution.')
    return


def mono_next_grlex(m, x):

    # *****************************************************************************80
    #
    # mono_next_grlex() returns the next monomial in grlex order.
    #
    #  Discussion:
    #
    #    Example:
    #
    #    M = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     0        0
    #      |
    #    2 |  0     0     1        1
    #    3 |  0     1     0        1
    #    4 |  1     0     0        1
    #      |
    #    5 |  0     0     2        2
    #    6 |  0     1     1        2
    #    7 |  0     2     0        2
    #    8 |  1     0     1        2
    #    9 |  1     1     0        2
    #   10 |  2     0     0        2
    #      |
    #   11 |  0     0     3        3
    #   12 |  0     1     2        3
    #   13 |  0     2     1        3
    #   14 |  0     3     0        3
    #   15 |  1     0     2        3
    #   16 |  1     1     1        3
    #   17 |  1     2     0        3
    #   18 |  2     0     1        3
    #   19 |  2     1     0        3
    #   20 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #

    #
    #  Ensure that 1 <= D.
    #
    if (m < 1):
        print('')
        print('mono_next_grlex - Fatal error!')
        print('  M < 1')
        raise Exception('mono_next_grlex - Fatal error!')
#
#  Ensure that 0 <= X(I).
#
    for i in range(0, m):
        if (x[i] < 0):
            print('')
            print('mono_next_grlex - Fatal error!')
            print('  X[I] < 0')
            raise Exception('mono_next_grlex - Fatal error!')
#
#  Find I, the index of the rightmost nonzero entry of X.
#
    i = 0
    for j in range(m, 0, -1):
        if (0 < x[j-1]):
            i = j
            break
#
#  set T = X(I)
#  set X(I) to zero,
#  increase X(I-1) by 1,
#  increment X(M) by T-1.
#
    if (i == 0):
        x[m-1] = 1
        return x
    elif (i == 1):
        t = x[0] + 1
        im1 = m
    elif (1 < i):
        t = x[i-1]
        im1 = i - 1

    x[i-1] = 0
    x[im1-1] = x[im1-1] + 1
    x[m-1] = x[m-1] + t - 1

    return x


def mono_next_grlex_test():

    # *****************************************************************************80
    #
    # mono_next_grlex_test() tests mono_next_grlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 4

    print('')
    print('mono_next_grlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_next_grlex computes the next monomial')
    print('  in M variables in grlex order.')
    print('')
    print('  Let M =  %d' % (m))

    a = 0
    b = 3

    for i in range(0, 10):

        x = np.random.random_integers(a, b, size=m)
        print('')
        print('  ', end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        for j in range(0, 5):
            x = mono_next_grlex(m, x)
            print('  ', end='')
            for k in range(0, m):
                print('%2d' % (x[k]), end='')
            print('')
#
#  Terminate.
#
    print('')
    print('mono_next_grlex_test')
    print('  Normal end of execution.')
    return


def mono_print(m, f, title):

    # *****************************************************************************80
    #
    # mono_print() prints a monomial.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer F[M], the exponents.
    #
    #    string TITLE, a title.
    #
    import sys

    sys.stdout.write(title)

    sys.stdout.write('  x^')
    if (1 < m or f[0] < 0):
        sys.stdout.write('(')
    for i in range(0, m):
        sys.stdout.write(repr(f[i]))
        if (i < m - 1):
            sys.stdout.write(',')
        elif (1 < m or f[0] < 0):
            sys.stdout.write(')')
    sys.stdout.write('\n')

    return


def mono_print_test():

    # *****************************************************************************80
    #
    # mono_print_test() tests mono_print().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('mono_print_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_print can print out a monomial.')
    print('')

    m = 1
    f = np.array([5], dtype=np.int32)
    mono_print(m, f, '  Monomial [5]:')

    m = 1
    f = np.array([-5], dtype=np.int32)
    mono_print(m, f, '  Monomial [-5]:')

    m = 4
    f = np.array([2, 1, 0, 3], dtype=np.int32)
    mono_print(m, f, '  Monomial [2,1,0,3]:')

    m = 3
    f = np.array([17, -3, 199], dtype=np.int32)
    mono_print(m, f, '  Monomial [17,-3,199]:')
#
#  Terminate.
#
    print('')
    print('mono_print_test')
    print('  Normal end of execution.')
    return


def mono_rank_grlex(m, x):

    # *****************************************************************************80
    #
    # mono_rank_grlex() computes the graded lexicographic rank of a monomial.
    #
    #  Discussion:
    #
    #    The graded lexicographic ordering is used, over all monomials in
    #    M dimensions, for total degree = 0, 1, 2, ...
    #
    #    For example, if M = 3, the ranking begins:
    #
    #    Rank  Sum    1  2  3
    #    ----  ---   -- -- --
    #       1    0    0  0  0
    #
    #       2    1    0  0  1
    #       3    1    0  1  0
    #       4    1    1  0  1
    #
    #       5    2    0  0  2
    #       6    2    0  1  1
    #       7    2    0  2  0
    #       8    2    1  0  1
    #       9    2    1  1  0
    #      10    2    2  0  0
    #
    #      11    3    0  0  3
    #      12    3    0  1  2
    #      13    3    0  2  1
    #      14    3    0  3  0
    #      15    3    1  0  2
    #      16    3    1  1  1
    #      17    3    1  2  0
    #      18    3    2  0  1
    #      19    3    2  1  0
    #      20    3    3  0  0
    #
    #      21    4    0  0  4
    #      ..   ..   .. .. ..
    #
    #  Licensing:
    #
    #   This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 October 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #    1 <= M.
    #
    #    integer X[M], the composition.
    #    For each 1 <= I <= M, we have 0 <= X(I).
    #
    #  Output:
    #
    #    integer RANK, the rank.
    #
    from scipy.special import comb
    import numpy as np
#
#  Ensure that 1 <= M.
#
    if (m < 1):
        print('')
        print('mono_rank_grlex - Fatal error!')
        print('  M < 1')
        raise Exception('mono_rank_grlex - Fatal error!')
#
#  Ensure that 0 <= X(I).
#
    for i in range(0, m):
        if (x[i] < 0):
            print('')
            print('mono_rank_grlex - Fatal error!')
            print('  X[I] < 0')
            raise Exception('mono_rank_grlex - Fatal error!')
#
#  NM = sum ( X )
#
    nm = np.sum(x)
#
#  Convert to KSUBSET format.
#
    ns = nm + m - 1
    ks = m - 1
    if (0 < ks):
        xs = np.zeros(ks, dtype=np.int32)
        xs[0] = x[0] + 1
        for i in range(2, m):
            xs[i-1] = xs[i-2] + x[i-1] + 1
#
#  Compute the rank.
#
    rank = 1

    for i in range(1, ks + 1):
        if (i == 1):
            tim1 = 0
        else:
            tim1 = xs[i-2]

        if (tim1 + 1 <= xs[i-1] - 1):
            for j in range(tim1 + 1, xs[i-1]):
                rank = rank + comb(ns - j, ks - i)

    for n in range(0, nm):
        rank = rank + comb(n + m - 1, n)

    return rank


def mono_rank_grlex_test():

    # ******************************************************************************/
    #
    # mono_rank_grlex_test() tests mono_rank_grlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3
    test_num = 8
    x_test = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 0, 1,
        0, 2, 0,
        1, 0, 2,
        0, 3, 1,
        3, 2, 1,
        5, 2, 1], dtype=np.int32)

    print('')
    print('mono_rank_grlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_rank_grlex returns the rank of a monomial in the sequence')
    print('  of all monomials in M dimensions, in grlex order.')

    print('')
    print('  Print a monomial sequence with ranks assigned.')

    n = 4

    print('')
    print('  Let M = %d' % (m))
    print('      N = %d' % (n))
    print('')

    x = np.zeros(m, dtype=np.int32)

    x[0] = 0
    x[1] = 0
    x[2] = 0

    i = 1

    while (True):
        print('  %2d    ' % (i), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')

        if (x[0] == n):
            break

        mono_upto_next_grlex(m, n, x)
        i = i + 1

    print('')
    print('  Now, given a monomial, retrieve its rank in the sequence:')
    print('')

    for test in range(0, test_num):
        for j in range(0, m):
            x[j] = x_test[j+test*m]
        rank = mono_rank_grlex(m, x)

        print('  %3d    ' % (rank), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_rank_grlex_test')
    print('  Normal end of execution.')
    return


def mono_total_enum(m, n):

    # *****************************************************************************80
    #
    # mono_total_enum() enumerates monomials in M dimensions of degree equal to N.
    #
    #  Discussion:
    #
    #    For M = 3, we have the following values:
    #
    #    N  VALUE
    #
    #    0    1
    #    1    3
    #    2    6
    #    3   10
    #    4   15
    #    5   21
    #
    #    In particular, VALUE(3,3) = 10 because we have the 10 monomials:
    #
    #      x^3, x^2y, x^2z, xy^2, xyz, xz^3, y^3, y^2z, yz^2, z^3.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    23 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the maximum degree.
    #
    #  Output:
    #
    #    integer VALUE, the number of monomials in M variables,
    #    of total degree N.
    #
    from scipy.special import comb

    value = comb(n + m - 1, n)

    return value


def mono_total_enum_test():

    # *****************************************************************************80
    #
    # mono_total_enum_test() tests mono_total_enum().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    23 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    print('')
    print('mono_total_enum_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_total_enum can enumerate the number of monomials')
    print('  in M variables, of total degree N.')

    print('')
    print('    N:', end='')
    for n in range(0, 9):
        print('  %4d' % (n), end='')
    print('')
    print('   M +---------------------------------------------------------------')
    for m in range(1, 9):
        print('  %2d |' % (m), end='')
        for n in range(0, 9):
            v = mono_total_enum(m, n)
            print('  %4d' % (v), end='')
        print('')
#
#  Terminate
#
    print('')
    print('mono_total_enum_test')
    print('  Normal end of execution.')
    return


def mono_total_next_grevlex(m, n, x):

    # *****************************************************************************80
    #
    # mono_total_next_grevlex(): grevlex next monomial, total degree equal to N.
    #
    #  Discussion:
    #
    #    We consider all monomials in an M-dimensional space, with total
    #    degree N.
    #
    #    For example:
    #
    #    M = 3
    #    N = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     3        3
    #    2 |  0     1     2        3
    #    3 |  1     0     2        3
    #    4 |  0     2     1        3
    #    5 |  1     1     1        3
    #    6 |  2     0     1        3
    #    7 |  0     3     0        3
    #    8 |  1     2     0        3
    #    9 |  2     1     0        3
    #   10 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the total degree.
    #    0 <= N1 <= N2.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, N ].
    #    The last is [ N, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (n < 0):
        print('')
        print('mono_total_next_grevlex - Fatal error!')
        print('  N < 0.')
        raise Exception('mono_total_next_grevlex - Fatal error!')

    if (np.sum(x) != n):
        print('')
        print('mono_total_next_grevlex - Fatal error!')
        print('  Input X sums is not N.')
        raise Exception('mono_total_next_grevlex - Fatal error!')

    if (n == 0):
        return x

    if (x[0] == n):
        x[0] = 0
        x[m-1] = n
    else:
        x = mono_next_grevlex(m, x)

    return x


def mono_total_next_grevlex_test():

    # *****************************************************************************80
    #
    # mono_total_next_grevlex_test() tests mono_total_next_grevlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    print('')
    print('mono_total_next_grevlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_total_next_grevlex can list the monomials')
    print('  in M variables, of total degree N,')
    print('  in grevlex order, one at a time.')
    print('')
    print('  We start the process with (0,0,...,0,N).')
    print('  The process ends with (N,0,...,0,0)')

    n = 3

    print('')
    print('  Let M =   %d' % (m))
    print('      N =   %d' % (n))
    print('')

    x = np.array([0, 0, n], dtype=np.int32)

    i = 1

    while (True):

        print('  %2d    ' % (i), end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        if (x[0] == n):
            break

        x = mono_total_next_grevlex(m, n, x)
        i = i + 1
#
#  Terminate
#
    print('')
    print('mono_total_next_grevlex_test')
    print('  Normal end of execution.')
    return


def mono_total_next_grlex(m, n, x):

    # *****************************************************************************80
    #
    # mono_total_next_grlex(): grlex next monomial, total degree equal to N.
    #
    #  Discussion:
    #
    #    We consider all monomials in an M-dimensional space, with total
    #    degree N.
    #
    #    For example:
    #
    #    M = 3
    #    N = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     3        3
    #    2 |  0     1     2        3
    #    3 |  0     2     1        3
    #    4 |  0     3     0        3
    #    5 |  1     0     2        3
    #    6 |  1     1     1        3
    #    7 |  1     2     0        3
    #    8 |  2     0     1        3
    #    9 |  2     1     0        3
    #   10 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the total degree.
    #    0 <= N1 <= N2.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, N ].
    #    The last is [ N, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (n < 0):
        print('')
        print('mono_total_next_grlex - Fatal error!')
        print('  N < 0.')
        raise Exception('mono_total_next_grlex - Fatal error!')

    if (np.sum(x) != n):
        print('')
        print('mono_total_next_grlex - Fatal error!')
        print('  Input X sums is not N.')
        raise Exception('mono_total_next_grlex - Fatal error!')

    if (n == 0):
        return x

    if (x[0] == n):
        x[0] = 0
        x[m-1] = n
    else:
        x = mono_next_grlex(m, x)

    return x


def mono_total_next_grlex_test():

    # *****************************************************************************80
    #
    # mono_total_next_grlex_test() tests mono_total_next_grlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    print('')
    print('mono_total_next_grlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_total_next_grlex can list the monomials')
    print('  in M variables, of total degree N,')
    print('  in grlex order, one at a time.')
    print('')
    print('  We start the process with (0,0,...,0,N).')
    print('  The process ends with (N,0,...,0,0)')

    n = 3

    print('')
    print('  Let M =   %d' % (m))
    print('      N =   %d' % (n))
    print('')

    x = np.array([0, 0, n], dtype=np.int32)

    i = 1

    while (True):

        print('  %2d    ' % (i), end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        if (x[0] == n):
            break

        x = mono_total_next_grlex(m, n, x)
        i = i + 1
#
#  Terminate.
#
    print('')
    print('mono_total_next_grlex_test')
    print('  Normal end of execution.')
    return


def mono_total_random(m, n):

    # *****************************************************************************80
    #
    # mono_total_random(): random monomial with total degree equal to N.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    21 November 2013
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the degree.
    #    0 <= N.
    #
    #  Output:
    #
    #    integer X[M], the random monomial.
    #
    #    integer RANK, the rank of the monomial.
    #
    import numpy as np

    rank_min = mono_upto_enum(m, n - 1) + 1
    rank_max = mono_upto_enum(m, n)
    rank = np.random.random_integers(rank_min, rank_max)
    x = mono_unrank_grlex(m, rank)

    return x, rank


def mono_total_random_test():

    # *****************************************************************************80
    #
    # mono_total_random_test() tests mono_total_random().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 November 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    m = 3

    print('')
    print('mono_total_random_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_total_random selects at random a monomial')
    print('  in M dimensions of total degree N.')

    n = 4

    print('')
    print('  Let M = %d' % (m))
    print('      N = %d' % (n))
    print('')

    test_num = 5

    for test in range(0, test_num):
        x, rank = mono_total_random(m, n)
        print('  %2d    ' % (rank), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_total_random_test:')
    print('  Normal end of execution.')
    return


def mono_unrank_grlex(m, rank):

    # *****************************************************************************80
    #
    # mono_unrank_grlex() computes the monomial of given grlex rank.
    #
    #  Licensing:
    #
    #   This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #    1 <= M.
    #
    #    integer RANK, the rank of the monomial.
    #
    #  Output:
    #
    #    integer X[M], the monomial.
    #
    from scipy.special import comb
    import numpy as np
#
#  Ensure that 1 <= M.
#
    if (m < 1):
        print('')
        print('mono_unrank_grlex - Fatal error!')
        print('  M < 1')
        raise Exception('mono_unrank_grlex - Fatal error!')
#
#  Ensure that 1 <= RANK.
#
    if (rank < 1):
        print('')
        print('mono_unrank_grlex - Fatal error!')
        print('  RANK < 1')
        raise Exception('mono_unrank_grlex - Fatal error!')

    x = np.zeros(m, dtype=np.int32)
#
#  Special case M = 1.
#
    if (m == 1):
        x[0] = rank - 1
        return x
#
#  Determine the appropriate value of NM.
#  Do this by adding up the number of compositions of sum 0, 1, 2,
#  ..., without exceeding RANK.  Moreover, RANK - this sum essentially
#  gives you the rank of the composition within the set of compositions
#  of sum NM.  And that's the number you need in order to do the
#  unranking.
#
    rank1 = 1
    nm = -1
    while (True):
        nm = nm + 1
        r = comb(nm + m - 1, nm)
        if (rank < rank1 + r):
            break
        rank1 = rank1 + r

    rank2 = rank - rank1
#
#  Convert to KSUBSET format.
#  Apology: an unranking algorithm was available for KSUBSETS,
#  but not immediately for compositions.  One day we will come back
#  and simplify all this.
#
    ks = m - 1
    ns = nm + m - 1
    xs = np.zeros(ks, dtype=np.int32)

    nksub = comb(ns, ks)

    j = 1

    for i in range(1, ks + 1):
        r = comb(ns - j, ks - i)

        while (r <= rank2 and 0 < r):
            rank2 = rank2 - r
            j = j + 1
            r = comb(ns - j, ks - i)

        xs[i-1] = j
        j = j + 1
#
#  Convert from KSUBSET format to COMP format.
#
    x[0] = xs[0] - 1
    for i in range(2, m):
        x[i-1] = xs[i-1] - xs[i-2] - 1
    x[m-1] = ns - xs[ks-1]

    return x


def mono_unrank_grlex_test():

    # ******************************************************************************/
    #
    # mono_unrank_grlex_test() tests mono_unrank_grlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3
    print('')
    print('mono_unrank_grlex')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_unrank_grlex is given a rank, and returns the corresponding')
    print('  monomial in the sequence of all monomials in M dimensions')
    print('  in grlex order.')

    print('')
    print('  For reference, print a monomial sequence with ranks.')

    n = 4
    rank_max = mono_upto_enum(m, n)

    print('')
    print('  Let M = %d' % (m))
    print('      N = %d' % (n))
    print('')

    x = np.zeros(m, dtype=np.int32)

    i = 1

    while (True):
        print('  %2d    ' % (i), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')

        if (x[0] == n):
            break

        mono_upto_next_grlex(m, n, x)
        i = i + 1

    print('')
    print('  Now choose random ranks between 1 and %d' % (rank_max))
    print('')

    test_num = 5

    for test in range(0, test_num):
        rank = np.random.random_integers(1, rank_max)
        x = mono_unrank_grlex(m, rank)
        print('  %2d    ' % (rank), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_unrank_grlex_test')
    print('  Normal end of execution.')
    return


def mono_upto_enum(m, n):

    # *****************************************************************************80
    #
    # mono_upto_enum() enumerates monomials in M dimensions of degree up to N.
    #
    #  Discussion:
    #
    #    For M = 2, we have the following values:
    #
    #    N  VALUE
    #
    #    0    1
    #    1    3
    #    2    6
    #    3   10
    #    4   15
    #    5   21
    #
    #    In particular, VALUE(2,3) = 10 because we have the 10 monomials:
    #
    #      1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    23 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the maximum degree.
    #
    #  Output:
    #
    #    integer VALUE, the number of monomials in
    #    M variables, of total degree N or less.
    #
    from scipy.special import comb

    value = comb(n + m, n)

    return value


def mono_upto_enum_test():

    # *****************************************************************************80
    #
    # mono_upto_enum_test() tests mono_upto_enum().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    23 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    print('')
    print('mono_upto_enum_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_upto_enum can enumerate the number of monomials')
    print('  in M variables, of total degree between 0 and N.')

    print('')
    print('    N:', end='')
    for n in range(0, 9):
        print('  %4d' % (n), end='')
    print('')
    print('   M +---------------------------------------------------------------')
    for m in range(1, 9):
        print('  %2d |' % (m), end='')
        for n in range(0, 9):
            v = mono_upto_enum(m, n)
            print(' %5d' % (v), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_upto_enum_test')
    print('  Normal end of execution.')
    return


def mono_upto_next_grevlex(m, n, x):

    # *****************************************************************************80
    #
    # mono_upto_next_grevlex(): grevlex next monomial, total degree up to N.
    #
    #  Discussion:
    #
    #    We consider all monomials in an M-dimensional space, with total
    #    degree N.
    #
    #    For example:
    #
    #    M = 3
    #    N = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     0        0
    #      |
    #    2 |  0     0     1        1
    #    3 |  0     1     0        1
    #    4 |  1     0     0        1
    #      |
    #    5 |  0     0     2        2
    #    6 |  0     1     1        2
    #    7 |  1     0     1        2
    #    8 |  0     2     0        2
    #    9 |  1     1     0        2
    #   10 |  2     0     0        2
    #      |
    #   11 |  0     0     3        3
    #   12 |  0     1     2        3
    #   13 |  1     0     2        3
    #   14 |  0     2     1        3
    #   15 |  1     1     1        3
    #   16 |  2     0     1        3
    #   17 |  0     3     0        3
    #   18 |  1     2     0        3
    #   19 |  2     1     0        3
    #   20 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the total degree.
    #    0 <= N.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, 0 ].
    #    The last is [ N, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (n < 0):
        print('')
        print('mono_upto_next_grevlex - Fatal error!')
        print('  N < 0.')
        raise Exception('mono_upto_next_grevlex - Fatal error!')

    if (np.sum(x) < 0):
        print('')
        print('mono_upto_next_grevlex - Fatal error!')
        print('  Input X sum is less than 0.')
        raise Exception('mono_upto_next_grevlex - Fatal error!')

    if (n < np.sum(x)):
        print('')
        print('mono_upto_next_grevlex - Fatal error!')
        print('  Input X sum is more than N.')
        raise Exception('mono_upto_next_grevlex - Fatal error!')

    if (n == 0):
        return x

    if (x[0] == n):
        x[0] = 0
    else:
        x = mono_next_grevlex(m, x)

    return x


def mono_upto_next_grevlex_test():

    # *****************************************************************************80
    #
    # mono_upto_next_grevlex_test() tests mono_upto_next_grevlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    print('')
    print('mono_upto_next_grevlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_upto_next_grevlex can list the monomials')
    print('  in M variables, of total degree up to N,')
    print('  in grevlex order, one at a time.')
    print('')
    print('  We start the process with (0,0,...,0,0).')
    print('  The process ends with (N,0,...,0,0)')

    n = 4

    print('')
    print('  Let M =   %d' % (m))
    print('      N =   %d' % (n))
    print('')

    x = np.array([0, 0, 0], dtype=np.int32)

    i = 1

    while (True):

        print('  %2d    ' % (i), end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        if (x[0] == n):
            break

        x = mono_upto_next_grevlex(m, n, x)
        i = i + 1
#
#  Terminate.
#
    print('')
    print('mono_upto_next_grevlex_test')
    print('  Normal end of execution.')
    return


def mono_upto_next_grlex(m, n, x):

    # *****************************************************************************80
    #
    # mono_upto_next_grlex(): grlex next monomial, total degree up to N.
    #
    #  Discussion:
    #
    #    We consider all monomials in an M-dimensional space, with total
    #    degree N.
    #
    #    For example:
    #
    #    M = 3
    #    N = 3
    #
    #    #  X(1)  X(2)  X(3)  Degree
    #      +------------------------
    #    1 |  0     0     0        0
    #      |
    #    2 |  0     0     1        1
    #    3 |  0     1     0        1
    #    4 |  1     0     0        1
    #      |
    #    5 |  0     0     2        2
    #    6 |  0     1     1        2
    #    7 |  0     2     0        2
    #    8 |  1     0     1        2
    #    9 |  1     1     0        2
    #   10 |  2     0     0        2
    #      |
    #   11 |  0     0     3        3
    #   12 |  0     1     2        3
    #   13 |  0     2     1        3
    #   14 |  0     3     0        3
    #   15 |  1     0     2        3
    #   16 |  1     1     1        3
    #   17 |  1     2     0        3
    #   18 |  2     0     1        3
    #   19 |  2     1     0        3
    #   20 |  3     0     0        3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the maximum degree.
    #    0 <= N.
    #
    #    integer X[M], the current monomial.
    #    The first element is X = [ 0, 0, ..., 0, 0 ].
    #    The last is [ N, 0, ..., 0, 0 ].
    #
    #  Output:
    #
    #    integer X[M], the next monomial.
    #
    import numpy as np

    if (n < 0):
        print('')
        print('mono_upto_next_grlex - Fatal error!')
        print('  N < 0.')
        raise Exception('mono_upto_next_grlex - Fatal error!')

    if (np.sum(x) < 0):
        print('')
        print('mono_upto_next_grlex - Fatal error!')
        print('  Input X sum is less than 0.')
        raise Exception('mono_upto_next_grlex - Fatal error!')

    if (n < np.sum(x)):
        print('')
        print('mono_upto_next_grlex - Fatal error!')
        print('  Input X sum is more than N.')
        raise Exception('mono_upto_next_grlex - Fatal error!')

    if (n == 0):
        return x

    if (x[0] == n):
        x[0] = 0
    else:
        x = mono_next_grlex(m, x)

    return x


def mono_upto_next_grlex_test():

    # *****************************************************************************80
    #
    # mono_upto_next_grlex_test() tests mono_upto_next_grlex().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    24 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    print('')
    print('mono_upto_next_grlex_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_upto_next_grlex can list the monomials')
    print('  in M variables, of total degree up to N,')
    print('  in grlex order, one at a time.')
    print('')
    print('  We start the process with (0,0,...,0,0).')
    print('  The process ends with (N,0,...,0,0)')

    n = 4

    print('')
    print('  Let M =   %d' % (m))
    print('      N =   %d' % (n))
    print('')

    x = np.array([0, 0, 0], dtype=np.int32)

    i = 1

    while (True):

        print('  %2d    ' % (i), end='')
        for k in range(0, m):
            print('%2d' % (x[k]), end='')
        print('')

        if (x[0] == n):
            break

        x = mono_upto_next_grlex(m, n, x)
        i = i + 1
#
#  Terminate.
#
    print('')
    print('mono_upto_next_grlex_test')
    print('  Normal end of execution.')
    return


def mono_upto_random(m, n):

    # *****************************************************************************80
    #
    # mono_upto_random(): random monomial with total degree less than or equal to N.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the degree.
    #    0 <= N.
    #
    #  Output:
    #
    #    integer X[M], the random monomial.
    #
    #    integer RANK, the rank of the monomial.
    #
    import numpy as np

    rank_min = 1
    rank_max = mono_upto_enum(m, n)
    rank = np.random.random_integers(rank_min, rank_max)
    x = mono_unrank_grlex(m, rank)

    return x, rank


def mono_upto_random_test():

    # *****************************************************************************80
    #
    # mono_upto_random_test() tests mono_upto_random().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import platform

    m = 3

    print('')
    print('mono_upto_random_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_upto_random selects at random a monomial')
    print('  in M dimensions of total degree no greater than N.')

    n = 4

    print('')
    print('  Let M = %d' % (m))
    print('      N = %d' % (n))
    print('')

    test_num = 5

    for test in range(0, test_num):
        x, rank = mono_upto_random(m, n)
        print('  %2d    ' % (rank), end='')
        for j in range(0, m):
            print('%2d' % (x[j]), end='')
        print('')
#
#  Terminate.
#
    print('')
    print('mono_upto_random_test:')
    print('  Normal end of execution.')
    return


def mono_value(m, n, f, x):

    # *****************************************************************************80
    #
    # mono_value() evaluates a monomial.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Input:
    #
    #    integer M, the spatial dimension.
    #
    #    integer N, the number of evaluation points.
    #
    #    integer F[M], the exponents of the monomial.
    #
    #    real X[M*N], the coordinates of the evaluation points.
    #
    #  Output:
    #
    #    real mono_value[N], the value of the monomial at X.
    #
    import numpy as np

    v = np.zeros(n, dtype=np.float64)

    for j in range(0, n):
        v[j] = 1.0
        for i in range(0, m):
            v[j] = v[j] * (x[i+j*m] ** f[i])

    return v


def mono_value_test():

    # *****************************************************************************80
    #
    # mono_value_test() tests mono_value().
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    25 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3
    nx = 2
    x = np.array([1.0, 2.0, 3.0, -2.0, 4.0, 1.0], dtype=np.float64)

    print('')
    print('mono_value_test')
    print('  Python version: %s' % (platform.python_version()))
    print('  mono_value evaluates a monomial.')

    n = 6

    print('')
    print('  Let M = %d' % (m))
    print('      N = %d' % (n))

    test_num = 5

    for test in range(0, test_num):
        f, rank = mono_upto_random(m, n)
        print('')
        mono_print(m, f, '  M(X) = ')
        v = mono_value(m, nx, f, x)
        for j in range(0, nx):
            print('  M(%g,%g,%g) = %g' % (x[0+j*m], x[1+j*m], x[2+j*m], v[j]))
#
#  Terminate.
#
    print('')
    print('mono_value_test:')
    print('  Normal end of execution.')
    return


def timestamp():

    # *****************************************************************************80
    #
    # timestamp() prints the date as a timestamp.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 April 2013
    #
    #  Author:
    #
    #    John Burkardt
    #
    import time

    t = time.time()
    print(time.ctime(t))

    return None


if (__name__ == '__main__'):
    timestamp()
    monomial_test()
    timestamp()
