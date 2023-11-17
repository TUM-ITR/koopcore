import jax.numpy as jnp
import jax
from jax import vmap, jit
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from typing import Literal
import koopcore


def make_kernel(kernel_name: Literal["square-exponential",
                                     "kern_poly",
                                     "kern_matern_1",
                                     "kern_matern_3",
                                     "kern_matern_5"],
                *args, **p_kernel):
    def kern_poly(xq, xb):
        dot = jnp.dot(xq, xb)
        similarity = (dot + p_kernel["inhomogenity"]) ** p_kernel["degree"]
        return similarity

    def kern_matern_1(xq, xb):
        euc_dist = jnp.sum(((xq - xb) ** 2 / p_kernel["scale"] ** 2)) ** 0.5
        similarity = jnp.exp(-euc_dist)
        return similarity

    def kern_matern_3(xq, xb):
        euc_dist = jnp.sum(((xq - xb) ** 2 / p_kernel["scale"] ** 2)) ** 0.5
        sqrt3 = 1.73205080757
        similarity = (1.0 + sqrt3 * euc_dist) * jnp.exp(-sqrt3 * euc_dist)
        return similarity

    def kern_matern_5(xq, xb):
        euc_dist = jnp.sum(((xq - xb) ** 2 / p_kernel["scale"] ** 2)) ** 0.5
        sqrt5 = 2.2360679775
        similarity = (1.0 + sqrt5 * euc_dist + 5 / 3 * euc_dist**2) * jnp.exp(
            -sqrt5 * euc_dist
        )
        return similarity

    def kern_sq_exp(xq, xb):
        sq_euc_dist = jnp.sum(((xq - xb) ** 2 / p_kernel["scale"] ** 2))
        similarity = jnp.exp(-0.5 * sq_euc_dist)
        return similarity

    if kernel_name == "square-exponential":
        kernel = kern_sq_exp
    if kernel_name == "kern_poly":
        kernel = kern_poly
    if kernel_name == "kern_matern_1":
        kernel = kern_matern_1
    if kernel_name == "kern_matern_3":
        kernel = kern_matern_3
    if kernel_name == "kern_matern_5":
        kernel = kern_matern_5

    _batched_kernel_xb = vmap(kernel, in_axes=[None, 0], out_axes=0)
    batched_kernel = vmap(_batched_kernel_xb, in_axes=[0, None], out_axes=0)
    return batched_kernel


def main():
    kernel = make_kernel("square-exponential", 2, 100)

    def f_d(x):
        d = jnp.sum((x) ** 2) ** 0.5
        return jnp.sin(d * 10.0)

    Nit, Bit = 1000, 1
    N = Nit * Bit
    x_data = jnp.array(koopcore.sample.sample_box(N)[0])
    x_data_2 = jnp.array(koopcore.sample.sample_box(2000)[0])

    y_data = vmap(f_d)(x_data)
    y_data_2 = vmap(f_d)(x_data_2)

    def loss(a, b):
        return jnp.sum((a - b) ** 2) / a.shape[0]

    def pred(w, x=None, gramian=None, gramian_given=False):
        if gramian_given:
            _gramian = gramian
        else:
            _gramian = kernel(x, x_data)
        return _gramian @ w

    def fit_lstsq(gramian, y):
        w = jnp.linalg.lstsq(gramian, y)[0]
        return w

    def fit_solve(gramian, y):
        w = jnp.linalg.solve(gramian, y)
        return w

    def fit_sgd(gramian, y):
        w, info = jax.scipy.sparse.linalg.cg(gramian, y)
        return w

    print("make gramian")
    t0_g = timer()
    gramian = jit(kernel)(x_data, x_data).block_until_ready()
    t1_g = timer()
    print("\tfinished in {:.2e}s".format(t1_g - t0_g))
    print("solver           loss     time / s")
    t0_it = timer()
    w_it = jit(fit_sgd)(gramian, y_data).block_until_ready()
    t1_it = timer()
    y_rec_it = pred(w_it, gramian=gramian, gramian_given=True)
    l_train_it = loss(y_data, y_rec_it)
    print("iterative:       {:.2e}   {:.2e}".format(l_train_it, t1_it - t0_it))
    t0_lstsq = timer()
    w_lstsq = jit(fit_lstsq)(gramian, y_data).block_until_ready()
    t1_lstsq = timer()
    y_rec_lstsq = pred(w_lstsq, gramian=gramian, gramian_given=True)
    l_train_lstsq = loss(y_data, y_rec_lstsq)
    print(
        "leastsquare:     {:.2e}   {:.2e}".format(l_train_lstsq, t1_lstsq - t0_lstsq),
    )
    t0_ex = timer()
    w_ex = jit(fit_solve)(gramian, y_data).block_until_ready()
    t1_ex = timer()
    y_rec_ex = pred(w_ex, gramian=gramian, gramian_given=True)
    l_train_ex = loss(y_data, y_rec_ex)
    print(
        "exact:           {:.2e}   {:.2e}".format(l_train_ex, t1_ex - t0_ex),
    )
    l_train_diff = loss(y_rec_it, y_rec_ex)

    print("norm difference: {:.2e}".format(l_train_diff))

    y_2_rec_it = pred(w_it, x=x_data_2)

    f, a = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    a.plot_trisurf(x_data[:, 0], x_data[:, 1], y_data)
    plt.savefig("./results/__debug_1.png")
    a.clear()
    a.plot_trisurf(x_data[:, 0], x_data[:, 1], y_rec_it)
    plt.savefig("./results/__debug_2.png")
    a.clear()
    a.plot_trisurf(x_data_2[:, 0], x_data_2[:, 1], y_data_2)
    plt.savefig("./results/__debug_3.png")
    a.clear()
    a.plot_trisurf(x_data_2[:, 0], x_data_2[:, 1], y_2_rec_it)
    plt.savefig("./results/__debug_4.png")
    a.clear()
    a.plot_trisurf(
        x_data[:, 0], x_data[:, 1], kernel(jnp.array([[0.0, 0.0]]), x_data).squeeze()
    )
    plt.savefig("./results/__debug_g.png")

    print("done")


if __name__ == "__main__":
    main()
