import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import os
import warnings


import koopcore
from koopcore.auxilliary.data_classes import trajectory as trajectory


def get_subplots_fig_ax(fig_ax, **subplots_kwargs):
    if fig_ax is not None:
        return fig_ax
    subplots_kwargs.update({"squeeze": False})
    return plt.subplots(**subplots_kwargs)


def _save(save, f, dir, tags):
    if save:
        os.makedirs(f"./{dir}/", exist_ok=True)
        name = ""

        def _f(tag, name):
            if tag is None:
                return name
            if tag != "":
                return name + "_" + tag
            else:
                return tag
        name = ""
        for tag in tags:
            name = _f(tag, name)
        f.savefig(f"./{dir}/{name}.png")


def _close(close, f):
    if close:
        if hasattr(f, '__iter__'):
            for fi in f:
                _close(close, fi)
        else:
            plt.close(f)


def plot_data(gamma, prepend_tag="", d_plot="all", dd_plot=[], i_plot="all", dir="results", cmap_name=None,save=False, close=False, fig=None, subplots_kw={}, plt_params={}):
    d = gamma.d
    N1 = gamma.N
    from matplotlib import colormaps
    if type(i_plot) == type("all") and i_plot == "all":
        i_plot = range(N1)
    if type(i_plot) == int:
        i_plot = range(i_plot)
    if type(d_plot) == type("all") and d_plot == "all":
        d_plot = range(d)
    if type(d_plot) == int:
        d_plot = list(range(d_plot))
    if d == 2 and not (0, 1) in dd_plot:
        dd_plot.append((0, 1))
    
    plt_params.update({"linestyle" :plt_params.get("linestyle", "solid")})
    plt_params.update({"linewidth" :plt_params.get("linewidth", 2)})
    plt_params.update({"alpha" :plt_params.get("alpha", 1.)})
    
    _axs = len(d_plot) + len(dd_plot)
    _cols = min(_axs, 3)
    _rows = _axs // _cols + (_axs % _cols)

    subplots_kw.update({"figsize": subplots_kw.get("figsize", (15, 5))})
    if cmap_name:
        cmap = colormaps[cmap_name]
        colormap = lambda i: cmap(15 / (i+1) /len(i_plot) )
    else:
        colormap = lambda i: np.random.rand(3).tolist()
    if fig and len(fig.axes) >= _cols * _rows:
        f, a = (fig, np.array(fig.axes).reshape(-1, _cols))
    else:
        f, a = plt.subplots(_rows, _cols, squeeze=False, **subplots_kw)
    for ic, i in enumerate(i_plot):
        color = colormap(ic)
        for id in d_plot:
            i1 = id // _cols
            i2 = id % _cols
            a[i1, i2].plot(gamma.T[i].real, gamma.X[i, :, id].real, color=color, **plt_params)
            a[i1, i2].scatter(
                gamma.T[i, 0].real, gamma.X[i, 0, id].real, color=color,**plt_params)
            a[i1, i2].scatter(
                gamma.T[i, -1].real, gamma.X[i, -1, id].real, color=color,**plt_params)
            a[i1, i2].set_xlabel("Time")
            a[i1, i2].set_ylabel(f"Data Dimension {id}")
        for idd, dd in enumerate(dd_plot):
            i1 = (idd + len(d_plot)) // _cols
            i2 = (idd + len(d_plot)) % _cols
            a[i1, i2].scatter(gamma.X[i, -1, 0].real,
                             gamma.X[i, -1, 1].real, color=color, **plt_params)
            a[i1, i2].scatter(gamma.X[i, 0, 0].real,
                             gamma.X[i, 0, 1].real, color=color, **plt_params)
            a[i1, i2].plot(gamma.X[i, :, 0].real,
                          gamma.X[i, :, 1].real, color=color, **plt_params)
            a[i1, i2].set_xlabel(f"Data Dimension {0}")
            a[i1, i2].set_ylabel(f"Data Dimension {1}")
            a[i1, i2].set_aspect("equal")
    f.suptitle(
        f"{prepend_tag.replace('_', ' ')}: showing Rollouts of {len(i_plot)} out of {N1} Trajectories")
    if save:
        _save(save, f, dir, [prepend_tag, "data"])
    if close:
        plt.draw()
        _close(close, f)
    return (f, a)


def plot_kernel_2d(
    kernel,
    min_max=[-1, 1],
    center=np.zeros([1, 2]),
    prepend_tag=None,
    d=2,
    resolution=10,
    dtype=jnp.float32,
    dir="results",
    save=False,
    close=False,
    fig_ax_in=None,
    colorbar=True
):
    if fig_ax_in is None:
        plt.figure()
        ax = plt.gcf().add_subplot(111)
    else:
        ax = fig_ax_in[1][0, 0]
    _ref = np.array(koopcore.auxilliary.sample.sample_grid(
        resolution, d, min_max, dtype)[0])
    _kref = jnp.flip(
        (np.array(kernel(center, _ref)).reshape(resolution, resolution)), 0)

    im = ax.imshow(_kref.real, aspect="auto", extent=[*min_max, *min_max])
    if colorbar:
        plt.colorbar(im)

    xt = (_ref[:, 0][[0, 33, 66, -1]].real).tolist()
    xt.insert(2, 1.0)
    yt = (_ref[:, 1][[0, 33, 66, -1]].real).tolist()
    yt.insert(2, 1.0)
    ax.set_yticks(yt)
    ax.set_xticks(xt)
    ax.set_xlabel("d_0")
    ax.set_title(f"{prepend_tag}_kernel")
    _save(save, plt.gcf(), dir, [prepend_tag, "kernel"])
    _close(close, plt.gcf())


def plot_gramian(gramian, prepend_tag=None, dir="results", save=False, close=False, fig_ax_in=None):
    if fig_ax_in is not None:
        f, a = fig_ax_in
    if np.any(np.iscomplex(gramian)):
        if fig_ax_in is None:
            f, a = plt.subplots(1, 2, squeeze=False)
        im = a[0, 0].imshow(gramian.real, aspect="auto", interpolation="none")
        plt.colorbar(im, orientation="horizontal")
        im = a[0, 1].imshow(gramian.imag, aspect="auto", interpolation="none")
        plt.colorbar(im, orientation="horizontal")
    else:
        if fig_ax_in is None:
            f, a = plt.subplots(1, 1, squeeze=False)
        im = a[0, 0].imshow(gramian.real, aspect="auto", interpolation="none")
        plt.colorbar(im, orientation="horizontal")
    _save(save, f, dir, [prepend_tag, "gramian"])
    _close(close, f)


def plot_eigevalues(ev, ev_are_dt=True, dt=None, fig_ax=None, prepend_tag="", i_plot="all", dir="results", save=False, close=False):
    import koopcore.auxilliary.eigenvalues as eigenvalues
    import matplotlib.patches as plp
    ncols = (dt is not None) + 1
    f, a = get_subplots_fig_ax(fig_ax, nrows=1, ncols=ncols)
    ev = np.asarray(ev)
    D = len(ev)
    ev = ev[i_plot] if i_plot != "all" else ev

    def plot_ev_dt(ev_dt, a):
        a.scatter(ev_dt.real, ev_dt.imag)
        a.add_patch(plp.Circle((0, 0), 1, fill=False))
        a.set_title("discrete time eigenvalues")
        return a

    def plot_ev_ct(ev_ct, a):
        a.scatter(ev_ct.real, ev_ct.imag)
        min_max = a.get_ylim()
        a.plot([0, 0], min_max, color="black", linewidth=0.5)
        a.set_ylim(min_max)
        a.set_title("continuous time eigenvalues")
        return a
    if ev_are_dt:
        ev_dt = ev
        a[0, 0] = plot_ev_dt(ev_dt, a[0, 0])
        a[0, 0].axis("equal")
        if dt:
            ev_ct = eigenvalues.convert_dt_to_ct(ev_dt, dt)
            a[0, 1] = plot_ev_ct(ev_ct, a[0, 1])

    else:
        ev_ct = ev
        a[0, 0] = plot_ev_ct(ev_ct, a[0, 0])
        if dt:
            ev_dt = eigenvalues.convert_ct_to_dt(ev_ct, dt)
            a[0, 1] = plot_ev_ct(ev_ct, a[0, 1])
    f.suptitle(f"showing {len(ev)} out of {D} eigenvalues")
    _save(save, f, dir, [prepend_tag, "phi_traj"])
    _close(close, f)
    return fig_ax
