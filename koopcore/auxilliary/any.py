import jax.numpy as jnp
import jax
import pandas as pd
import os
from jaxtyping import Array, Num


def get_all_one_step(X):
    conv_y = jnp.array([1, 0])
    conv_x = jnp.array([0, 1])
    def conv_val(a, v): return jnp.convolve(a, v, mode="valid")
    _Y = jax.vmap(jax.vmap(conv_val, [0, None], 0), [2, None], 2)(
        X, conv_y).reshape(-1, *X.shape[2:])
    _X = jax.vmap(jax.vmap(conv_val, [0, None], 0), [2, None], 2)(
        X, conv_x).reshape(-1, *X.shape[2:])
    return _X, _Y


def rmse(X, Y=None, mean_axis=[0], sum_axis=None):
    r = X - Y if Y is not None else X
    if sum_axis is None:
        sum_axis = list(
            set([i for i in range(len(X.shape))]).difference(set(mean_axis)))
    return jnp.squeeze(
        jnp.expand_dims(jnp.mean(
            jnp.expand_dims(jnp.sum((r)**2, axis=sum_axis), sum_axis), axis=mean_axis
        )**0.5, axis=mean_axis),
        axis=set(mean_axis).union(set(sum_axis))
    )


def p_norm(e, a, p, mean=True): 
    if mean:
        return (e.__abs__()**p).mean(axis=a)**(1 / p)
    return (e.__abs__()**p).sum(axis=a)**(1 / p)
def sup_norm(e, a): return (e.__abs__()).max(axis=a)
def mah_norm(e, a): return p_norm(e, a, 1)
def sq_loss(e, a): return p_norm(e, a, 2)**2


def log_ls_fit(x, y, w=None):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if w is None:
        w = jnp.ones_like(y)
    idrop = jnp.where(w < 1e-3)[0]
    lx = jnp.log(x)
    ly = jnp.log(y)
    w = jnp.delete(w, idrop)
    lx = jnp.delete(lx, idrop)
    ly = jnp.delete(ly, idrop)

    o = jnp.ones_like(lx)
    xr = jnp.stack([o, lx], axis=1)

    C, u = jnp.linalg.pinv(xr.T @ jnp.diag(w**-2) @ xr) @ xr.T @ ly

    return jnp.exp(C), u


def ls_fit(x, y, s=0):
    ym = y[-s:].mean()
    xm = x[-s:].mean()
    lx = x[-s:]
    ly = y[-s:]
    u = sum((lx - xm) * (ly - ym)) / sum((lx - xm)**2)
    C = ym.mean() - u * xm
    return C, u


def array_to_df(arr, N_RUNS, D_list, H_list, N_list, dir, name, save=True):
    # -> R, train/test, D, H, N, inv/isom
    arr_values = arr.transpose(3, 0, 2, 1, 4, 5)
    arr_values = jnp.concatenate([arr_values, jnp.expand_dims(
        (arr_values[1] - arr_values[0]).__abs__(), 0)])
    arr_index = []
    for iR in range(N_RUNS):
        for iD in D_list:
            for iH in H_list:
                for iN in N_list:
                    arr_index.append((iR, iD, iH, iN))
    index_names = ["R", "D", "H", "N"]
    idx = pd.MultiIndex.from_tuples(arr_index, names=index_names)
    df = pd.concat([pd.DataFrame(i_arr, columns=["invariant", "isometric"], index=idx)
                   for i_arr in arr_values.reshape(3, -1, 2)], axis=1, keys=["train", "test", "excess"])
    df.to_csv(os.path.join(dir, name), index_label=index_names)
    return df


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def _to_int(v):
    try:
        return int(v)
    except:
        return v


def nested_dict_to_df(
    values_dict,
    lowkeys=["train", "test"],
    highkeys=["trajectories", "risk", "residuals"]
):
    _sl = slice(None)
    fd = flatten_dict(values_dict)
    fd = dict(zip(fd.keys(), [[it] for it in fd.values()]))
    dfnn = pd.DataFrame.from_dict(fd, orient="index")
    dfnn.index = pd.MultiIndex.from_tuples(dfnn.index)

    df = pd.concat([pd.concat([
        dfnn.loc[(_sl, _sl, _sl, _sl, ki, _sl)
                 ].loc[(_sl, _sl, _sl, _sl, kiii)]
        for kiii in lowkeys], keys=lowkeys, axis=1)
        for ki in highkeys], keys=highkeys, axis=1)
    df = df.droplevel(-1, axis=1)
    df.index = pd.MultiIndex.from_tuples(
        [[_to_int(i) for i in ii] for ii in df.index])
    df.index.names = ["R", "H", "D", "N"]
    return df.sort_index()


def chunk(A: Array, chunks: int):
    return jnp.reshape(A, (chunks, -1, *A.shape[1:]))


def chunked_kernel_NM_dot_v(X: Num[Array, "N d"] = None, Y: Num[Array, "M d"] = None, v_l: Num[Array, "M"] = None, kernel=None, n_chunks=1):
    N = X.shape[0]
    w = jnp.zeros([n_chunks, N // n_chunks, *v_l.shape[1:]])
    c_X = chunk(X, n_chunks)

    def inner(i, w):
        Kr = kernel(c_X[i], Y)
        w = w.at[i].set(Kr @ v_l)
        return w
    w = jax.lax.fori_loop(0, n_chunks, inner, w)
    return w.reshape(N, *w.shape[2:])
