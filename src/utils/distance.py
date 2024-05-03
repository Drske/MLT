import numpy as np


def parameters_distance(pre: dict, post: dict, metric: str = "euclidean", p: float = 1):
    if pre.keys() != post.keys():
        raise ValueError("Pre & post parameters are different!")

    distance = 0.0
    for name, pre_params in pre.items():
        post_params = post[name]
        pre_params, post_params = (
            pre_params.cpu().numpy().flatten(),
            post_params.cpu().numpy().flatten(),
        )

        if "running_mean" in name:
            continue
        if "running_var" in name:
            continue
        if "num_batches_tracked" in name:
            continue

        match metric:
            case "euclidean":
                distance += np.linalg.norm(pre_params - post_params)
            case "cosine":
                dot_product = np.dot(pre_params, post_params)
                norm_product = np.linalg.norm(pre_params) * np.linalg.norm(post_params)
                distance += dot_product / norm_product
            case "manhattan":
                distance += np.sum(np.abs(pre_params - post_params))
            case "minkowski":
                distance += np.power(np.sum(np.power(np.abs(pre_params - post_params), p)), 1 / p)
            case "chebyshev":
                distance += np.max(np.abs(pre_params - post_params))
            case "correlation":
                correlation = np.corrcoef(pre_params.flatten(), post_params.flatten())[0, 1]
                distance += 1 - correlation
            case _:
                raise ValueError(f"Metric {metric} is not a valid metric.")

    return distance
