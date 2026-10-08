import torch
from mcoplib.op import silu_and_mul_clamp
from mcoplib.profiler import profiler

def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))

# @profiler(output_dir="./profiles", warmup=2, repeat=3)
def silu_mul_clamp_test(
    M: int,
    hidden_size_list: list[int],
    swiglu_limit: float | None = None,
    dtype: torch.dtype = torch.bfloat16,
    elem_bytes: int = 2,
    device: str | torch.device = "cuda"
):
    """
    CUDA版本，对齐C++ silu_mul_clamp_kernel，交替Vec布局 gate0,up0,gate1,up1...
    :param M: token数量
    :param hidden_size_list: list，支持 [16,128,8192]
    :param swiglu_limit: 截断上限
    :param dtype: torch.bfloat16 / torch.float32
    :param elem_bytes: bf16=2, float32=4
    :param device: cuda / cuda:0
    :return: dict[hidden_size] = (input, output)
    """
    assert torch.cuda.is_available(), "CUDA not available!"
    dev = torch.device(device)
    vec_bytes = 16
    vec_elem = vec_bytes // elem_bytes
    print(f">> vec_bytes={vec_bytes}, elem_bytes={elem_bytes}, vec_elem={vec_elem}, device={dev}")

    result_dict = {}
    for hidden_size in hidden_size_list:
        print(f"\n===== hidden_size = {hidden_size} =====")
        assert hidden_size % vec_elem == 0, f"hidden_size={hidden_size} must be multiple of {vec_elem}"
        num_vec_pairs_per_row = hidden_size // vec_elem
        D = 2 * hidden_size

        # 直接在CUDA上构造randn
        input_tensor = torch.randn((M, D), dtype=dtype, device=dev)
        out = torch.empty((M, hidden_size), dtype=dtype, device=dev)
        out_check = torch.empty((M, hidden_size), dtype=dtype, device=dev)

        x = input_tensor.reshape(M, num_vec_pairs_per_row, 2, vec_elem)
        gate = x[:, :, 0, :]   # [M, num_pairs, vec_elem]
        up   = x[:, :, 1, :]   # [M, num_pairs, vec_elem]

        g = gate.float()
        u = up.float()
        if swiglu_limit is not None:
            g = torch.clamp(g, max=swiglu_limit)
            u = torch.clamp(u, min=-swiglu_limit, max=swiglu_limit)

        s_g = silu(g)
        res = s_g * u
        res = res.to(dtype)

        # [M, num_pairs, vec_elem] -> [M, hidden_size]
        out = res.flatten(1, 2)

        silu_and_mul_clamp(input_tensor, out_check, swiglu_limit)
        all_close = torch.allclose(out, out_check, atol=1e-2, rtol=1e-2)
        mask = ~torch.isclose(out, out_check, atol=1e-3, rtol=1e-3)
        print("是否存在差异：", mask.any().item())
        print("差异元素总数：", mask.sum().item())
        assert all_close 
        print("done")


if __name__ == "__main__":
    hidden_sizes = [4096]
    list_M=[1024,2048,1024*3,1024*4,1024*5,1024*6,1024*7]
    for iter in list_M:
        silu_mul_clamp_test(
            M=iter,
            hidden_size_list=hidden_sizes,
            swiglu_limit=5.0,
            dtype=torch.bfloat16,
            elem_bytes=2,
            device="cuda:0"
        )