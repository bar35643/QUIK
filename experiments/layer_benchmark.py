import torch
from qlinear import MixedQLinear, Linear8bit, Linear4bit
import time
import argparse
import numpy as np


fp_features_num = 256
model_sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (16384, 16384)]


def benchmark(args):
    global model_sizes
    #input_size = args.input_size
    for dtype in [torch.float16, torch.bfloat16]:
        for input_size in [256, 512, 1024, 2048, 4096]:
            for (feature_dim_in, feature_dim_out) in model_sizes:
                x = torch.rand((input_size, feature_dim_in)).cuda().to(dtype)
                def run_benchmark(module):
                    num_bench_steps = 100
                    for i in range(10):
                        out = module(x)
                    start_time = time.perf_counter()
                    torch.cuda.synchronize()
                    if args.profile:
                        torch.cuda.cudart().cudaProfilerStart()
                    for i in range(num_bench_steps):
                        out = module(x)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    if args.profile:
                        torch.cuda.cudart().cudaProfilerStop()
                    return (end_time - start_time) * 1000 / num_bench_steps
                baseline_mod = torch.nn.Linear(feature_dim_in, feature_dim_out, bias=False).cuda().to(dtype)
                baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data, low=-8, high=7).to(dtype)
                fp_indices = torch.randperm(feature_dim_in)[:fp_features_num]
                s_w = torch.ones((feature_dim_out, 1), dtype=dtype, device='cuda')
                int4_mod = MixedQLinear.from_float(baseline_mod,
                                                baseline_mod.weight.data,
                                                s_w, shared_input=None,
                                                fp_indices=fp_indices, bits=4).cuda()
                int8_mod = MixedQLinear.from_float(baseline_mod,
                                                baseline_mod.weight.data,
                                                s_w, shared_input=None,
                                                fp_indices=None, bits=8).cuda()

                int4_optim = Linear4bit(feature_dim_in, feature_dim_out).cuda()
                int8_optim = Linear8bit(feature_dim_in, feature_dim_out).cuda()

                print(f"{dtype}. Sizes: {input_size}{baseline_mod.weight.shape}")
                baseline_mod_times = [run_benchmark(baseline_mod) for i in range(10)]
                int4_mod_times = [run_benchmark(int4_mod) for i in range(10)]
                int4_optim_times = [run_benchmark(int4_optim) for i in range(10)]          
                int8_mod_times = [run_benchmark(int8_mod) for i in range(10)]
                int8_optim_times = [run_benchmark(int8_optim) for i in range(10)]
    
                print(#f"Int4 time: {np.mean(int4_mod_times):.3f} +- {1.96 * np.std(int4_mod_times):.3f}ms" + "---" +
                      #f"Int4 Optim time: {np.mean(int4_optim_times):.3f} +- {1.96 * np.std(int4_optim_times):.3f}ms" + "---" +
                      f"Int8 time: {np.mean(int8_mod_times):.3f} +- {1.96 * np.std(int8_mod_times):.3f}ms" + "---" +
                      f"Int8 Optim time: {np.mean(int8_optim_times):.3f} +- {1.96 * np.std(int8_optim_times):.3f}ms" + "---" +
                      f"{dtype} time: {np.mean(baseline_mod_times):.3f} +- {1.96 * np.std(baseline_mod_times):.3f}ms"
                    )
                
        print("")
    print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-size', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--profile', help='Do profile',
        action='store_true',
    )
    args = parser.parse_args()
    benchmark(args)
