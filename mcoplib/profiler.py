# profiler_plus.py
import os
import functools
from datetime import datetime
import threading

def _is_profiler_enabled() -> bool:
    """
    Env switch: PROFILER_ENABLED==0 -> disabled, else enabled.
    """
    v = os.getenv("PROFILER_ENABLED", "1")
    try:
        return not (str(v).strip() == "0")
    except Exception:
        return True


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def _trace_file_path(output_dir, func_name, rank):
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in func_name)[:128]
    filename = (
        f"{safe_name}_trace_rank_{rank}_"
        f"{_timestamp()}_pid_{os.getpid()}_tid_{threading.get_ident()}.json"
    )
    return os.path.join(output_dir, filename)


def _track_handler(prof, output_dir, func_name):
    """
    Track handler implementation that matches test.py track_handler format.
    """
    try:
        # Print detailed profiler tables (matching test.py format)
        print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                      max_name_column_width=10000,
                                      max_src_column_width=10000,
                                      row_limit=-1))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total",
                                                                max_src_column_width=10000,
                                                                max_name_column_width=10000,
                                                                row_limit=-1))
        print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                      max_src_column_width=10000,
                                      row_limit=-1))

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Safe rank value acquisition (supports both distributed and non-distributed scenarios)
        try:
            import torch
            rank = torch.distributed.get_rank()
        except Exception:
            # If distributed environment is not initialized, use default value 0
            rank = 0

        # Export trace to a unique file so repeated benchmark runs do not overwrite evidence.
        trace_path = _trace_file_path(output_dir, func_name, rank)
        prof.export_chrome_trace(trace_path)
        print(f"Chrome trace exported to: {trace_path}")

    except Exception as e:
        print(f"Warning: Failed to export trace: {e}")
        # Continue execution, don't let trace export failure affect main process


def profiler(
    _func=None,
    *,
    output_dir="./profiles",
    warmup=1,
    repeat=1,
):
    """
    Simplified profiler decorator based on test.py implementation.

    Usage:
      @profiler
      def f(...): ...

    or with options:
      @profiler(output_dir='./prof', warmup=2, repeat=3)
      def f(...): ...

    Parameters:
      output_dir: directory for chrome trace files
      warmup: number of warmup calls (not profiled)
      repeat: number of times to call function inside a profiler run
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _is_profiler_enabled():
                # disabled by env
                return func(*args, **kwargs)

            # Try to use torch.profiler
            try:
                import torch
                from torch.profiler import profile, ProfilerActivity
                has_torch = True
            except Exception:
                has_torch = False

            # Warm-up (no profiling) to stabilize JIT / caches
            try:
                for _ in range(max(0, int(warmup))):
                    func(*args, **kwargs)
            except Exception:
                # keep raising actual function exceptions
                raise

            # If torch and torch.profiler available -> prof mode
            if has_torch:
                try:
                    activities = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(ProfilerActivity.CUDA)

                    # Use profile with schedule similar to test.py
                    with profile(
                        activities=activities,
                        schedule=torch.profiler.schedule(
                            wait=0,
                            warmup=warmup,
                            active=1,
                            repeat=repeat
                        ),
                        on_trace_ready=lambda prof: _track_handler(prof, output_dir, func.__name__),
                        with_modules=True,
                        record_shapes=True,
                        profile_memory=True
                    ) as prof:

                        # Run the function the specified number of times
                        result = None
                        for _ in range(max(1, int(repeat))):
                            result = func(*args, **kwargs)
                            prof.step()  # Step the profiler

                        # Ensure CUDA completion
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                    return result

                except Exception as e:
                    print(f"torch.profiler raised exception: {e}")
                    # Fall back to direct function call
                    return func(*args, **kwargs)

            # Fallback: just run the function directly
            return func(*args, **kwargs)

        return wrapper

    # Support both @profiler and @profiler(...)
    if _func is None:
        return decorator
    else:
        return decorator(_func)
