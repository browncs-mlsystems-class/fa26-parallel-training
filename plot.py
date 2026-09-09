import os, sys
import math
import argparse
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tabulate

import utils
import re
 
 
def draw_column_plot(
    x_data: list[str], y_datas: list[list[list[float]]], title: str, 
    y_labels: list[list[str]] | None, x_label: str, y_label: str, 
    output_file: str | None = None
):
    """
    Draws a column plot for multiple segmented datasets, with one column per 
    dataset that is split into groupings based on the dataset's segmentation.
    
    Args:
        x_data (list[str]): the x-axis categories, one for each column
        y_datas (list[list[list[float]]]): a list of datasets, where
        each element of the list is a segmented dataset, and each element of 
        that is a segment of the dataset, with one value for each column (i.e.
        the inner list has a length equal to that of `x_data`)
        title (str): the title of the plot
        y_labels (list[list[str]] | None): the optional labels for each segment 
        of the dataset, or None if not applicable
        x_label (str): the label for the x-axis
        y_label (str): the label for the y-axis
        output_file (str | None): the output file to output the plot to; 
        otherwise displayed to the user via stdout
    """
    if output_file is not None:
        # create plot and save to file
        bar_width = 0.4
        _, ax = plt.subplots(figsize=(8, 5))

        indices = np.arange(len(x_data))
        bar_shift = (1 - len(y_datas)) / 2
        for i, y_data in enumerate(y_datas):
            prev_y = np.zeros(len(x_data))
            for j, y in enumerate(y_data):
                y += [0] * (len(x_data) - len(y))
                ax.bar(indices + (i + bar_shift) * bar_width, y, bar_width, 
                    label=y_labels[i][j] if y_labels else None, alpha=0.7, bottom=prev_y)
                prev_y += np.array(y)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(indices)
        ax.set_xticklabels(x_data)
        if y_labels is not None:
            ax.legend()
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        
        if os.path.dirname(output_file) != '':
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
    else:
        # display to stdout
        for i, dataset in enumerate(y_datas):
            table = [([""] if y_labels is not None else []) + x_data]
            assert (y_labels is None) == (len(dataset) == 1), "Y labels should be provided if bar is split into categories!"
            for j, data_row in enumerate(dataset):
                row = ([y_labels[i][j]] if y_labels is not None else [y_label]) + [f"{datum:.3f}" for datum in data_row]
                table.append(row)
            if len(dataset) != 1: # add summary row, if split
                row = ["Total " + y_label] + [f"{sum(datums):.3f}" for datums in zip(*dataset)]
                table.append(row)
            if i != 0:
                print()
            print(tabulate.tabulate(table, headers="firstrow", numalign="center", stralign="center", tablefmt="grid"))

def plot_timeline(timelines: list[list[float]], output_file: str | None = None):
    """
    Plots a timeline of the work of each worker process for model and pipeline
    parallelism.

    Args:
        timelines (list[list[float]]): a list containing for each worker, by
        ascending rank, the list of (unnormalized) times that the worker
        started/ended a forward and backward pass
        output_file (str | None): the output file to output the plot to;
        otherwise displayed to the user via stdout
    """
    rectangles: list[tuple[float, float, float, str, str]] = []
    TIME_MIN, TIME_MAX = math.inf, -math.inf
    for rank, times in enumerate(timelines):
        assert len(times) % 4 == 0  # should contain start/end time of each forward and backward pass
        for i in range(0, len(times), 2):
            start, end = times[i], times[i+1]
            TIME_MIN = min(start, TIME_MIN)
            TIME_MAX = max(end, TIME_MAX)
            rectangles.append((
                start, end, rank,
                'tab:green' if i < len(times) / 2 else 'tab:blue',
                'g' if i < len(times) / 2 else 'b',
            ))

    # auto-pick a unit (s / ms / µs / ns) based on the actual span of the data
    time_span = TIME_MAX - TIME_MIN
    if time_span >= 1:
        scale, unit = 1, "s"
    elif time_span >= 1e-3:
        scale, unit = 1e3, "ms"
    elif time_span >= 1e-6:
        scale, unit = 1e6, "\u00b5s"
    else:
        scale, unit = 1e9, "ns"

    scaled_span = time_span * scale

    _, ax = plt.subplots()
    for x_min, x_max, rank, facecolor, edgecolor in rectangles:
        x_start = (x_min - TIME_MIN) * scale
        width = (x_max - x_min) * scale
        height = 1
        ax.add_patch(Rectangle(
            (x_start, rank - 0.5), width, height,
            alpha=0.5, facecolor=facecolor, edgecolor=edgecolor
        ))
        ax.text(x_start + width / 2, rank, f"{width:.3g}{unit}",
                color='black', ha='center', va='center', rotation=90)

    if output_file is not None:
        # create plot and save to file
        padding = scaled_span * 0.05 if scaled_span > 0 else 1  # 5% padding on each side, scales with your data
        ax.set_xlim(0 - padding, scaled_span + padding)
        ax.set_xticks(np.linspace(0, scaled_span, 10))
        ax.set_yticks(range(len(timelines)), [f"Rank {rank}" for rank in range(len(timelines))])
        ax.set_ylim(-0.5, len(timelines) - 0.5)
        ax.set_xlabel(f"Time ({unit})")
        ax.set_ylabel("Workers")
        plt.tight_layout()

        if os.path.dirname(output_file) != '':
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
    else:
        # display to stdout
        num_workers = len(timelines)
        num_microbatches = len(timelines[0]) // 4
        headers = [f"Forward pass\nWorker {i}" for i in range(1, num_workers+1)] + [f"Backward pass\nWorker {i}" for i in range(num_workers, 0, -1)]
        if num_microbatches != 1:
            headers = [""] + headers  # add headers for row for each microbatch for pipeline parallel

        table = [headers]
        for i in range(num_microbatches):
            row = [] if num_microbatches == 1 else [f"Microbatch {i+1}"]
            row += [f"{start-TIME_MIN:.3f}-{end-TIME_MIN:.3f}" for start, end, _, _, _ in rectangles[i::num_microbatches*2]]
            row += reversed([f"{start-TIME_MIN:.3f}-{end-TIME_MIN:.3f}" for start, end, _, _, _ in rectangles[i+num_microbatches::num_microbatches*2]])
            table.append(row)
        print(tabulate.tabulate(table, headers="firstrow", numalign="center", stralign="center", tablefmt="grid"))

def plot_memory_over_time(
    logs_by_rank: dict[int, list[dict]],
    out_path: str = "memory_plot.png",
    bytes_field: str = "allocated_bytes",
    title: str = "Active GPU memory over time",
):
    """
    Plots active memory vs. time (normalized to the earliest timestamp
    across all ranks), one line per rank.
 
    Args:
        logs_by_rank (dict[int, list[dict]]): output of load_logs
        out_path (str): file path to save the resulting plot image
        bytes_field (str): which byte field to plot, e.g. "allocated_bytes"
        or "peak_bytes"
        title (str): plot title
    """
    if not logs_by_rank:
        raise ValueError("No log entries found - check log_dir path")
 
    # normalize time to the earliest timestamp seen across all ranks, so
    # t=0 lines up across ranks regardless of process start-time skew
    global_min_time = min(
        entry["timestamp"] for entries in logs_by_rank.values() for entry in entries
    )
 
    fig, ax = plt.subplots(figsize=(12, 6))

    for rank in sorted(logs_by_rank):
        entries = logs_by_rank[rank]
        times = [e["timestamp"] - global_min_time for e in entries]
        mem_mb = [e[bytes_field] / (1024 ** 2) for e in entries]
        ax.plot(times, mem_mb, marker="o", markersize=3, linewidth=1, label=f"rank {rank}")
 
    ax.set_xlabel("Time (s, normalized to first logged event)")
    ax.set_ylabel(f"{bytes_field.replace('_', ' ').title()} (MB)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
 
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

def load_memory_logs(parallelism: str, num_workers: int | None) -> dict:
    """
    Loads the memory logs for a ddp or fsdp run, optionally narrowd down to one
    with a certain number of orkers.
    Args:
        parallelism (str): the parallelism technique ran
        num_workers (int | None): the number of workers used for a run of the
        given technique; otherwise loads the stored statistics for the run with
        the lowest number of workers
    Returns:
        dict[int, list[dict]]: a mapping from rank -> that rank's list of log entries, sorted by timestamp
    """
    logs_by_rank: dict[int, list[dict]] = {}
    log_dir = os.path.join(utils.DATA_DIR, "memory_logs")
 
    # matches e.g. "fsdp4_rank0_memory_log.pt" -> world_size=4, rank=0
    pattern = re.compile(rf"^{re.escape(parallelism)}(\d+)_rank(\d+)_memory_log\.pt$")
 
    # first pass: find all matching files and figure out which world_size to use
    matches: list[tuple[int, int, str]] = []  # (world_size, rank, filepath)
    for filepath in glob.glob(os.path.join(log_dir, f"{parallelism}*_memory_log.pt")):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match is None:
            continue
        found_world_size, found_rank = int(match.group(1)), int(match.group(2))
        matches.append((found_world_size, found_rank, filepath))
 
    if not matches:
        raise ValueError(
            f"No memory log files found for parallelism='{parallelism}' in {log_dir}"
        )
 
    if num_workers is None:
        num_workers = min(world_size for world_size, _, _ in matches)
 
    for world_size, rank, filepath in matches:
        if world_size != num_workers:
            continue
        with open(filepath) as f:
            entries = json.load(f)
        logs_by_rank[rank] = sorted(entries, key=lambda e: e["timestamp"])
 
    if not logs_by_rank:
        raise ValueError(
            f"No memory log files found for parallelism='{parallelism}' "
            f"with num_workers={num_workers} in {log_dir}"
        )
 
    return logs_by_rank


def load_stats(parallelism: str, num_workers: int | None) -> dict:
    """
    Loads the stored statistics for a parallelism run, optionally narrowed down
    to one with a certain number of workers.
    
    Args:
        parallelism (str): the parallelism technique ran
        num_workers (int | None): the number of workers used for a run of the
        given technique; otherwise loads the stored statistics for the run with
        the lowest number of workers
    Returns:
        dict: the stored statistics for the given run
    """
    if num_workers is not None:
        file_path = os.path.join(utils.DATA_DIR, f"stats_{parallelism}{num_workers}.json")
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print("ERROR: could not find the stored statistics for a run with "
                  f"parallelism technique \"{parallelism}\" with {num_workers} workers")
            sys.exit(1)
    else:
        file_paths = sorted(glob.glob(os.path.join(utils.DATA_DIR, f"stats_{parallelism}*.json")))
        if len(file_paths) == 0:
            print("ERROR: could not find any stored statistics for a run with "
                  f"parallelism technique \"{parallelism}\"")
            sys.exit(1)
        with open(file_paths[0]) as f:
            return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS1390 Project 1: Parallelism Techniques Plotter")
    parser.add_argument(
        "plot", type=str,
        choices=["throughput", "split_time", "timeline", "memory"],
        help="Type of plot to generate"
    )
    parser.add_argument(
        "parallelism", type=str,
        help="The parallelism techniques whose runs are to be analyzed, of the form "
             "\"<tech1>[=<num_workers>],<tech2>[=<num_workers>],...\". Each element "
             "of the comma-separated list is a parallelism technique (\"ddp\", "
             "\"fsdp\", \"model\", or \"pipeline\"), with an optional specification "
             "a run with a certain number of workers used for that technique. Example "
             "arguments include: \"ddp,model\", \"ddp=2,ddp=4\", \"pipeline\", and "
             "\"model,pipeline=4\""
    )
    parser.add_argument(
        "--output", "-o", type=str, 
        help="The (optional) comma-separated output paths for the generated plot(s). "
        "If the plot type is \"split_time\" or \"throughput\", only one output path "
        "should be specified. For all others, there should be one output path for "
        "each specified parallelism technique to analyze")
    args = parser.parse_args()
    
    # Parse and validate arguments
    VALID_TECHS = {"ddp", "fsdp", "model", "pipeline"}
    techs_dict: dict[str, list[int] | None] = {}
    techs_to_analyze: list[tuple[str, int | None]] = []
    for tech in args.parallelism.split(","):
        if tech.count("=") != 1: # no number of workers specified
            if tech not in VALID_TECHS:
                print(f"ERROR: Invalid parallelism technique: {tech}")
                sys.exit(1)
            if tech in techs_to_analyze:
                print("ERROR: Cannot distinguish between two specified instances "
                      f"of parallelism technique {tech}, number of workers must "
                      "be specified for both instances")
                sys.exit(1)
            techs_dict[tech] = None
            techs_to_analyze.append((tech, None))
        elif tech.count("=") == 1:
            tech, num_workers = tech.split("=")
            if tech not in VALID_TECHS:
                print(f"ERROR: Invalid parallelism technique: {tech}")
                sys.exit(1)
            try:
                num_workers = int(num_workers)
            except ValueError:
                print(f"ERROR: Invalid number of workers: {num_workers}")
                sys.exit(1)
            if tech in techs_to_analyze:
                if techs_to_analyze[tech] is None:
                    print("ERROR: Cannot distinguish between two specified instances "
                          f"of parallelism technique {tech}, number of workers must "
                          "be specified for both instances")
                    sys.exit(1)
                techs_dict[tech].append(num_workers)
            else:
                techs_dict[tech] = [num_workers]
            techs_to_analyze.append((tech, num_workers))
    
    output_files = args.output.split(",") if args.output is not None else None
    if args.plot == "split_time" or args.plot == "throughput":
        if args.plot == "split_time" and list(techs_dict.keys()) != ["ddp"]:
            print("ERROR: cannot generate comparison of split time for any technique besides DDP")
            sys.exit(1)
        if output_files is not None and len(output_files) != 1:
            print(f"ERROR: expected 1 output file for {args.plot.replace('_', ' ')} plot, received {len(output_files)}")
            sys.exit(1)
    elif args.plot == "timeline": # timeline plot
        if output_files is not None and len(output_files) != len(techs_to_analyze):
            print(f"ERROR: expected {len(techs_to_analyze)} output files for {args.plot} plot, received {len(output_files)}")
            sys.exit(1)
        
        if "ddp" in techs_dict or "fsdp" in techs_dict:
            print(f"ERROR: can only generate a timeline plot for model and pipeline parallelism runs")
            sys.exit(1)
    elif args.plot == "memory":
        if output_files is not None and len(output_files) != len(techs_to_analyze):
            print(f"ERROR: expected {len(techs_to_analyze)} output files for {args.plot} plot, received {len(output_files)}")
            sys.exit(1)
        if "model" in techs_dict or "pipeline" in techs_dict:
            print(f"ERROR: can only generate a memory plot for fsdp and ddp parallelism runs")
            sys.exit(1)

                                            
    # Generate plots
    print("Generating plot(s)...")
    all_stats = [load_stats(*techs) for techs in techs_to_analyze]
    if args.plot == "split_time":
        output_file = output_files[0] if output_files else None
        max_workers = max(map(lambda stats: stats[utils.NUM_WORKERS], all_stats))
        y_values = []
        y_labels = []
        for stats in all_stats:
            num_workers: int = stats[utils.NUM_WORKERS]
            comp_times = [stats[str(rank)][utils.COMP_TIME] + stats[str(rank)][utils.OPT_TIME] 
                          for rank in range(num_workers)]
            comm_times = [stats[str(rank)][utils.COMM_TIME] for rank in range(num_workers)]
            y_values.append([comp_times, comm_times])
            y_labels.append([f"Computation time (W={num_workers})", f"Communication time (W={num_workers})"])

        draw_column_plot(
            [f"Rank {rank}" for rank in range(max_workers)], y_values, 
            'Computation vs. Communication Time per Worker for DDP', 
            y_labels, "Worker Rank", "Time (s)", output_file
        ) 
    elif args.plot == "throughput":
        output_file = output_files[0] if output_files else None
        parallel_to_display = {'ddp': 'DDP', 'fsdp': 'FSDP', 'model': 'Model Parallelism', 
                               'pipeline': 'Pipeline Parallelism'}
        y_values = []
        for stats, (p, _) in zip(all_stats, techs_to_analyze):
            num_workers = stats[utils.NUM_WORKERS]
            items_processed = stats[utils.BATCH_SIZE] * stats[utils.NUM_BATCHES] * (num_workers if p == 'ddp' or p == 'fsdp' else 1)
            time_taken = max([stats[str(rank)][utils.TOTAL_TIME] for rank in range(num_workers)])
            y_values.append(items_processed / time_taken)
        draw_column_plot(
            [f"{parallel_to_display[p]} (W={w})" for p, w in techs_to_analyze],
            [[y_values]], '', None, "Parallelism Techniques", 
            "Throughput (items/s)", output_file
        )
    elif args.plot == "timeline":
        output_files = output_files if output_files is not None else ([None] * len(techs_to_analyze))
        for stats, output_file in zip(all_stats, output_files):
            timelines = [] # contains the timeline for each worker, by ascending rank
            for rank in range(stats[utils.NUM_WORKERS]):
                batch_times = stats[str(rank)][utils.BATCHES_TIMES]
                # limit processing times to only that of first batch
                print(f"rank {rank} Dividing array by {stats[utils.NUM_BATCHES]}; originally {len(batch_times)}")
                timelines.append(batch_times[:len(batch_times) // stats[utils.NUM_BATCHES]])
            plot_timeline(timelines, output_file)

    elif args.plot == "memory":
        all_memory_logs = [load_memory_logs(*techs) for techs in techs_to_analyze]
        output_files = output_files if output_files is not None else ([None] * len(techs_to_analyze))
        for memory_log, output_file in zip(all_memory_logs, output_files):
            plot_memory_over_time(memory_log, output_file)
