"""

"""
import asyncio
import concurrent.futures
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys
import time

# import fire
import rich
import rouge_score


SCRIPT_DIR = Path(__file__).resolve().parent

###############################################################################
# General Utils
###############################################################################
def check_exists(path):
    assert path.exists(), path
    return path


def check_all_exist(paths):
    outputs = []
    indices_dont_exist = []
    
    for i, path in enumerate(paths):
        outputs.append(path)
        if not path.exists():
            indices_dont_exist.append(i)

    assert not indices_dont_exist, [
        f"{i}: {indices_dont_exist[i]}" for i in indices_dont_exist
    ]

    return outputs


def check_is_truthy(obj):
    assert obj, obj
    return obj


def extract_number(path):
        matches = re.match(r"val_predictions-(\w+).txt", path.name)
        number = int(matches.group(1))
        return number

def model_output_paths_and_targets(directory):
    model_output_paths = check_is_truthy(
        check_all_exist(directory.glob("val_predictions*.txt"))
    )
    model_output_paths.sort(key=extract_number, reverse=True)
    targets = check_exists(directory / "val_targets.txt")
    return model_output_paths, targets


def prep_command(model_output_path, directory, targets):
    number = extract_number(model_output_path)
    assert isinstance(number, int), type(number)
    our_output = directory / f"rouge_{number}.txt"
    if not our_output.exists():
        rich.print(f"[yellow]Stacking {our_output}")
        command_flags = [
            f"--target_filepattern={shlex.quote(str(targets))}",
            f"--prediction_filepattern={shlex.quote(str(model_output_path))}",
            "--use_stemmer=True",
            f"--output_filename={shlex.quote(str(our_output))}",
        ]
        return command_flags, our_output
    return None, None
        

###############################################################################
# Thread specific
###############################################################################
def action_threads(our_output, command_flags):
    rich.print(f"[green]Starting {our_output}")
    subprocess.check_call([
        "python", "-m", "rouge_score.rouge"
    ] + command_flags)
    rich.print(f"[blue]Done with {our_output}")
    

def main_threads(directory=SCRIPT_DIR / "outputs"):
    directory = Path(directory)
    model_output_paths, targets = model_output_paths_and_targets(directory)

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for model_output_path in model_output_paths:
            maybe_command_flags, our_output = prep_command(
                model_output_path, directory, targets
            )
            if maybe_command_flags:
                pool.submit(
                    action_threads, 
                    our_output, 
                    maybe_command_flags,
                )
        for future in futures:
            future.wait()
    rich.print("[green bold]All done!")
                

###############################################################################
# Coroutine Specific
###############################################################################
async def run(cmd):
    proc = await asyncio.create_subprocess_shell(
        shlex.join(cmd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()
    return stdout.decode(), stderr.decode()


async def action_asyncio(sem, our_output, command_flags):
    rich.print(f"[yellow]Stacking {our_output}")
    async with sem:
        rich.print(f"[green]Starting {our_output}")
        await run([
                "python", "-m", "rouge_score.rouge"
        ] + command_flags)
        rich.print(f"[blue]Done with {our_output}")


async def main_asyncio(directory=SCRIPT_DIR / "outputs"):
    directory = Path(directory)
    model_output_paths, targets = model_output_paths_and_targets(directory)
    sem = asyncio.Semaphore(os.cpu_count())

    tasks = []
    for model_output_path in model_output_paths:
        maybe_command_flags, our_output = prep_command(
                model_output_path, 
                directory,
                targets,
            )
        if maybe_command_flags:
            tasks.append(asyncio.create_task(
                action_asyncio(
                    sem, 
                    our_output, 
                    maybe_command_flags,
                ))
            )
        
    for task in tasks:
        await task
    rich.print("[green bold]All done.")


if __name__ == "__main__":
    start = time.monotonic()
    main_threads()
    # asyncio.run(main_asyncio())
    rich.print(f"[bold red]Total time: {time.monotonic() - start}")
