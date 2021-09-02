"""

"""
import concurrent.futures
from pathlib import Path
import re
import shlex
import subprocess
import sys

import fire
import rich
import rouge_score


SCRIPT_DIR = Path(__file__).resolve().parent


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

def action(our_output, command_flags):
    rich.print(f"[green]Starting {our_output}")
    p = subprocess.Popen([
        "python", "-m", "rouge_score.rouge"
    ] + command_flags)
    p.wait()
    rich.print(f"[blue]Done with {our_output}")
    

def main(directory=SCRIPT_DIR/"outputs"):
    directory = Path(directory)
    model_output_paths = check_is_truthy(
        check_all_exist(directory.glob("val_predictions*.txt"))
    )
    model_output_paths.sort(key=extract_number, reverse=True)
    targets = check_exists(directory / "val_targets.txt")

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for model_output_path in model_output_paths:
            number = extract_number(model_output_path)
            assert isinstance(number, int), type(number)
            our_output = Path(SCRIPT_DIR) / f"rouge_{number}.txt"
            if not our_output.exists():
                rich.print(f"[yellow]Stacking {our_output}")
                command_flags = [
                    f"--target_filepattern={shlex.quote(str(targets))}",
                    f"--prediction_filepattern={shlex.quote(str(model_output_path))}",
                    "--use_stemmer=True",
                    f"--output_filename={shlex.quote(str(our_output))}",
                ]
                
                pool.submit(action, our_output, command_flags)
        for future in futures:
            future.wait()
    rich.print("[green bold]All done!")
                
        
            
    

if __name__ == "__main__":
    fire.Fire(main)
