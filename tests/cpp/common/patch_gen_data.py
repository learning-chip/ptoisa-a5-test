#!/usr/bin/env python3
"""Patch upstream-style gen_data.py for standalone test layout."""

import argparse
import re
import sys
from pathlib import Path

LOOP_TEMPLATE = '''
    smoke_indices = {smoke_indices}

    for i, case_name in enumerate(case_name_list):
        if not args.all and i not in smoke_indices:
            continue
        short_name = case_name.split(".")[-1] if "." in case_name else case_name
        case_dir = os.path.join(args.output_dir, short_name)
        os.makedirs(case_dir, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_dir)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
        print(f"Generated {{case_dir}}")
'''


def patch_file(path: Path, smoke_indices: list[int], prefix: str) -> None:
    text = path.read_text()
    text = text.replace(f"{prefix}.", "")

    if "parser = argparse.ArgumentParser" not in text:
        text = text.replace(
            'if __name__ == "__main__":',
            'if __name__ == "__main__":\n    import argparse\n\n'
            '    parser = argparse.ArgumentParser(description="Generate golden data")\n'
            '    parser.add_argument("--output-dir", default="cases", help="Output directory for case data")\n'
            '    parser.add_argument("--all", action="store_true", help="Generate all regression cases")\n'
            '    args = parser.parse_args()\n',
            1,
        )

    text = re.sub(
        r"\n    for i, case_name in enumerate\(case_name_list\):.*?os\.chdir\(original_dir\)\n",
        LOOP_TEMPLATE.format(smoke_indices=set(smoke_indices)),
        text,
        count=1,
        flags=re.DOTALL,
    )

    # tmax/tmul build case_name_list inside __main__ via generate_case_name
    if "case_name_list" not in text and "case_params_list" in text:
        insert = '''
    case_name_list = [generate_case_name(p) for p in case_params_list]
'''
        text = text.replace("    args = parser.parse_args()\n", "    args = parser.parse_args()\n" + insert, 1)

    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--prefix", default="TMOVTest")
    parser.add_argument("--smoke-index", type=int, action="append", dest="smoke_indices", required=True)
    args = parser.parse_args()
    patch_file(Path(args.file), args.smoke_indices, args.prefix)


if __name__ == "__main__":
    main()
