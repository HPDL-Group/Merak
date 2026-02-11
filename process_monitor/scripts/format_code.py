#!/usr/bin/env python
"""
代码格式化脚本
支持 black 格式化、isort 排序导入、pylint 检查
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False, cwd=cwd)
    return result.returncode


def format_code():
    """格式化代码"""
    project_root = Path(__file__).parent.parent

    print("开始代码格式化...")

    exit_code = 0

    # 1. 使用 isort 排序导入
    print("\n[1/3] 排序导入语句...")
    cmd = [sys.executable, "-m", "isort", "--profile=black", "."]
    if run_command(cmd, "排序导入语句 (isort)", cwd=str(project_root)) != 0:
        exit_code = 1

    # 2. 使用 black 格式化代码
    print("\n[2/3] 格式化代码...")
    cmd = [sys.executable, "-m", "black", "--line-length", "100", "."]
    if run_command(cmd, "格式化代码 (black)", cwd=str(project_root)) != 0:
        exit_code = 1

    # 3. 使用 pylint 检查代码
    print("\n[3/3] 代码质量检查...")
    cmd = [sys.executable, "-m", "pylint", "--rcfile=configs/.pylintrc", "procguard/", "tests/"]
    if run_command(cmd, "代码质量检查 (pylint)", cwd=str(project_root)) != 0:
        exit_code = 1

    return exit_code


def check_only():
    """仅检查代码格式，不修改"""
    project_root = Path(__file__).parent.parent

    print("仅检查代码格式...")

    exit_code = 0

    # 1. 检查导入排序
    print("\n[1/3] 检查导入排序...")
    cmd = [sys.executable, "-m", "isort", "--check-only", "--profile=black", "."]
    if run_command(cmd, "检查导入排序 (isort --check-only)", cwd=str(project_root)) != 0:
        exit_code = 1

    # 2. 检查代码格式
    print("\n[2/3] 检查代码格式...")
    cmd = [sys.executable, "-m", "black", "--check", "--line-length", "100", "."]
    if run_command(cmd, "检查代码格式 (black --check)", cwd=str(project_root)) != 0:
        exit_code = 1

    # 3. 检查代码质量
    print("\n[3/3] 代码质量检查...")
    cmd = [sys.executable, "-m", "pylint", "--rcfile=configs/.pylintrc", "procguard/", "tests/"]
    if run_command(cmd, "代码质量检查 (pylint)", cwd=str(project_root)) != 0:
        exit_code = 1

    return exit_code


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="代码格式化脚本")
    parser.add_argument("--check", action="store_true", help="仅检查代码格式，不修改")

    args = parser.parse_args()

    if args.check:
        return check_only()
    else:
        return format_code()


if __name__ == "__main__":
    sys.exit(main())
