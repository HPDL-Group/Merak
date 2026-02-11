#!/usr/bin/env python
"""
测试启动脚本 - 支持一键测试
支持多种测试框架和运行方式
"""
import subprocess
import sys
import os
from pathlib import Path


def run_tests(test_path=None, verbose=True, cov=False):
    """
    运行测试

    Args:
        test_path: 测试路径，可以是文件、目录或测试用例名称
        verbose: 是否详细输出
        cov: 是否生成覆盖率报告
    """
    project_root = Path(__file__).parent.parent

    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.extend(["-v", "--tb=short"])

    if cov:
        cmd.extend(["--cov=procguard", "--cov-report=term-missing"])

    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    print(f"运行命令: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=str(project_root))

    return result.returncode


def run_unit_tests():
    """运行单元测试"""
    return run_tests()


def run_integration_tests():
    """运行集成测试"""
    test_path = Path(__file__).parent / "tests" / "test_*.py"
    return run_tests(str(test_path))


def run_specific_test(test_file):
    """运行特定测试文件"""
    test_path = Path(__file__).parent / "tests" / test_file
    if test_path.exists():
        return run_tests(str(test_path))
    else:
        print(f"测试文件不存在: {test_path}")
        return 1


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ProcGuard 测试启动脚本")
    parser.add_argument("test_path", nargs="?", default=None, help="测试路径（可选）")
    parser.add_argument("--cov", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    returncode = run_tests(test_path=args.test_path, verbose=not args.quiet, cov=args.cov)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
