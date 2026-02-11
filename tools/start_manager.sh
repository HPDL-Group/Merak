# ProcGuard 管理服务端启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认配置
CONFIG_FILE="${SCRIPT_DIR}/../process_monitor/configs/procguard_config.yaml"
LOG_FILE="${SCRIPT_DIR}/logs/procguard.log"

current_dir=`pwd`
export PYTHONPATH=$PYTHONPATH:$current_dir/process_monitor

# 创建日志目录
mkdir -p "${SCRIPT_DIR}/logs"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --daemon)
            DAEMON=true
            shift
            ;;
        --coupled)
            COUPLED=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config <文件>  配置文件路径 (默认: configs/procguard_config.yaml)"
            echo "  --daemon         作为守护进程运行"
            echo "  --coupled        使用耦合模式 (直接管理本地worker，不推荐)"
            echo "  --help, -h       显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 启动 ProcGuard (默认使用解耦模式)
echo "启动 ProcGuard 管理服务 (解耦模式)..."
echo "配置文件: $CONFIG_FILE"
echo "日志文件: $LOG_FILE"
echo ""
echo "说明: 在解耦模式下，管理服务不会直接运行 worker。"
echo "      请使用 worker_launcher.py 或 start_worker.sh 启动 worker。"
echo ""

cd "$SCRIPT_DIR"

if [ "$DAEMON" = true ]; then
    if [ "$COUPLED" = true ]; then
        nohup python -m procguard --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    else
        nohup python -m procguard --config "$CONFIG_FILE" --decoupled > "$LOG_FILE" 2>&1 &
    fi
    echo "ProcGuard 已作为守护进程启动 (PID: $!)"
else
    if [ "$COUPLED" = true ]; then
        python -m procguard --config "$CONFIG_FILE"
    else
        python -m procguard --config "$CONFIG_FILE" --decoupled
    fi
fi
