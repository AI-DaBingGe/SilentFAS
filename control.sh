#!/bin/bash
# ---------------------------------------------------------
# SilentFAS CentOS 7 控制脚本
# 包含功能: 启动 (start)、停止 (stop)、重启 (restart)
# ---------------------------------------------------------

# 获取当前脚本所在目录
APP_DIR=$(cd $(dirname $0); pwd)
cd $APP_DIR

PID_FILE="$APP_DIR/silentfas.pid"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/silentfas.log"
PORT=8000

# 日志配置
LOG_KEEP_DAYS=3
LOG_MAX_SIZE="500M"

mkdir -p "$LOG_DIR"

start() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo "服务已在运行 (PID: $PID)"
            return
        fi
    fi
    
    echo "正在启动 SilentFAS 服务 (端口: $PORT)..."
    
    # 1. 主动清理过期日志 (满足只保留3天的要求)
    # 使用 find 查找 3 天前的以 silentfas.log 开头的文件并删除
    find "$LOG_DIR" -name "silentfas.log*" -type f -mtime +$LOG_KEEP_DAYS -exec rm -f {} \;
    
    # 2. 生成 Linux logrotate 配置 (满足单文件不超过 500M 的要求)
    setup_logrotate
    
    # 2.5 自动检查并安装依赖包
    echo "正在检查 Python 依赖包..."
    if ! python3 -c "import uvicorn, fastapi, cv2, onnxruntime" &> /dev/null; then
        echo "检测到核心依赖缺失，正在为您自动安装 (使用阿里云镜像)..."
        python3 -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    fi
    
    # 3. 启动应用
    # 使用 python3 -m uvicorn 替代直接调用 uvicorn，防止因环境变量路径问题导致“找不到命令”
    nohup python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    echo "服务已启动 (PID: $PID)，日志文件: $LOG_FILE"
    echo "提示: 可以通过 tail -f $LOG_FILE 查看实时日志。"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "正在停止服务 (PID: $PID)..."
        kill $PID 2>/dev/null
        rm -f "$PID_FILE"
        echo "服务已停止。"
    else
        echo "服务未运行或找不到 PID 文件 ($PID_FILE)。"
        # 尝试暴力清理
        pkill -f "uvicorn main:app --host 0.0.0.0 --port $PORT"
    fi
}

restart() {
    stop
    sleep 2
    start
}

setup_logrotate() {
    CONF_FILE="$APP_DIR/silentfas_logrotate.conf"
    
    # 动态生成 logrotate 配置文件
    cat > "$CONF_FILE" <<EOF
$LOG_FILE {
    size $LOG_MAX_SIZE
    missingok
    copytruncate
    rotate 50
    notifempty
}
EOF
    echo "=========================================================="
    echo "【日志管理提示】"
    echo "已生成日志截断配置文件: $CONF_FILE"
    echo "为确保超过 $LOG_MAX_SIZE 自动截断生效，请在 CentOS 7 上使用 root 执行一次以下命令建立软连接："
    echo "sudo ln -sf $CONF_FILE /etc/logrotate.d/silentfas"
    echo "=========================================================="
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    *)
        echo "用法: $0 {start|stop|restart}"
        exit 1
esac
