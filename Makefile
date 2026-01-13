.PHONY: help compile compile-cuda compile-rocm compile-metal clean cache-info install-tools

# 默认目标
help:
	@echo "gllm-kernels 编译工具"
	@echo ""
	@echo "使用方法:"
	@echo "  make compile        - 编译所有平台 kernels（需要编译器）"
	@echo "  make compile-cuda   - 只编译 CUDA kernels"
	@echo "  make compile-rocm   - 只编译 ROCm kernels"
	@echo "  make compile-metal  - 只编译 Metal kernels（macOS only）"
	@echo "  make cache-info     - 显示缓存信息"
	@echo "  make clean          - 清理编译缓存"
	@echo ""
	@echo "Docker 编译（推荐）:"
	@echo "  make docker-cuda    - 使用 Docker 编译 CUDA kernels"
	@echo "  make docker-rocm    - 使用 Docker 编译 ROCm kernels"
	@echo ""

# 编译所有平台
compile:
	@echo "编译所有平台 kernels..."
	@./scripts/compile-kernels.sh

# 只编译 CUDA
compile-cuda:
	@echo "编译 CUDA kernels..."
	@bash -c ' \
		if command -v nvcc >/dev/null 2>&1; then \
			./scripts/compile-kernels.sh | grep -A 100 "CUDA"; \
		else \
			echo "错误: nvcc 未安装"; \
			echo "请运行: make install-cuda"; \
			exit 1; \
		fi \
	'

# 只编译 ROCm
compile-rocm:
	@echo "编译 ROCm kernels..."
	@bash -c ' \
		if command -v hipcc >/dev/null 2>&1; then \
			./scripts/compile-kernels.sh | grep -A 100 "ROCm"; \
		else \
			echo "错误: hipcc 未安装"; \
			echo "请运行: make install-rocm"; \
			exit 1; \
		fi \
	'

# 只编译 Metal（macOS）
compile-metal:
	@echo "编译 Metal kernels..."
	@bash -c ' \
		if [ "$$(uname)" = "Darwin" ]; then \
			./scripts/compile-kernels.sh | grep -A 100 "Metal"; \
		else \
			echo "错误: Metal 只能在 macOS 上编译"; \
			exit 1; \
		fi \
	'

# Docker 编译 CUDA
docker-cuda:
	@echo "使用 Docker 编译 CUDA kernels..."
	@docker build -t gllm-kernels-cuda -f docker/Dockerfile.cuda .
	@docker run --rm -v ~/.gsc:/output gllm-kernels-cuda

# Docker 编译 ROCm
docker-rocm:
	@echo "使用 Docker 编译 ROCm kernels..."
	@docker build -t gllm-kernels-rocm -f docker/Dockerfile.rocm .
	@docker run --rm -v ~/.gsc:/output gllm-kernels-rocm

# 显示缓存信息
cache-info:
	@echo "Kernel 缓存目录: ~/.gsc/gllm/kernels/"
	@echo ""
	@if [ -d ~/.gsc/gllm/kernels ]; then \
		echo "已编译的 kernels:"; \
		find ~/.gsc/gllm/kernels/ -type f | wc -l | xargs echo "  总文件数:"; \
		du -sh ~/.gsc/gllm/kernels/ | xargs echo "  总大小:"; \
		echo ""; \
		echo "文件列表:"; \
		find ~/.gsc/gllm/kernels/ -type f -exec ls -lh {} \; | awk '{print "  " $$9 " (" $$5 ")"}'; \
	else \
		echo "缓存目录不存在，请先运行: make compile"; \
	fi

# 清理缓存
clean:
	@echo "清理 kernel 缓存..."
	@rm -rf ~/.gsc/gllm/kernels/
	@echo "缓存已清理"

# 安装 CUDA 工具（Ubuntu/Debian）
install-cuda:
	@echo "安装 CUDA Toolkit..."
	@sudo apt-get update
	@sudo apt-get install -y cuda-toolkit-12
	@echo "CUDA 安装完成"
	@echo "请运行: source ~/.bashrc"

# 安装 ROCm 工具（Ubuntu/Debian）
install-rocm:
	@echo "安装 ROCm..."
	@sudo apt-get install -y rocm-dev hip-dev
	@echo "ROCm 安装完成"
	@echo "请运行: source ~/.bashrc"
