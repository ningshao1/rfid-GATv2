# -*- coding: utf-8 -*-
"""
RFID-GAT定位系统启动脚本
"""

import sys
import os
from src.main import main

if __name__ == "__main__":
    # 创建结果目录（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')

    # 运行主程序
    main()
