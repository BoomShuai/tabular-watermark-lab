# 贡献说明

通过独立分支和Pull Request提交修改，说明具体问题、修改后行为和验证结果。运行：

```bash
python -m unittest discover -s tests -v
python -I scripts/smoke_installed.py
python scripts/privacy_check.py --history
```

安装后的CLI烟雾检查需要先 `pip install .`。提交只包含源码、自写文档和合成样例生成逻辑。真实数据、密钥、报告、输入路径和账号配置留在被忽略的private/或outputs/。新增算法或指标需提供独立预期值测试，明确模拟、测量和复现的区别。

贡献按仓库许可证提供，不改变上游代码/数据的授权。CI通过后squash合并，合并后删除工作分支。
