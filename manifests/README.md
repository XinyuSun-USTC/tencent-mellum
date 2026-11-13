# 外部资产清单

- [results_raw_inventory.json](results_raw_inventory.json)：完整results中51501个文件的相对路径、大小、SHA256；原始根路径及本地tar路径。
- [local_backups.json](local_backups.json)：完整Git引用bundle、含未提交文件的source.tar、完整results.tar及迁移前git status文本的哈希与本地路径。
- [external_assets.json](external_assets.json)：只读模型配置/index及现有数据文件指纹、解释器位置；未重复读取全部模型weight payload，不声称该清单是全权重哈希。

这些是索引，不是大文件本身。机器内路径不能当成云端存储地址；首次Git推送不包含本地归档或权重。
