# SVG → WSD 转换器

将 potrace 生成的 SVG 文件转换为 EduEditor (WSD) 格式。

## 快速使用

### Windows 用户
从 [Releases](https://github.com/yourname/svg2wsd/releases) 下载最新的 `svg2wsd-windows.zip`，解压后：

```
svg2wsd.exe input.svg output.wsd
```

### 从源码运行
```bash
pip install pyinstaller
python svg2wsd.py input.svg output.wsd
```

## 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--color rainbow` | 按面积分配彩虹色 | ✓ |
| `--color single` | 单色填充 | |
| `--color svg` | 使用SVG自带fill颜色 | |
| `--fill-color "#ff6600"` | 单色填充颜色 | `#3366ff` |
| `--linewidth 80` | 轮廓线宽 (40=0.1mm) | `80` |
| `--no-outline` | 不绘制黑色轮廓 | |

## 示例

```bash
# 彩虹色
svg2wsd.exe butterfly.svg butterfly.wsd

# 纯红色填充
svg2wsd.exe butterfly.svg butterfly.wsd --color single --fill-color "#ff0000"

# 无轮廓
svg2wsd.exe butterfly.svg butterfly.wsd --no-outline
```

## 注意

- `template` 文件夹必须和 `svg2wsd.exe` 放在同一目录
- 支持 potrace 生成的 SVG 文件
- 输出可直接用 EduEditor 打开
