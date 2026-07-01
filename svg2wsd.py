#!/usr/bin/env python3
"""
图像 → WSD 转换器 (命令行版 v3)
支持格式: SVG, PNG, JPG, JPEG, BMP, GIF, WebP, TIFF
"""

import os
import sys
import glob

from svg2wsd_core import convert_to_wsd, is_supported_image


def main():
    import argparse
    parser = argparse.ArgumentParser(description='图像 → WSD 转换器')
    parser.add_argument('input', help='输入文件 (SVG/PNG/JPG/BMP等，支持通配符)')
    parser.add_argument('output', nargs='?', help='输出WSD文件 (单文件) 或 输出目录 (批量)')
    parser.add_argument('--color', default='rainbow',
                        choices=['rainbow', 'single', 'svg'],
                        help='填充颜色模式 (默认: rainbow)')
    parser.add_argument('--fill-color', default='#3366ff',
                        help='单色填充时的颜色 (#rrggbb)')
    parser.add_argument('--linewidth', type=int, default=80,
                        help='轮廓线宽 (WSD单位, 40=0.1mm, 默认80)')
    parser.add_argument('--no-outline', action='store_true',
                        help='不绘制黑色轮廓')
    parser.add_argument('--flip-v', action='store_true',
                        help='垂直翻转输出')
    parser.add_argument('--width', type=int, default=0,
                        help='自定义输出宽度 (WSD单位)')
    parser.add_argument('--height', type=int, default=0,
                        help='自定义输出高度 (WSD单位)')
    parser.add_argument('--outdir', default='',
                        help='批量输出目录')
    parser.add_argument('--threshold', type=int, default=128,
                        help='图片二值化阈值 (10-245, 默认128)')
    parser.add_argument('--turdsize', type=int, default=2,
                        help='忽略的最小区域像素数 (默认2)')
    args = parser.parse_args()

    custom_size = None
    if args.width > 0 and args.height > 0:
        custom_size = (args.width, args.height)

    # 收集输入文件
    input_files = []
    if '*' in args.input or '?' in args.input:
        input_files = sorted(glob.glob(args.input))
    elif os.path.isfile(args.input):
        input_files = [args.input]
    else:
        print(f"✗ 找不到文件: {args.input}")
        sys.exit(1)

    # 过滤支持的格式
    input_files = [f for f in input_files if is_supported_image(f)]
    if not input_files:
        print("✗ 没有找到支持的文件格式")
        sys.exit(1)

    # 判断单文件还是批量
    if len(input_files) == 1 and args.output and not os.path.isdir(args.output or ''):
        # 单文件转换
        in_file = input_files[0]
        out_file = args.output or os.path.splitext(in_file)[0] + '.wsd'
        result = convert_to_wsd(
            in_file, out_file,
            color_mode=args.color,
            linewidth=args.linewidth,
            fill_color=args.fill_color,
            outline=not args.no_outline,
            flip_v=args.flip_v,
            custom_size=custom_size,
            img_threshold=args.threshold,
            img_turdsize=args.turdsize,
        )
        print(f"✓ 转换完成!")
        print(f"  输入: {in_file} ({result['file_type']})")
        print(f"  输出: {out_file}")
        print(f"  大小: {result['size']} 字节")
        print(f"  子路径: {result['subpaths']} 个")
        print(f"  对象数: {result['objects']} 个")
    else:
        # 批量转换
        out_dir = args.outdir or args.output or '.'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        success = 0
        failed = []
        for in_file in input_files:
            base = os.path.splitext(os.path.basename(in_file))[0]
            out_file = os.path.join(out_dir, base + '.wsd')
            try:
                convert_to_wsd(
                    in_file, out_file,
                    color_mode=args.color,
                    linewidth=args.linewidth,
                    fill_color=args.fill_color,
                    outline=not args.no_outline,
                    flip_v=args.flip_v,
                    custom_size=custom_size,
                    img_threshold=args.threshold,
                    img_turdsize=args.turdsize,
                )
                success += 1
                print(f"  ✓ {base}.wsd")
            except Exception as e:
                failed.append((base, str(e)))
                print(f"  ✗ {base}: {e}")

        print(f"\n完成! 成功: {success}, 失败: {len(failed)}")
        print(f"输出目录: {out_dir}")


if __name__ == '__main__':
    main()
