#!/usr/bin/env python3
"""
图像 → WSD 转换器 (命令行版 v3)
支持格式: SVG, PNG, JPG, JPEG, BMP, GIF, WebP, TIFF
支持模式: 普通转换 (填充路径) / 几何转换 (识别直线、圆、矩形等)
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

    # 转换模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--normal', action='store_true',
                            help='普通转换模式 (默认，矢量化填充)')
    mode_group.add_argument('--geo', '--geometric', action='store_true', dest='geometric',
                            help='几何转换模式 (识别直线/圆/矩形/三角形等)')

    parser.add_argument('--color', default='rainbow',
                        choices=['rainbow', 'single', 'svg'],
                        help='填充颜色模式 (默认: rainbow)')
    parser.add_argument('--fill-color', default='#3366ff',
                        help='单色填充时的颜色 (#rrggbb)')
    parser.add_argument('--linewidth', type=int, default=80,
                        help='轮廓线宽 (WSD单位, 40=0.1mm, 默认80)')
    parser.add_argument('--no-outline', action='store_true',
                        help='不绘制黑色轮廓 (仅普通模式)')
    parser.add_argument('--flip-v', action='store_true',
                        help='垂直翻转输出')
    parser.add_argument('--width', type=int, default=0,
                        help='自定义输出宽度 (WSD单位)')
    parser.add_argument('--height', type=int, default=0,
                        help='自定义输出高度 (WSD单位)')
    parser.add_argument('--outdir', default='',
                        help='批量输出目录')
    parser.add_argument('--merge', action='store_true',
                        help='合并到同一个WSD的不同画布')
    parser.add_argument('--compound-mode', default='auto',
                        choices=['auto', 'split', 'merge'],
                        help='复合路径处理模式 (auto=自动, split=拆分, merge=合并, 默认auto)')

    # 普通转换参数
    normal_group = parser.add_argument_group('普通转换参数')
    normal_group.add_argument('--threshold', type=int, default=128,
                              help='图片二值化阈值 (10-245, 默认128)')
    normal_group.add_argument('--turdsize', type=int, default=2,
                              help='忽略的最小区域像素数 (默认2)')

    # 几何转换参数
    geo_group = parser.add_argument_group('几何转换参数')
    geo_group.add_argument('--min-area', type=int, default=50,
                           help='最小面积/长度 (像素, 默认50)')
    geo_group.add_argument('--epsilon', type=float, default=0.02,
                           help='近似精度 (0.005-0.05, 默认0.02)')

    args = parser.parse_args()

    # 确定转换模式
    is_geometric = args.geometric

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

    mode_name = "几何转换" if is_geometric else "普通转换"
    print(f"模式: {mode_name}")
    print(f"输入文件: {len(input_files)} 个")

    # 合并模式
    if args.merge:
        out_file = args.output or 'merged.wsd'
        if is_geometric:
            from svg2wsd_geo import convert_geo_to_wsd_multi
            result = convert_geo_to_wsd_multi(
                input_files, out_file,
                color_mode=args.color,
                linewidth=args.linewidth,
                fill_color=args.fill_color,
                flip_v=args.flip_v,
                custom_size=custom_size,
                min_area=args.min_area,
                epsilon_ratio=args.epsilon,
            )
        else:
            from svg2wsd_core import convert_to_wsd_multi
            result = convert_to_wsd_multi(
                input_files, out_file,
                color_mode=args.color,
                linewidth=args.linewidth,
                fill_color=args.fill_color,
                outline=not args.no_outline,
                flip_v=args.flip_v,
                custom_size=custom_size,
                img_threshold=args.threshold,
                img_turdsize=args.turdsize,
            )
        print(f"✓ 合并完成!")
        print(f"  画布数: {result['canvases']}")
        print(f"  输入文件: {result['files']} 个")
        print(f"  输出: {out_file}")
        print(f"  大小: {result['size']} 字节")
        sys.exit(0)

    # 判断单文件还是批量
    if len(input_files) == 1 and args.output and not os.path.isdir(args.output or ''):
        # 单文件转换
        in_file = input_files[0]
        out_file = args.output or os.path.splitext(in_file)[0] + '.wsd'

        if is_geometric:
            from svg2wsd_geo import convert_geo_to_wsd
            result = convert_geo_to_wsd(
                in_file, out_file,
                color_mode=args.color,
                linewidth=args.linewidth,
                fill_color=args.fill_color,
                flip_v=args.flip_v,
                custom_size=custom_size,
                min_area=args.min_area,
                epsilon_ratio=args.epsilon,
            )
            print(f"✓ 转换完成!")
            print(f"  输入: {in_file}")
            print(f"  输出: {out_file}")
            print(f"  大小: {result['size']} 字节")
            print(f"  形状数: {result['shapes']} 个")
            print(f"  形状类型: {', '.join(result['shape_types'])}")
            print(f"  对象数: {result['objects']} 个")
        else:
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
                compound_mode=args.compound_mode,
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
                if is_geometric:
                    from svg2wsd_geo import convert_geo_to_wsd
                    convert_geo_to_wsd(
                        in_file, out_file,
                        color_mode=args.color,
                        linewidth=args.linewidth,
                        fill_color=args.fill_color,
                        flip_v=args.flip_v,
                        custom_size=custom_size,
                        min_area=args.min_area,
                        epsilon_ratio=args.epsilon,
                    )
                else:
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
                        compound_mode=args.compound_mode,
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
