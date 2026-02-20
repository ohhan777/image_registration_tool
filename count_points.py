import argparse
import os
import glob


def count_points_in_txt(txt_path):
    count = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 3:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='L1G_tiles 라벨 포인트 카운터')
    parser.add_argument('-i', '--input', required=True, help='기준 디렉토리 경로')
    args = parser.parse_args()

    png_dir = os.path.join(args.input, 'L1G_tiles', 'tiles', 'png')
    if not os.path.isdir(png_dir):
        print(f"디렉토리를 찾을 수 없습니다: {png_dir}")
        return

    png_files = sorted(glob.glob(os.path.join(png_dir, '*.png')))
    if not png_files:
        print(f"PNG 파일이 없습니다: {png_dir}")
        return

    labeled_files = []
    total_points = 0

    for png_path in png_files:
        txt_path = os.path.splitext(png_path)[0] + '.txt'
        if os.path.exists(txt_path):
            n = count_points_in_txt(txt_path)
            labeled_files.append((os.path.basename(png_path), n))
            total_points += n

    if not labeled_files:
        print(f"라벨 파일(.txt)이 없습니다. (전체 PNG: {len(png_files)}개)")
        return

    print(f"{'파일명':<40} {'포인트 수':>10}")
    print('-' * 52)
    for name, n in labeled_files:
        print(f"{name:<40} {n:>10}")
    print('-' * 52)
    print(f"{'총 작업 파일':<40} {len(labeled_files):>10}개")
    print(f"{'총 포인트 수':<40} {total_points:>10}")
    print(f"{'미작업 파일':<40} {len(png_files) - len(labeled_files):>10}개")


if __name__ == '__main__':
    main()
