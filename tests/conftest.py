"""
Helper và tiện ích dùng chung cho tests
"""
import pathlib


def get_project_root():
    """Lấy thư mục gốc của project"""
    return pathlib.Path(__file__).parent.parent


def get_input_dir(subdir=""):
    """Lấy đường dẫn thư mục input"""
    root = get_project_root()
    if subdir:
        return root / "data" / "input" / subdir
    return root / "data" / "input"


def get_output_dir(subdir=""):
    """Lấy đường dẫn thư mục output"""
    root = get_project_root()
    if subdir:
        return root / "data" / "output" / subdir
    return root / "data" / "output"


def check_file_exists(filepath, description="File"):
    """Kiểm tra file có tồn tại và trả về kết quả"""
    exists = filepath.exists()
    status = "[PASS]" if exists else "[FAIL]"
    print(f"{status} {description}: {filepath}")
    return exists


def check_dir_exists(dirpath, description="Directory"):
    """Kiểm tra thư mục có tồn tại và trả về kết quả"""
    exists = dirpath.exists()
    status = "[PASS]" if exists else "[FAIL]"
    print(f"{status} {description}: {dirpath}")
    return exists


def get_files_by_pattern(directory, pattern="*"):
    """Lấy tất cả file khớp với pattern trong thư mục"""
    if not directory.exists():
        return []
    return list(directory.glob(pattern))


def count_files(directory, pattern="*"):
    """Đếm số file khớp với pattern trong thư mục"""
    return len(get_files_by_pattern(directory, pattern))
