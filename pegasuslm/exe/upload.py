import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        metavar="username/repo",
        required=True,
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        metavar="model",
        help="model, dataset, or space",
    )
    parser.add_argument(
        "--path",
        type=str,
        metavar="/path/to/local/dir",
        help="local directory path to upload",
        required=True,
    )
    args = parser.parse_args()

    api = HfApi()
    api.upload_folder(
        folder_path=args.path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )


if __name__ == "__main__":
    main()
