from huggingface_hub import snapshot_download


def main():
    dataset_names = [
        "range3/narou20",
        "range3/wiki40b-ja",
        "range3/cc100-ja",
        "range3/wikipedia-ja-20230101",
        "range3/narou",
    ]

    for repo_id in dataset_names:
        snapshot_download(
            repo_id=repo_id, revision="main", repo_type="dataset", token=True
        )


if __name__ == "__main__":
    main()
