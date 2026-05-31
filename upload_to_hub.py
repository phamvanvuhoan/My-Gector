import modal

app    = modal.App("gector-to-hf")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/gector-data"

@app.function(
    image   = modal.Image.debian_slim().pip_install("huggingface_hub"),
    volumes = {MOUNT: volume},
    secrets = [modal.Secret.from_name("huggingface-secret")],
)
def upload_from_volume(repo_id: str, dir: str):
    import os
    from huggingface_hub import create_repo, upload_folder
    create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    upload_folder(folder_path=dir, repo_id=repo_id, repo_type="model")
    print(f"Uploaded → https://huggingface.co/{repo_id}")

@app.local_entrypoint()
def upload(
    repo_id: str,
    dir:     str = "/gector-data/checkpoints/stage3/last",
):
    upload_from_volume.remote(repo_id=repo_id, dir=dir)