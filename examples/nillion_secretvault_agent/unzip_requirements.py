import os
import sys
import zipfile
import tempfile
import requests

pkgurl = "https://s3.us-west-2.amazonaws.com/we.public/.requirements.zip"
pkgdir = "/tmp/agent-py-req"

# We want our path to look like [working_dir, serverless_requirements, ...]
sys.path.insert(1, pkgdir)

if not os.path.exists(pkgdir):
    tempdir = tempfile.mkdtemp()

    default_lambda_task_root = os.environ.get("LAMBDA_TASK_ROOT", os.getcwd())
    lambda_task_root = (
        os.getcwd()
        if os.environ.get("IS_LOCAL") == "true"
        else default_lambda_task_root
    )
    zip_requirements = os.path.join(tempdir, ".requirements.zip")

    response = requests.get(pkgurl, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with open(zip_requirements, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    zipfile.ZipFile(zip_requirements, "r").extractall(tempdir)
    os.rename(tempdir, pkgdir)  # Atomic
